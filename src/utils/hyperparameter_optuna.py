import numpy as np
import gc
import time
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

# --- RMSLE Scorer ---
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-5)
    y_true = np.maximum(y_true, 1e-5)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# --- GPU Stacking Ensemble ---
class GPUAcceleratedStackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, meta_model=None, n_folds=5, random_state=42, gpu_id=0):
        self.meta_model = meta_model if meta_model else ElasticNet(random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        self.gpu_id = gpu_id
        self.base_models = []
        self.base_models_trained = []

    def _initialize_models(self):
        self.base_models = [
            xgb.XGBRegressor(objective='reg:squaredlogerror', random_state=self.random_state,
                             tree_method='gpu_hist', gpu_id=self.gpu_id),
            lgb.LGBMRegressor(objective='regression', random_state=self.random_state,
                              device='gpu', gpu_platform_id=0, gpu_device_id=self.gpu_id),
            ctb.CatBoostRegressor(loss_function='RMSE', random_state=self.random_state,
                                  verbose=0, task_type='GPU', devices=f'{self.gpu_id}')
        ]

    def fit(self, X, y):
        if not self.base_models:
            self._initialize_models()

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                clone_model = clone(model)
                clone_model.fit(X_train, y_train)
                meta_features[val_idx, i] = clone_model.predict(X_val)
                del clone_model
                gc.collect()

            full_model = clone(model)
            full_model.fit(X, y)
            self.base_models_trained.append(full_model)
            gc.collect()

        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.base_models_trained])
        return self.meta_model.predict(meta_features)


# --- Optuna Hyperparameter Tuning ---
def gpu_accelerated_tuning_optuna(X_train, y_train, X_test=None, y_test=None, cv=5, gpu_id=0, n_trials=20):
    def tune_xgb(trial):
        params = {
            'objective': 'reg:squaredlogerror',
            'tree_method': 'gpu_hist',
            'gpu_id': gpu_id,
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'max_bin': trial.suggest_int('max_bin', 128, 512),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        return cross_val_score(model, X_train, y_train, scoring=rmsle_scorer, cv=cv, n_jobs=1).mean()

    def tune_lgb(trial):
        params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': gpu_id,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'max_bin': trial.suggest_int('max_bin', 128, 255),
            'random_state': 42
        }
        model = lgb.LGBMRegressor(**params)
        return cross_val_score(model, X_train, y_train, scoring=rmsle_scorer, cv=cv, n_jobs=1).mean()

    def tune_ctb(trial):
        params = {
            'loss_function': 'RMSE',
            'task_type': 'GPU',
            'devices': f'{gpu_id}',
            'depth': trial.suggest_int('depth', 8, 16),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4.0, 16.0),
            'gpu_ram_part': trial.suggest_float('gpu_ram_part', 0.3, 0.7),
            'random_state': 42,
            'verbose': 0
        }
        model = ctb.CatBoostRegressor(**params)
        return cross_val_score(model, X_train, y_train, scoring=rmsle_scorer, cv=cv, n_jobs=1).mean()

    def optimize_model(name, objective_func):
        print(f"\nTuning {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_func, n_trials=n_trials)
        print(f"{name} Best RMSLE: {-study.best_value:.5f}")
        print(f"{name} Best Params: {study.best_params}")
        return study.best_params

    xgb_best_params = optimize_model("XGBoost", tune_xgb)
    lgb_best_params = optimize_model("LightGBM", tune_lgb)
    ctb_best_params = optimize_model("CatBoost", tune_ctb)

    best_xgb = xgb.XGBRegressor(**xgb_best_params)
    best_lgb = lgb.LGBMRegressor(**lgb_best_params)
    best_ctb = ctb.CatBoostRegressor(**ctb_best_params)

    best_xgb.fit(X_train, y_train)
    best_lgb.fit(X_train, y_train)
    best_ctb.fit(X_train, y_train)

    gc.collect()

    # Meta-model tuning
    def tune_meta(trial):
        alpha = trial.suggest_float('alpha', 0.0001, 1.0)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.8)
        meta_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        ensemble = GPUAcceleratedStackingEnsemble(meta_model=meta_model, n_folds=cv, random_state=42, gpu_id=gpu_id)
        ensemble._initialize_models()
        ensemble.base_models = [best_xgb, best_lgb, best_ctb]
        return cross_val_score(ensemble, X_train, y_train, scoring=rmsle_scorer, cv=cv, n_jobs=1).mean()

    print("\nTuning Meta-Model (ElasticNet)...")
    meta_study = optuna.create_study(direction='maximize')
    meta_study.optimize(tune_meta, n_trials=10)

    final_meta_model = ElasticNet(**meta_study.best_params, random_state=42)
    best_ensemble = GPUAcceleratedStackingEnsemble(meta_model=final_meta_model, n_folds=cv,
                                                   random_state=42, gpu_id=gpu_id)
    best_ensemble._initialize_models()
    best_ensemble.base_models = [best_xgb, best_lgb, best_ctb]
    best_ensemble.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        y_pred = best_ensemble.predict(X_test)
        test_rmsle = rmsle(y_test, y_pred)
        print(f"\nTest RMSLE (Ensemble): {test_rmsle:.5f}")

    return best_ensemble, [('xgb', best_xgb), ('lgb', best_lgb), ('ctb', best_ctb)]

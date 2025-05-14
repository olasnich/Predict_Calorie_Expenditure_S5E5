import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import gc
import time
import optuna
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback, CatBoostPruningCallback

# Custom RMSLE scorer
def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    y_pred = np.maximum(y_pred, 1e-5)
    y_true = np.maximum(y_true, 1e-5)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


class GPUAcceleratedStackingEnsemble(BaseEstimator, RegressorMixin):
    """GPU-accelerated stacking ensemble that efficiently utilizes GPU resources"""

    def __init__(self, meta_model=None, n_folds=5, random_state=42, gpu_id=0):
        self.meta_model = meta_model if meta_model else ElasticNet(random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        self.gpu_id = gpu_id

        # Will be initialized later
        self.xgb_model = None
        self.lgb_model = None
        self.ctb_model = None

        self.base_models = []
        self.base_models_trained = []

    def _initialize_models(self):
        # Initialize with GPU settings
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squaredlogerror',
            random_state=self.random_state,
            tree_method='gpu_hist',  # GPU acceleration
            gpu_id=self.gpu_id
        )

        self.lgb_model = lgb.LGBMRegressor(
            objective='regression',
            random_state=self.random_state,
            device='gpu',  # GPU acceleration
            gpu_platform_id=0,
            gpu_device_id=self.gpu_id
        )

        self.ctb_model = ctb.CatBoostRegressor(
            loss_function='RMSE',
            random_state=self.random_state,
            verbose=0,
            task_type='GPU',  # GPU acceleration
            devices=f'{self.gpu_id}'
        )

        self.base_models = [self.xgb_model, self.lgb_model, self.ctb_model]

    def fit(self, X, y):
        if len(self.base_models) == 0:
            self._initialize_models()

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            start_time = time.time()
            print(f"Training model {i+1}/{len(self.base_models)} ({model.__class__.__name__})")

            # Generate out-of-fold predictions
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                clone_model = clone(model)
                clone_model.fit(X_train, y_train)
                meta_features[val_idx, i] = clone_model.predict(X_val)

                del clone_model
                gc.collect()

            # Fit on full dataset
            model_copy = clone(model)
            model_copy.fit(X, y)
            self.base_models_trained.append(model_copy)

            print(f"Model {i+1} training completed in {time.time() - start_time:.2f} seconds")
            gc.collect()

        # Train meta-model
        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_trained
        ])
        return self.meta_model.predict(meta_features)


def xgboost_objective(trial, X_train, y_train, cv, gpu_id):
    """Objective function for XGBoost optimization"""
    param = {
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'objective': 'reg:squaredlogerror',
        'tree_method': 'gpu_hist',
        'gpu_id': gpu_id,
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**param)
    
    # Use pruning callback for early stopping
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
    
    # Define validation data for pruning
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**param)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=0
        )
        
        y_pred = model.predict(X_fold_val)
        fold_score = rmsle(y_fold_val, y_pred)
        scores.append(fold_score)
        
    return np.mean(scores)


def lightgbm_objective(trial, X_train, y_train, cv, gpu_id):
    """Objective function for LightGBM optimization"""
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'max_bin': trial.suggest_int('max_bin', 128, 255),
        'objective': 'regression',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': gpu_id,
        'random_state': 42
    }
    
    # Implement pruning callback
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l2")
    
    # Use cross validation to get robust estimates
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric="l2",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=0
        )
        
        y_pred = model.predict(X_fold_val)
        fold_score = rmsle(y_fold_val, y_pred)
        scores.append(fold_score)
        
    return np.mean(scores)


def catboost_objective(trial, X_train, y_train, cv, gpu_id):
    """Objective function for CatBoost optimization"""
    param = {
        'depth': trial.suggest_int('depth', 8, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4, 16),
        'gpu_ram_part': trial.suggest_float('gpu_ram_part', 0.3, 0.7),
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'devices': f'{gpu_id}',
        'random_seed': 42,
        'verbose': 0
    }
    
    # Implement pruning callback
    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "RMSE")
    
    # Use cross validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = ctb.CatBoostRegressor(**param)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=0
        )
        
        y_pred = model.predict(X_fold_val)
        fold_score = rmsle(y_fold_val, y_pred)
        scores.append(fold_score)
        
    return np.mean(scores)


def meta_model_objective(trial, X_train, y_train, base_models, cv, gpu_id):
    """Objective function for meta model optimization"""
    # Choose meta model type
    model_type = trial.suggest_categorical('model_type', ['ElasticNet', 'Ridge', 'Lasso'])
    
    if model_type == 'ElasticNet':
        meta_model = ElasticNet(
            alpha=trial.suggest_float('alpha', 0.0001, 1.0),
            l1_ratio=trial.suggest_float('l1_ratio', 0.1, 0.8),
            random_state=42
        )
    elif model_type == 'Ridge':
        meta_model = Ridge(
            alpha=trial.suggest_float('alpha', 0.0001, 10.0),
            random_state=42
        )
    else:  # Lasso
        meta_model = Lasso(
            alpha=trial.suggest_float('alpha', 0.0001, 1.0),
            random_state=42
        )
    
    # Initialize and fit stacking ensemble
    stacking = GPUAcceleratedStackingEnsemble(
        meta_model=meta_model,
        n_folds=cv,
        random_state=42,
        gpu_id=gpu_id
    )
    
    stacking._initialize_models()
    stacking.base_models = [model for model in base_models]
    
    # Use cross validation to evaluate
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Clone the stacking ensemble to avoid data leakage
        stacking_clone = clone(stacking)
        stacking_clone.fit(X_fold_train, y_fold_train)
        
        y_pred = stacking_clone.predict(X_fold_val)
        fold_score = rmsle(y_fold_val, y_pred)
        scores.append(fold_score)
        
        del stacking_clone
        gc.collect()
    
    return np.mean(scores)


def gpu_accelerated_tuning_with_optuna(X_train, y_train, X_test=None, y_test=None, cv=5, gpu_id=0, n_trials=20):
    """
    GPU-accelerated hyperparameter tuning with Optuna for automated optimization
    and pruning of underperforming trials
    """
    best_models = []
    data_size_mb = X_train.nbytes / (1024 * 1024)
    print(f"Dataset size: {data_size_mb:.2f} MB")
    
    # Convert numpy arrays to correct format if needed
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    
    # Create Optuna study for XGBoost
    print("Tuning XGBoost with GPU acceleration and Optuna...")
    start_time = time.time()
    
    study_xgb = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="xgboost_optimization"
    )
    
    study_xgb.optimize(
        lambda trial: xgboost_objective(trial, X_train, y_train, cv, gpu_id),
        n_trials=n_trials
    )
    
    # Get best parameters and create model
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({
        'objective': 'reg:squaredlogerror',
        'tree_method': 'gpu_hist',
        'gpu_id': gpu_id,
        'random_state': 42
    })
    
    best_xgb = xgb.XGBRegressor(**best_xgb_params)
    best_xgb.fit(X_train, y_train)
    
    print(f"Best XGBoost RMSLE: {study_xgb.best_value:.5f}")
    print(f"Best XGBoost params: {study_xgb.best_params}")
    print(f"XGBoost tuning completed in {time.time() - start_time:.2f} seconds")
    best_models.append(('xgb', best_xgb))
    
    # Optuna visualization (optional)
    # optuna.visualization.plot_optimization_history(study_xgb)
    # optuna.visualization.plot_param_importances(study_xgb)
    
    del study_xgb
    gc.collect()
    
    # Create Optuna study for LightGBM
    print("\nTuning LightGBM with GPU acceleration and Optuna...")
    start_time = time.time()
    
    study_lgb = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="lightgbm_optimization"
    )
    
    study_lgb.optimize(
        lambda trial: lightgbm_objective(trial, X_train, y_train, cv, gpu_id),
        n_trials=n_trials
    )
    
    # Get best parameters and create model
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({
        'objective': 'regression',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': gpu_id,
        'random_state': 42
    })
    
    best_lgb = lgb.LGBMRegressor(**best_lgb_params)
    best_lgb.fit(X_train, y_train)
    
    print(f"Best LightGBM RMSLE: {study_lgb.best_value:.5f}")
    print(f"Best LightGBM params: {study_lgb.best_params}")
    print(f"LightGBM tuning completed in {time.time() - start_time:.2f} seconds")
    best_models.append(('lgb', best_lgb))
    
    del study_lgb
    gc.collect()
    
    # Create Optuna study for CatBoost
    print("\nTuning CatBoost with GPU acceleration and Optuna...")
    start_time = time.time()
    
    study_ctb = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="catboost_optimization"
    )
    
    study_ctb.optimize(
        lambda trial: catboost_objective(trial, X_train, y_train, cv, gpu_id),
        n_trials=n_trials
    )
    
    # Get best parameters and create model
    best_ctb_params = study_ctb.best_params
    best_ctb_params.update({
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'devices': f'{gpu_id}',
        'random_seed': 42,
        'verbose': 0
    })
    
    best_ctb = ctb.CatBoostRegressor(**best_ctb_params)
    best_ctb.fit(X_train, y_train)
    
    print(f"Best CatBoost RMSLE: {study_ctb.best_value:.5f}")
    print(f"Best CatBoost params: {study_ctb.best_params}")
    print(f"CatBoost tuning completed in {time.time() - start_time:.2f} seconds")
    best_models.append(('ctb', best_ctb))
    
    del study_ctb
    gc.collect()
    
    # Now tune the meta-model
    print("\nTuning Stacking Ensemble with Optuna...")
    start_time = time.time()
    
    # Prepare best base models for meta model optimization
    base_models = [best_xgb, best_lgb, best_ctb]
    
    study_meta = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="meta_model_optimization"
    )
    
    study_meta.optimize(
        lambda trial: meta_model_objective(trial, X_train, y_train, base_models, cv, gpu_id),
        n_trials=10
    )
    
    # Create final stacking ensemble with best meta model
    best_meta_params = study_meta.best_params
    model_type = best_meta_params.pop('model_type')
    
    if model_type == 'ElasticNet':
        best_meta_model = ElasticNet(random_state=42, **best_meta_params)
    elif model_type == 'Ridge':
        best_meta_model = Ridge(random_state=42, **best_meta_params)
    else:  # Lasso
        best_meta_model = Lasso(random_state=42, **best_meta_params)
    
    final_ensemble = GPUAcceleratedStackingEnsemble(
        meta_model=best_meta_model,
        n_folds=cv,
        random_state=42,
        gpu_id=gpu_id
    )
    
    final_ensemble._initialize_models()
    final_ensemble.base_models = base_models
    final_ensemble.fit(X_train, y_train)
    
    print(f"Best Meta-Model: {model_type}")
    print(f"Best Meta-Model params: {best_meta_params}")
    print(f"Best Stacking Ensemble RMSLE: {study_meta.best_value:.5f}")
    print(f"Meta-Model tuning completed in {time.time() - start_time:.2f} seconds")
    
    del study_meta
    gc.collect()
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
            
        y_pred = final_ensemble.predict(X_test)
        test_rmsle = rmsle(y_test, y_pred)
        print(f"\nEnsemble Test RMSLE: {test_rmsle:.5f}")
        
        for name, model in best_models:
            y_pred = model.predict(X_test)
            test_rmsle = rmsle(y_test, y_pred)
            print(f"{name} Test RMSLE: {test_rmsle:.5f}")
    
    return final_ensemble, best_models


# Example usage
if __name__ == "__main__":
    # Sample data (replace with your own)
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Run optimization
    best_ensemble, best_models = gpu_accelerated_tuning_with_optuna(
        X_train, y_train, X_test, y_test, 
        cv=3,  # Using 3 folds for faster example
        gpu_id=0,
        n_trials=5  # Using just 5 trials for demonstration
    )
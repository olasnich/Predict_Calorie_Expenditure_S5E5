import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import gc
from scipy.stats import uniform, randint
import time

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
        self.meta_model = meta_model if meta_model else ElasticNet(
            random_state=random_state)
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

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            start_time = time.time()
            print(
                f"Training model {i+1}/{len(self.base_models)} ({model.__class__.__name__})")

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

            print(
                f"Model {i+1} training completed in {time.time() - start_time:.2f} seconds")
            gc.collect()

        # Train meta-model
        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_trained
        ])
        return self.meta_model.predict(meta_features)


def gpu_accelerated_tuning(X_train, y_train, X_test=None, y_test=None, cv=5, gpu_id=0):
    """
    GPU-accelerated hyperparameter tuning with automated batch size selection
    and memory management for optimal performance
    """
    best_models = []

    # Calculate appropriate batch sizes based on dataset size and available GPU memory
    # These are examples and should be adjusted based on your GPU memory
    data_size_mb = X_train.nbytes / (1024 * 1024)
    print(f"Dataset size: {data_size_mb:.2f} MB")

    # Parameter distributions
    xgb_param_dist = {
        'max_depth': randint(5, 30),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(500, 3000),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 5),
        # GPU-specific parameters
        'max_bin': randint(128, 512),  # Controls GPU memory usage
        'gpu_id': [gpu_id]
    }

    lgb_param_dist = {
        'num_leaves': randint(20, 100),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(500, 3000),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1),
        # GPU-specific parameters
        'device': ['gpu'],
        'gpu_platform_id': [0],
        'gpu_device_id': [gpu_id],
        'max_bin': randint(128, 255)  # Controls GPU memory usage
    }

    ctb_param_dist = {
        'depth': randint(8, 16),
        'learning_rate': uniform(0.01, 0.5),
        'iterations': randint(500, 3000),
        'l2_leaf_reg': uniform(4, 16),
        # GPU-specific parameters
        'task_type': ['GPU'],
        'devices': [f'{gpu_id}'],
        'gpu_ram_part': uniform(0.3, 0.7)  # Portion of GPU memory to use
    }

    # Tune XGBoost with GPU
    print("Tuning XGBoost with GPU acceleration...")
    start_time = time.time()
    xgb_model = xgb.XGBRegressor(
        objective='reg:squaredlogerror',
        random_state=42,
        tree_method='gpu_hist',  # GPU algorithm
        gpu_id=gpu_id
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_dist,
        n_iter=20,
        scoring=rmsle_scorer,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=1  # Use 1 for GPU to avoid conflicts
    )

    xgb_search.fit(X_train, y_train)
    print(f"Best XGBoost RMSLE: {-xgb_search.best_score_:.5f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    print(
        f"XGBoost tuning completed in {time.time() - start_time:.2f} seconds")
    best_models.append(('xgb', xgb_search.best_estimator_))

    best_xgb = clone(xgb_search.best_estimator_)
    del xgb_search, xgb_model
    gc.collect()

    # Tune LightGBM with GPU
    print("\nTuning LightGBM with GPU acceleration...")
    start_time = time.time()
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        random_state=42,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=gpu_id
    )

    lgb_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=lgb_param_dist,
        n_iter=20,
        scoring=rmsle_scorer,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=1  # Use 1 for GPU to avoid conflicts
    )

    lgb_search.fit(X_train, y_train)
    print(f"Best LightGBM RMSLE: {-lgb_search.best_score_:.5f}")
    print(f"Best LightGBM params: {lgb_search.best_params_}")
    print(
        f"LightGBM tuning completed in {time.time() - start_time:.2f} seconds")
    best_models.append(('lgb', lgb_search.best_estimator_))

    best_lgb = clone(lgb_search.best_estimator_)
    del lgb_search, lgb_model
    gc.collect()

    # Tune CatBoost with GPU
    print("\nTuning CatBoost with GPU acceleration...")
    start_time = time.time()
    ctb_model = ctb.CatBoostRegressor(
        loss_function='RMSE',
        random_state=42,
        verbose=0,
        task_type='GPU',
        devices=f'{gpu_id}'
    )

    ctb_search = RandomizedSearchCV(
        estimator=ctb_model,
        param_distributions=ctb_param_dist,
        n_iter=20,
        scoring=rmsle_scorer,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=1  # Use 1 for GPU to avoid conflicts
    )

    ctb_search.fit(X_train, y_train)
    print(f"Best CatBoost RMSLE: {-ctb_search.best_score_:.5f}")
    print(f"Best CatBoost params: {ctb_search.best_params_}")
    print(
        f"CatBoost tuning completed in {time.time() - start_time:.2f} seconds")
    best_models.append(('ctb', ctb_search.best_estimator_))

    best_ctb = clone(ctb_search.best_estimator_)
    del ctb_search, ctb_model
    gc.collect()

    # Now tune the meta-model (CPU-based since these are simple models)
    print("\nTuning Stacking Ensemble...")

    meta_models = [
        {
            'name': 'ElasticNet',
            'model': ElasticNet(random_state=42),
            'param_dist': {
                'alpha': uniform(0.0001, 1.0),
                'l1_ratio': uniform(0.1, 0.8)
            }
        },
        {
            'name': 'Ridge',
            'model': Ridge(random_state=42),
            'param_dist': {
                'alpha': uniform(0.0001, 10.0)
            }
        },
        {
            'name': 'Lasso',
            'model': Lasso(random_state=42),
            'param_dist': {
                'alpha': uniform(0.0001, 1.0)
            }
        }
    ]

    best_score = float('inf')
    best_meta_model = None
    best_ensemble = None

    for meta_info in meta_models:
        print(f"\nTesting {meta_info['name']} as meta-model...")
        start_time = time.time()

        stacking = GPUAcceleratedStackingEnsemble(
            meta_model=meta_info['model'],
            n_folds=cv,
            random_state=42,
            gpu_id=gpu_id
        )

        stacking._initialize_models()
        stacking.base_models = [best_xgb, best_lgb, best_ctb]

        param_dist = {f'meta_model__{k}': v for k,
                      v in meta_info['param_dist'].items()}

        search = RandomizedSearchCV(
            estimator=stacking,
            param_distributions=param_dist,
            n_iter=10,
            scoring=rmsle_scorer,
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=1  # Use 1 for GPU
        )

        search.fit(X_train, y_train)
        current_score = -search.best_score_

        print(f"{meta_info['name']} best RMSLE: {current_score:.5f}")
        print(f"{meta_info['name']} best params: {search.best_params_}")
        print(
            f"{meta_info['name']} tuning completed in {time.time() - start_time:.2f} seconds")

        if current_score < best_score:
            best_score = current_score
            best_meta_model = meta_info['name']
            best_ensemble = search.best_estimator_

        del search, stacking
        gc.collect()

    print(f"\nBest Meta-Model: {best_meta_model}")
    print(f"Best Stacking Ensemble RMSLE: {best_score:.5f}")

    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        y_pred = best_ensemble.predict(X_test)
        test_rmsle = rmsle(y_test, y_pred)
        print(f"Test RMSLE: {test_rmsle:.5f}")

        for name, model in best_models:
            y_pred = model.predict(X_test)
            test_rmsle = rmsle(y_test, y_pred)
            print(f"{name} Test RMSLE: {test_rmsle:.5f}")

    return best_ensemble, best_models

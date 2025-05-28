import optuna
import numpy as np
import os
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib
from utils.feature_engineering import create_features

# Ensure output directory exists
os.makedirs("plot", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global variables for KFold
N_SPLITS = 50
RANDOM_STATE = 42


def objective(trial):
    """Objective function for Optuna optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'num_leaves': trial.suggest_int('num_leaves', 32, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'device': 'gpu',  # Use GPU if available
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    model = LGBMRegressor(**params)

    # Use KFold for optimization as well
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y, cv=kf,
                             scoring='neg_mean_squared_error', n_jobs=1)

    return -scores.mean()


def load_data(path):
    """Load and preprocess the data."""
    df = pd.read_csv(path)
    df = create_features(df)

    X = df.drop(columns=["id", "Calories"])
    y = np.log1p(df['Calories'])  # Log transform the target

    # Identify categorical features - in this case just "Sex"
    numerical_features = [col for col in X.columns if col not in ["Sex"]]
    categorical_features = ["Sex"]

    # Create a preprocessor for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="passthrough"  # Keep categorical features as is
    ).fit(X)

    X_train = preprocessor.transform(X)

    return X_train, y, df, categorical_features, preprocessor


def train_kfold_ensemble(X_train, y, best_params):
    """Train ensemble of models using KFold cross-validation."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = []
    oof_predictions = np.zeros(len(y))

    print(f"Training {N_SPLITS} models with KFold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1}/{N_SPLITS}")

        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create model with best parameters
        model = LGBMRegressor(
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            verbose=-1,
            random_state=RANDOM_STATE + fold,  # Different seed for each fold
            **best_params
        )

        # Train the model
        model.fit(X_fold_train, y_fold_train)

        # Store out-of-fold predictions
        oof_predictions[val_idx] = model.predict(X_fold_val)

        # Store the trained model
        models.append(model)

    # Calculate OOF score
    oof_score = np.sqrt(mean_squared_log_error(
        np.expm1(y), np.expm1(oof_predictions)))
    print(f"Out-of-fold RMSLE: {oof_score:.6f}")

    return models, oof_predictions


def make_ensemble_predictions(models, X):
    """Make predictions using ensemble of models."""
    predictions = np.zeros((len(X), len(models)))

    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X)

    # Return mean of all model predictions
    return predictions.mean(axis=1)


def main():
    """Main execution function."""
    global X_train, y  # Make these available to the objective function

    # Load and process data
    X_train, y, df, cat_features, preprocessor = load_data("data/train.csv")

    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print(f"Best MSLE: {-study.best_value}")

    # Train ensemble of models using KFold
    models, oof_predictions = train_kfold_ensemble(X_train, y, best_params)

    # Save all models
    print("Saving trained models...")
    for i, model in enumerate(models):
        joblib.dump(model, f'models/lightgbm_model_fold_{i}.pkl')

    # Save ensemble information
    model_info = {
        'n_models': len(models),
        'best_params': best_params,
        'oof_score': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_predictions))),
        'cat_features': cat_features
    }
    joblib.dump(model_info, 'models/lightgbm_ensemble_info.pkl')

    # Generate visualization plots
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances

        history_plot = plot_optimization_history(study)
        importance_plot = plot_param_importances(study)

        print("Optimization history and parameter importance plots saved.")
        history_plot.write_html("plot/lightgbm_optimization_history.html")
        importance_plot.write_html("plot/lightgbm_parameter_importance.html")

    except ImportError:
        print(
            "Optuna visualization tools not available. Install with: pip install optuna[visualization]")

    # Generate training predictions using ensemble
    print("Generating training predictions...")
    train_predictions = make_ensemble_predictions(models, X_train)
    out = df[['id']].copy()
    out['calories'] = train_predictions
    out.to_csv('data/lightgbm_train_pred.csv', index=False)

    # Generate test predictions using ensemble
    print("Generating test predictions...")
    X_submission = pd.read_csv("data/test.csv")
    submission_ids = X_submission[["id"]].copy()

    # Apply same feature engineering to test data
    X_submission_features = create_features(X_submission.drop(columns=["id"]))

    # Apply same transformations as training data
    X_submission = preprocessor.transform(X_submission_features)

    # Generate ensemble predictions
    test_predictions = make_ensemble_predictions(models, X_submission)

    # Convert back from log scale and clip values
    submission_ids["Calories"] = np.clip(np.expm1(test_predictions), 1, 314)
    submission_ids.to_csv("data/lightgbm_submission.csv", index=False)

    print(f"Ensemble training complete! Used {len(models)} models.")
    print(f"Final out-of-fold RMSLE: {model_info['oof_score']:.6f}")


def load_ensemble_models():
    """Helper function to load all trained models."""
    info = joblib.load('models/lightgbm_ensemble_info.pkl')
    models = []
    for i in range(info['n_models']):
        model = joblib.load(f'models/lightgbm_model_fold_{i}.pkl')
        models.append(model)
    return models, info


if __name__ == "__main__":
    main()

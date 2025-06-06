import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, KFold
import os
import pandas as pd
import numpy as np
from utils.feature_engineering import create_features
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib

# Ensure output directory exists
os.makedirs("plot", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global variables for KFold
N_SPLITS = 50
RANDOM_STATE = 42


def objective(trial):
    """Objective function for Optuna optimization."""
    params = {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': RANDOM_STATE
    }

    model = XGBRegressor(**params, enable_categorical=True)

    # Use KFold for optimization as well
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y, cv=kf,
                             scoring='neg_mean_squared_error', n_jobs=1)

    return -scores.mean()


def load_data(path):
    """Load and preprocess the data."""
    df = pd.read_csv("data/train.csv")
    df = create_features(df)

    X = df.drop(columns=["id", "Calories"])
    y = np.log1p(df['Calories'])

    numerical_features = [col for col in X.columns if col not in ["Sex"]]
    categorical_features = ["Sex"]

    # Create a ColumnTransformer to apply different preprocessing strategies
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="passthrough"
    ).fit(X)

    X_train = preprocessor.transform(X)

    transformed_columns = numerical_features + ['Sex']
    X_train = pd.DataFrame(X_train, columns=transformed_columns)
    
    # Ensure Sex column is categorical
    X_train['Sex'] = X_train['Sex'].astype('category')

    return X_train, y, df, preprocessor


def train_kfold_ensemble(X_train, y, best_params):
    """Train ensemble of models using KFold cross-validation."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = []
    oof_predictions = np.zeros(len(y))

    print(f"Training {N_SPLITS} models with KFold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1}/{N_SPLITS}")

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        # Create model with best parameters
        model = XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            enable_categorical=True,
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
    global X_train, y  # Ensure X_train and y are defined before calling optimization

    X_train, y, df, preprocessor = load_data("data/train.csv")

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
        joblib.dump(model, f'models/xgb_model_fold_{i}.pkl')

    # Save the list of model paths for easy loading later
    model_info = {
        'n_models': len(models),
        'best_params': best_params,
        'oof_score': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_predictions)))
    }
    joblib.dump(model_info, 'models/ensemble_info.pkl')

    # Generate visualization plots
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances

        history_plot = plot_optimization_history(study)
        importance_plot = plot_param_importances(study)

        print("Optimization history and parameter importance plots saved.")
        history_plot.write_html("plot/optimization_history.html")
        importance_plot.write_html("plot/parameter_importance.html")

    except ImportError:
        print(
            "Optuna visualization tools not available. Install with: pip install optuna[visualization]")

    # Generate training predictions using ensemble
    print("Generating training predictions...")
    train_predictions = make_ensemble_predictions(models, X_train)
    out = df[['id']].copy()
    out['calories'] = train_predictions
    out.to_csv('data/xgb_train_pred.csv', index=False)

    # Generate test predictions using ensemble
    print("Generating test predictions...")
    test_df = pd.read_csv("data/test.csv")
    submission_ids = test_df[["id"]].copy()

    # Apply same feature engineering to test data
    X_test_features = create_features(test_df.drop(columns=["id"]))
    X_test_features['Sex'] = X_test_features['Sex'].map({1: 'female', 0: 'male'})
    X_test_features['Sex'] = X_test_features['Sex'].astype('category')

    # Apply same transformations as training data
    X_test_transformed = preprocessor.transform(X_test_features)
    
    # Create DataFrame with same column structure as training
    numerical_features = [col for col in X_test_features.columns if col not in ["Sex"]]
    transformed_columns = numerical_features + ['Sex']
    X_test = pd.DataFrame(X_test_transformed, columns=transformed_columns)
    X_test['Sex'] = X_test['Sex'].astype('category')

    test_predictions = make_ensemble_predictions(models, X_test)
    submission_ids["Calories"] = np.clip(np.expm1(test_predictions), 1, 314)
    submission_ids.to_csv("data/xgb_submission.csv", index=False)

    print(f"Ensemble training complete! Used {len(models)} models.")
    print(f"Final out-of-fold RMSLE: {model_info['oof_score']:.6f}")


def load_ensemble_models():
    """Helper function to load all trained models."""
    info = joblib.load('models/ensemble_info.pkl')
    models = []
    for i in range(info['n_models']):
        model = joblib.load(f'models/xgb_model_fold_{i}.pkl')
        models.append(model)
    return models, info


if __name__ == "__main__":
    main()

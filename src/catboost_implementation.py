import optuna
import numpy as np
import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib
from utils.feature_engineering import create_features
# Ensure output directory exists
os.makedirs("plot", exist_ok=True)


def objective(trial):
    """Objective function for Optuna optimization."""
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        'task_type': 'GPU',  # Use GPU if available
        'devices': '0',      # GPU device index
    }

    model = CatBoostRegressor(**params, verbose=False)

    scores = cross_val_score(model, X_train, y, cv=5,
                             scoring='neg_mean_squared_error', n_jobs=1)

    return -scores.mean()


def load_data(path):
    # Load your data
    df = pd.read_csv(path)

    # Assuming create_features is a function that adds/transforms features

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
        remainder="passthrough"  # Keep categorical features as is for CatBoost
    ).fit(X)

    X_train = preprocessor.transform(X)

    # Apply polynomial features
    #poly = PolynomialFeatures(2)
    #X_train = poly.fit_transform(X_train)

    return X_train, y, df, categorical_features, preprocessor


def main():
    """Main execution function."""
    global X_train, y  # Make these available to the objective function

    # Load and process data
    X_train, y, df, cat_features, preprocessor = load_data(
        "data/train.csv")

    # Identify categorical feature indices after polynomial transformation
    # Note: For CatBoost, we'd typically pass original categorical features directly,
    # but since we're keeping the polynomial transformation for consistency,
    # we'll need to handle this differently

    # Create and optimize study
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print(f"Best MSLE: {-study.best_value}")

    # Create and train best model
    best_model = CatBoostRegressor(
        task_type='GPU', devices='0', verbose=False, **best_params)
    best_model.fit(X_train, y)

    # Save visualization plots if possible
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

    # Save model
    joblib.dump(best_model, 'models/catboost_model.pkl')

    # Generate predictions
    y_pred = best_model.predict(X_train)
    out = df[['id']]
    out['calories'] = y_pred
    out.to_csv(
        'data/catboost_train_pred.csv', index=False)

    # Process test data and create submission
    X_submission = pd.read_csv("data/test.csv")
    out = X_submission[["id"]].copy()

    # Apply same feature engineering to test data
    X_submission_features = create_features(X_submission.drop(columns=["id"]))

    # Apply same transformations as training data
    X_submission = preprocessor.transform(X_submission_features)
    #X_submission = poly.transform(X_submission)

    # Predict and convert back from log scale
    out["Calories"] = np.clip(np.expm1(best_model.predict(X_submission)), 1, 314)
    out.to_csv("data/catboost_submission.csv", index=False)


if __name__ == "__main__":
    main()

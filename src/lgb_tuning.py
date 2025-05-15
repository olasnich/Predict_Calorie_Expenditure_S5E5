import optuna
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
import os
import pandas as pd
import numpy as np
from utils.feature_engineering import create_features
from sklearn.preprocessing import StandardScaler,  OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib
# Ensure output directory exists
os.makedirs("plot", exist_ok=True)


def objective(trial):
    """Objective function for Optuna optimization."""
    params = {
        'boosting_type': 'gbdt',
        'device': 'gpu',  # Use GPU for acceleration
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5)
    }

    model = LGBMRegressor(**params)

    scores = cross_val_score(model, X_train, y, cv=5,
                             scoring='neg_mean_squared_error', n_jobs=1)

    return -scores.mean()


def load_data(path):
    # Load your data
    df = pd.read_csv("data/train.csv")
    df = create_features(df)

    X = df.drop(columns=["id", "Calories"])
    y = np.log(df['Calories'])

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
    poly = PolynomialFeatures(2, interaction_only=True)
    X_train = poly.fit_transform(X_train)

    return X_train, y, preprocessor, poly


def main():
    """Main execution function."""
    global X_train, y  # Ensure X_train and y are defined before calling optimization

    X_train, y, preprocessor, poly = load_data(
        "data/train.csv")

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print(f"Best MSLE: {-study.best_value}")

    best_model = LGBMRegressor(
        boosting_type='gbdt', device='gpu', **best_params)
    best_model.fit(X_train, y)

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

    joblib.dump(best_model, 'models/xgb_model.pkl')

    y_pred = best_model.predict(X_train)
    y_pred.to_csv('data/lgb_train_pred.csv', index=False)

    X_submission = pd.read_csv("data/test.csv")
    out = X_submission[["id"]]
    X_submission = create_features(X_submission.drop(columns=["id"]))
    X_submission = preprocessor.transform(X_submission)
    X_submission = poly.transform(X_submission)
    out["Calories"] = np.exp(best_model.predict(X_submission))
    out.to_csv("data/lgb_submission.csv", index=False)


if __name__ == "__main__":
    main()

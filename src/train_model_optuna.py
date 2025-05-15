import pandas as pd
import numpy as np
from utils.feature_engineering import create_features
import matplotlib.pyplot as plt
import seaborn as sns
from utils.hyperparameter_optuna import gpu_accelerated_tuning_optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import StandardScaler,  OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib


def load_data(path):
    # Load your data
    df = pd.read_csv("data/train.csv")
    df = create_features(df)

    X = df.drop(columns=["id", "Calories"])
    y = np.log(df['Calories'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    numerical_features = [col for col in X.columns if col not in ["Sex"]]
    categorical_features = ["Sex"]

    # Create a ColumnTransformer to apply different preprocessing strategies
    preprocessor = ColumnTransformer(

        transformers=[

            ("num", StandardScaler(), numerical_features),

        ],
        remainder="passthrough"

    ).fit(X_train)

    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Load your data
    X_train, X_test, y_train, y_test, preprocessor = load_data(
        "data/train.csv")

    # Run the GPU-accelerated tuning
    best_model, base_models = gpu_accelerated_tuning_optuna(
        X_train, y_train.to_numpy(), X_test, y_test.to_numpy(),         
        cv=5,
        gpu_id=0,
        n_trials=20
        )

    joblib.dump(best_model, 'stacking_ensemble_model.pkl')

    # generate submissions
    X_submission = pd.read_csv("data/test.csv")
    out = X_submission[["id"]]
    X_submission = create_features(X_submission.drop(columns=["id"]))
    X_submission = preprocessor.transform(X_submission)

    out["Calories"] = np.exp(best_model.predict(X_submission))
    out.to_csv("data/submission.csv", index=False)

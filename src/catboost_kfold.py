import optuna
import numpy as np
import os
import pandas as pd
from catboost import CatBoostRegressor
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

# Global variables for data access in objective function
X_train = None
y = None


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
        'random_seed': RANDOM_STATE
    }

    model = CatBoostRegressor(
        **params, 
        cat_features=['Sex'], 
        loss_function='RMSE', 
        eval_metric='RMSE',
        early_stopping_rounds=100, 
        verbose=False
    )

    # Use KFold for optimization as well
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y, cv=kf,
                             scoring='neg_root_mean_squared_error', n_jobs=1)

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
    
    # Map and convert Sex column
    X['Sex'] = X['Sex'].map({1: 'female', 0: 'male'})
    X['Sex'] = X['Sex'].astype('category')

    # Create a preprocessor for numerical features only
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="passthrough"  # Keep categorical features as-is
    )

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)
    
    # Create DataFrame with correct column ordering
    # StandardScaler transforms numerical features first, then remainder (Sex) is added
    transformed_columns = numerical_features + ['Sex']
    X_train = pd.DataFrame(X_transformed, columns=transformed_columns)
    
    # Ensure Sex column is categorical
    X_train['Sex'] = X_train['Sex'].astype('category')

    return X_train, y, df, categorical_features, preprocessor


def train_kfold_ensemble(X_train_data, y_data, best_params):
    """Train ensemble of models using KFold cross-validation."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = []
    oof_predictions = np.zeros(len(y_data))

    print(f"Training {N_SPLITS} models with KFold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_data)):
        print(f"Training fold {fold + 1}/{N_SPLITS}")

        X_fold_train = X_train_data.iloc[train_idx]
        X_fold_val = X_train_data.iloc[val_idx]
        y_fold_train = y_data.iloc[train_idx]
        y_fold_val = y_data.iloc[val_idx]

        # Create model with best parameters
        model = CatBoostRegressor(
            task_type='GPU',
            devices='0',
            verbose=False,
            cat_features=['Sex'],
            loss_function='RMSE', 
            eval_metric='RMSE',
            early_stopping_rounds=100,
            random_seed=RANDOM_STATE + fold,  # Different seed for each fold
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
        np.expm1(y_data), np.expm1(oof_predictions)))
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
        direction='minimize',  # Explicitly specify direction
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print(f"Best RMSE: {study.best_value:.6f}")

    # Train ensemble of models using KFold
    models, oof_predictions = train_kfold_ensemble(X_train, y, best_params)

    # Save all models
    print("Saving trained models...")
    for i, model in enumerate(models):
        joblib.dump(model, f'models/catboost_model_fold_{i}.pkl')

    # Save the preprocessor as well
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Save ensemble information
    model_info = {
        'n_models': len(models),
        'best_params': best_params,
        'oof_score': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_predictions))),
        'cat_features': cat_features
    }
    joblib.dump(model_info, 'models/catboost_ensemble_info.pkl')

    # Generate visualization plots
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances

        history_plot = plot_optimization_history(study)
        importance_plot = plot_param_importances(study)

        print("Optimization history and parameter importance plots saved.")
        history_plot.write_html("plot/catboost_optimization_history.html")
        importance_plot.write_html("plot/catboost_parameter_importance.html")

    except ImportError:
        print(
            "Optuna visualization tools not available. Install with: pip install optuna[visualization]")

    # Generate training predictions using ensemble
    print("Generating training predictions...")
    train_predictions = make_ensemble_predictions(models, X_train)
    out = df[['id']].copy()
    out['calories'] = train_predictions  # Convert back from log scale
    out.to_csv('data/catboost_train_pred.csv', index=False)

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

    # Generate ensemble predictions
    test_predictions = make_ensemble_predictions(models, X_test)

    # Convert back from log scale and clip values
    submission_ids["Calories"] = np.clip(np.expm1(test_predictions), 1, 314)
    submission_ids.to_csv("data/catboost_submission.csv", index=False)

    print(f"Ensemble training complete! Used {len(models)} models.")
    print(f"Final out-of-fold RMSLE: {model_info['oof_score']:.6f}")


def load_ensemble_models():
    """Helper function to load all trained models."""
    info = joblib.load('models/catboost_ensemble_info.pkl')
    models = []
    for i in range(info['n_models']):
        model = joblib.load(f'models/catboost_model_fold_{i}.pkl')
        models.append(model)
    return models, info


if __name__ == "__main__":
    main()
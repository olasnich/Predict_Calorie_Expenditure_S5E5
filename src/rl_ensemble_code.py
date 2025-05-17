import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from utils.feature_engineering import create_features

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # If you have multiple GPUs and want to use only one:
        # tf.config.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(
            f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found. Running on CPU.")


def load_data():
    # Load model predictions
    cat = pd.read_csv('data/catboost_train_pred.csv')
    cat['calories'] = np.exp(cat['calories'])
    xgb = pd.read_csv('data/xgb_train_pred.csv')
    xgb['calories'] = np.exp(xgb['calories'])
    lgb = pd.read_csv('data/lgb_train_pred.csv')
    lgb['calories'] = np.exp(lgb['calories'])

    # Load original features and target
    train_data = pd.read_csv('data/train.csv')
    train_data = create_features(train_data)
    # Merge all predictions
    predictions = pd.merge(cat, pd.merge(xgb, lgb, on='id'), on='id')
    predictions.columns = ['id', 'catboost_pred', 'xgb_pred', 'lgb_pred']

    # Merge with original data to get features
    full_data = pd.merge(train_data, predictions, on='id')

    # Extract relevant columns
    features = full_data.drop(
        ['id', 'Calories', 'catboost_pred', 'xgb_pred', 'lgb_pred'], axis=1)
    true_values = full_data['Calories'].values
    model_preds = full_data[['catboost_pred', 'xgb_pred', 'lgb_pred']].values

    # Exponentiate the predictions (assuming they were log-transformed)
    model_preds = np.exp(model_preds)

    return features, model_preds, true_values

# Define RMSLE function for evaluation


def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-5)
    y_true = np.maximum(y_true, 1e-5)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# RL Agent for model selection


class ModelSelectorAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # Number of models
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Main training function


def train_rl_ensemble():
    print("Setting up training environment...")
    # Load data
    features, model_preds, true_values = load_data()

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data for training and validation
    X_train, X_val, preds_train, preds_val, y_train, y_val = train_test_split(
        features_scaled, model_preds, true_values, test_size=0.2, random_state=42)

    # Initialize agent
    state_size = X_train.shape[1]
    action_size = model_preds.shape[1]  # Number of models
    agent = ModelSelectorAgent(state_size, action_size)

    # Training parameters - you can increase these with GPU acceleration
    episodes = 150  # Increased from 100
    batch_size = 64  # Increased from 32

    best_rmsle = float('inf')
    best_weights = None

    # Create TensorFlow dataset for faster loading
    # This can significantly improve performance on GPU
    train_dataset = tf.data.Dataset.from_tensor_slices((
        X_train, preds_train, y_train
    )).shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"Beginning training for {episodes} episodes...")

    # Training loop
    for e in range(episodes):
        total_reward = 0
        predictions = []
        actions_taken = []

        # Process batches for faster training
        for batch_idx, (states_batch, preds_batch, targets_batch) in enumerate(train_dataset):
            batch_predictions = []
            batch_actions = []
            batch_rewards = 0

            # For each data point in batch
            for i in range(len(states_batch)):
                state = states_batch[i].numpy().reshape(1, -1)

                # Choose model
                action = agent.act(state)
                batch_actions.append(action)

                # Get prediction
                prediction = preds_batch[i][action].numpy()
                batch_predictions.append(prediction)

                # Calculate reward
                error = abs(np.log1p(prediction) -
                            np.log1p(targets_batch[i].numpy()))
                reward = -error

                done = (batch_idx == len(train_dataset) -
                        1 and i == len(states_batch) - 1)

                # Next state
                next_state = states_batch[min(
                    i+1, len(states_batch)-1)].numpy().reshape(1, -1)

                # Remember experience
                agent.remember(state, action, reward, next_state, done)
                batch_rewards += reward

            # Train on batch
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            total_reward += batch_rewards
            predictions.extend(batch_predictions)
            actions_taken.extend(batch_actions)

        # Evaluate on validation set
        val_predictions = []
        for i in range(len(X_val)):
            state = X_val[i].reshape(1, -1)
            action = agent.act(state)
            val_predictions.append(preds_val[i][action])

        # Calculate metrics
        train_rmsle = rmsle(y_train[:len(predictions)], np.array(predictions))
        val_rmsle = rmsle(y_val, np.array(val_predictions))

        # Track best model
        if val_rmsle < best_rmsle:
            best_rmsle = val_rmsle
            agent.save('best_agent.h5')
            print(f"New best model saved! RMSLE: {val_rmsle:.4f}")

        print(f"Episode: {e+1}/{episodes}, Avg Reward: {total_reward/len(X_train):.4f}, "
              f"Train RMSLE: {train_rmsle:.4f}, Val RMSLE: {val_rmsle:.4f}, Epsilon: {agent.epsilon:.4f}")

        # Model usage statistics
        model_usage = np.bincount(
            actions_taken, minlength=action_size) / len(actions_taken)
        print(
            f"Model usage: CatBoost: {model_usage[0]:.2f}, XGBoost: {model_usage[1]:.2f}, LightGBM: {model_usage[2]:.2f}")

    # Load best model
    agent.load('best_agent.h5')

    print("Training completed successfully!")

    return agent, scaler

# Inference function


def predict_with_rl_ensemble(agent, scaler, features, model_preds):
    features_scaled = scaler.transform(features)
    predictions = []

    for i in range(len(features_scaled)):
        state = features_scaled[i].reshape(1, -1)
        action = agent.act(state)
        predictions.append(model_preds[i][action])

    return np.array(predictions)


def generate_predictions_with_full_pipeline():
    """
    Complete pipeline that:
    1. Trains the RL agent
    2. Generates test predictions
    """
    print("Training RL Ensemble...")
    agent, scaler = train_rl_ensemble()

    features, model_preds, true_values = load_data()
    # Generate predictions
    predictions = predict_with_rl_ensemble(
        agent, scaler, test_features, test_preds)

    print("Results:")
    print(f"R² Score: {r2_score(true_values, predictions)}")
    print(f"RMSLE: {rmsle(true_values, predictions):.4f}")

    print("\nGenerating test predictions...")
    # Load test data
    cat_test = pd.read_csv('data/catboost_submission.csv')
    xgb_test = pd.read_csv('data/xgb_submission.csv')
    lgb_test = pd.read_csv('data/lgb_submission.csv')
    test_features = pd.read_csv('data/test.csv')
    test_features = create_features(test_features)

    # Preprocess test data
    test_preds = pd.merge(cat_test, pd.merge(xgb_test[['id', 'Calories']], lgb_test[['id', 'Calories']],
                                             on='id', suffixes=('_xgb', '_lgb')), on='id')
    test_preds.columns = ['id', 'Calories_cat', 'Calories_xgb', 'Calories_lgb']

    full_test_data = pd.merge(test_features, test_preds, on='id')
    features = full_test_data.drop(
        ['id', 'Calories_cat', 'Calories_xgb', 'Calories_lgb'], axis=1)
    model_preds = full_test_data[['Calories_cat',
                                  'Calories_xgb', 'Calories_lgb']].values

    # Use trained agent to make predictions
    predictions = predict_with_rl_ensemble(
        agent, scaler, features, model_preds)

    # Create submission
    submission = pd.DataFrame({
        'id': test_preds['id'],
        'Calories': predictions
    })

    # Save submission
    submission.to_csv('rl_ensemble_submission.csv', index=False)
    print(f"Predictions saved to rl_ensemble_submission.csv")

    return submission


# Run the training
if __name__ == "__main__":

    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", "Yes" if tf.config.list_physical_devices('GPU') else "No")
    if tf.config.list_physical_devices('GPU'):
        print("GPU Devices:", tf.config.list_physical_devices('GPU'))

    # Run the test to validate GPU performance
    with tf.device('/GPU:0'):
        # Simple matrix multiplication to test GPU
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c)

    generate_predictions_with_full_pipeline()

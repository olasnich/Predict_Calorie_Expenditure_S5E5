import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from collections import deque
from utils.feature_engineering import create_features

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Found GPU: {torch.cuda.get_device_name(0)}")
    # Set memory growth equivalent - in PyTorch this is handled differently
    torch.cuda.empty_cache()
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU found. Running on CPU.")


def load_data():
    # Load model predictions
    cat = pd.read_csv('data/new3/catboost_train_pred.csv')
    xgb = pd.read_csv('data/new3/xgb_train_pred.csv')
    lgb = pd.read_csv('data/xgb_train_pred.csv')

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

# Neural network model for deep Q-learning


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(24, 24)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(24, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

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
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a minibatch
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(
                        next_state).to(device)
                    target = reward + self.gamma * \
                        torch.max(self.model(next_state_tensor)).item()

            # Get current Q values
            state_tensor = torch.FloatTensor(state).to(device)
            self.optimizer.zero_grad()

            # Forward pass
            current_q_values = self.model(state_tensor)

            # Create target Q value for the action taken
            target_q_values = current_q_values.clone()
            target_q_values[0, action] = target

            # Compute loss and backpropagate
            loss = self.criterion(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)

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

    # Training parameters
    episodes = 150
    batch_size = 64

    best_rmsle = float('inf')

    # Create PyTorch DataLoader
    train_tensor_x = torch.FloatTensor(X_train)
    train_tensor_preds = torch.FloatTensor(preds_train)
    train_tensor_y = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(
        train_tensor_x, train_tensor_preds, train_tensor_y)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Beginning training for {episodes} episodes...")

    # Training loop
    for e in range(episodes):
        total_reward = 0
        predictions = []
        actions_taken = []

        # Process batches
        for states_batch, preds_batch, targets_batch in train_dataloader:
            batch_predictions = []
            batch_actions = []
            batch_rewards = 0

            # Process each sample in the batch
            for i in range(len(states_batch)):
                state = states_batch[i].cpu().numpy().reshape(1, -1)

                # Choose model
                action = agent.act(state)
                batch_actions.append(action)

                # Get prediction
                prediction = preds_batch[i][action].item()
                batch_predictions.append(prediction)

                # Calculate reward
                error = abs(np.log1p(prediction) -
                            np.log1p(targets_batch[i].item()))
                reward = -error

                done = False  # Only set to True at the end of an episode

                # Next state (wrap around at the end of batch)
                next_idx = (i + 1) % len(states_batch)
                next_state = states_batch[next_idx].cpu(
                ).numpy().reshape(1, -1)

                # Remember experience
                agent.remember(state, action, reward, next_state, done)
                batch_rewards += reward

            # Mark the last example in the last batch as done
            if len(agent.memory) > 0:
                state, action, reward, next_state, _ = agent.memory[-1]
                agent.memory[-1] = (state, action, reward, next_state, True)

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
            agent.save('best_agent.pth')
            print(f"New best model saved! RMSLE: {val_rmsle:.4f}")

        print(f"Episode: {e+1}/{episodes}, Avg Reward: {total_reward/len(X_train):.4f}, "
              f"Train RMSLE: {train_rmsle:.4f}, Val RMSLE: {val_rmsle:.4f}, Epsilon: {agent.epsilon:.4f}")

        # Model usage statistics
        model_usage = np.bincount(
            actions_taken, minlength=action_size) / len(actions_taken)
        print(
            f"Model usage: CatBoost: {model_usage[0]:.2f}, XGBoost: {model_usage[1]:.2f}, LightGBM: {model_usage[2]:.2f}")

    # Load best model
    agent.load('best_agent.pth')

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

    # Load training data for validation
    features, model_preds, true_values = load_data()

    # Generate predictions on training data for metrics
    predictions = predict_with_rl_ensemble(
        agent, scaler, features, model_preds)

    print("Results:")
    print(f"R² Score: {r2_score(true_values, predictions)}")
    print(f"RMSLE: {rmsle(true_values, predictions):.4f}")

    print("\nGenerating test predictions...")
    # Load test data
    cat_test = pd.read_csv('data/new3/catboost_submission.csv')
    xgb_test = pd.read_csv('data/new3/xgb_submission.csv')
    lgb_test = pd.read_csv('data/xgb_submission.csv')
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
    print("PyTorch version:", torch.__version__)
    print("GPU Available:", "Yes" if torch.cuda.is_available() else "No")
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("GPU Device:", torch.cuda.get_device_name(0))

    # Run a simple test to validate GPU performance
    if torch.cuda.is_available():
        # Simple matrix multiplication to test GPU
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        c = torch.matmul(a, b)
        print("Matrix multiplication result:", c)

    generate_predictions_with_full_pipeline()

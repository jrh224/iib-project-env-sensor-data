import sys
import os

from sklearn.preprocessing import MinMaxScaler

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import LSTMModel, LSTM_CNN_Model
from utils.fake_data_gen import *
from utils.helper_functions import autoregressive_predict, create_sequences
import config

full_matrix = gen_r2c2_w_irregular_heating_real_meteo(seed=42) # 3 covariates
test_matrix = gen_r2c2_w_irregular_heating_real_meteo(seed=51) # 3 covariates
WITH_CNN = True # decide whether to include the CNN layer or not


train_split = config.TRAIN_SPLIT
train_split_i = int(np.floor(full_matrix.shape[0] * 0.8))

train_matrix = full_matrix[:train_split_i, :]
val_matrix = full_matrix[train_split_i:, :]

test_matrix_unscaled = test_matrix.copy()


# Scale each column using a different scaler
scalers = []
for i in range(train_matrix.shape[1]): # for each feature column
    scaler = MinMaxScaler()
    train_matrix[:, i] = scaler.fit_transform(train_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time
    val_matrix[:, i] = scaler.transform(val_matrix[:, i].reshape(-1, 1)).flatten() 
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten()
    scalers.append(scaler)



# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512])
    # num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    print(f"Now testing | LR {lr} | LR Decay {weight_decay} | Hidden Size {hidden_size} | Batch Size {batch_size}")
    
    # Load data and create sequences
    X_train, y_train = create_sequences(train_matrix, config.LOOKBACK, 1, config.STRIDE, 0)
    X_val, y_val = create_sequences(val_matrix, config.LOOKBACK, 1, config.STRIDE, 0)

    
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=True)

    # Initialise model
    if WITH_CNN:
        model = LSTM_CNN_Model(hidden_size=hidden_size)
    else:
        model = LSTMModel(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        for inputs, target in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, target in val_loader:
                output = model(inputs)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model before testing
    model.load_state_dict(best_model_state)

    
    # Evaluate model using MAE on TEST dataset
    y_predictions = []
    y_actuals = []

    # assuming i_start is the beginning of the lookback period
    for i_start in range(0, test_matrix.shape[0] - config.LOOKBACK - config.HORIZON, config.STRIDE): # stride = 1
        y_prediction = autoregressive_predict(model, test_matrix, config.HORIZON, i_start)
        y_prediction = np.array(y_prediction).reshape(-1, 1)
        y_predictions.extend(np.array(y_prediction).reshape(-1).flatten())

        y_actual = test_matrix[i_start+config.LOOKBACK:i_start+config.LOOKBACK+config.HORIZON, 0]
        y_actuals.extend(np.array(y_actual).reshape(-1).flatten())

    y_predictions = np.array(y_predictions)
    y_actuals = np.array(y_actuals)

    mae = np.mean(np.abs(y_actuals - y_predictions))
    
    return mae

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best hyperparameters:", study.best_params)
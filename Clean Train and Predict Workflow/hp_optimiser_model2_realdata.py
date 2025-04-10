import sys
import os

from sklearn.preprocessing import MinMaxScaler

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import Seq2SeqLSTMEncDec, Seq2SeqLSTMEncDec_CNN
from utils.fake_data_gen import *
from utils.helper_functions import autoregressive_predict, create_sequences, get_encdec_inputs
from utils.CustomDataframe import CustomDataframe
import config
import joblib

print(torch.cuda.is_available())  # Should return True if GPU is enabled
# print(torch.cuda.get_device_name(0))  # Displays the GPU model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WITH_CNN = True # decide whether to include the CNN layer or not
best_model_path = "26mar_1555_m3.pth"

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')

# Add external temperature to sensor_data object
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
# Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

sensor_data_train, idx_blocks_train = sensor_data.filter_by_date_ranges(dates=config.TRAIN_RANGE, in_place=False)
train_matrix = sensor_data_train.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Train Matrix Created Successfully [Shape: {train_matrix.shape}]")

sensor_data_val, idx_blocks_val = sensor_data.filter_by_date_ranges(dates=config.VALID_RANGE, in_place=False)
val_matrix = sensor_data_val.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Validation Matrix Created Successfully [Shape: {val_matrix.shape}]")

# Scale each column using a different scaler
scalers = []
for i in range(train_matrix.shape[1]): # for each feature column
    scaler = MinMaxScaler()
    train_matrix[:, i] = scaler.fit_transform(train_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time
    val_matrix[:, i] = scaler.transform(val_matrix[:, i].reshape(-1, 1)).flatten() 
    scalers.append(scaler)
joblib.dump(scalers, 'real_data_scalers.gz') # Store all the scalers in order to be used when predicting

# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 2e-4, 2e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 768])
    # num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    print(f"Now testing | LR {lr} | LR Decay {weight_decay} | Hidden Size {hidden_size} | Batch Size {batch_size}")
    
    # Load data and create sequences
    train_enc_inp, train_dec_inp, train_targets = get_encdec_inputs(train_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_train)
    val_enc_inp, val_dec_inp, val_targets = get_encdec_inputs(val_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_val)

    # Create dataset for training
    train_dataset = TensorDataset(train_enc_inp, train_dec_inp, train_targets)
    val_dataset = TensorDataset(val_enc_inp, val_dec_inp, val_targets)

    # Create DataLoader for efficient batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialise model
    if WITH_CNN:
        model = Seq2SeqLSTMEncDec_CNN(hidden_dim=hidden_size).to(device)
    else:
        model = Seq2SeqLSTMEncDec(hidden_dim=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for enc_inps, dec_inps, targets in train_loader:
            optimizer.zero_grad()
            output = model(enc_inps.to(device), dec_inps.to(device))
            loss = criterion(output, targets.to(device))
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_inps, dec_inps, targets in val_loader:
                output = model(enc_inps.to(device), dec_inps.to(device))
                loss = criterion(output, targets.to(device))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)  # Save best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best hyperparameters:", study.best_params)
print("Best model stored:", best_model_path)
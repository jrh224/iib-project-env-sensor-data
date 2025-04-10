import sys
import os

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

import config
from models import LSTMModel, Seq2SeqLSTMEncDec_CNN, Seq2SeqLSTM, Seq2SeqLSTMEncDec
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *
from utils.fake_data_gen import *

best_model_path = "08apr_0938_B0A732379968.pth"  # Path to save best model
scaler_file = "scalers_B0A732379968.gz"

# Can vary the data generated here
# full_matrix = gen_sum_of_consts(hours=2160, length=25920, no_covariates=6, seed=42)
# full_matrix = gen_sum_of_consts_w_lag(hours=2160, length=25920, no_covariates=6, seed=42)
# full_matrix = gen_sum_of_sine_waves_rand_phase(hours=2160, length=25920, no_covariates=6, seed=42)
# full_matrix = gen_r2c2_w_irregular_heating_real_meteo(hours=2160, length=25920, seed=42) # 3 covariates

WITH_CNN = True # decide whether to include the CNN layer or not

# train_split = config.TRAIN_SPLIT
# train_split_i = int(np.floor(full_matrix.shape[0] * 0.8))

# train_matrix = full_matrix[:train_split_i, :]
# val_matrix = full_matrix[train_split_i:, :]


# # Scale each column using a different scaler
# scalers = []
# for i in range(train_matrix.shape[1]): # for each feature column
#     scaler = MinMaxScaler()
#     train_matrix[:, i] = scaler.fit_transform(train_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time
#     val_matrix[:, i] = scaler.transform(val_matrix[:, i].reshape(-1, 1)).flatten()  # Transform validation data
#     scalers.append(scaler)
# joblib.dump(scalers, 'scalers.gz') # Store all the scalers in order to be used when predicting

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')

# Add external temperature to sensor_data object
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
# Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

# print(sensor_data.df)

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
joblib.dump(scalers, scaler_file) # Store all the scalers in order to be used when predicting


train_enc_inp, train_dec_inp, train_targets = get_encdec_inputs(train_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_train)
val_enc_inp, val_dec_inp, val_targets = get_encdec_inputs(val_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_val)

print(f"EncDec inputs successfully generated. EncInp: {train_enc_inp.shape}, DecInp: {train_dec_inp.shape}, Targets: {train_targets.shape}")

# Create dataset for training
train_dataset = TensorDataset(train_enc_inp, train_dec_inp, train_targets)
val_dataset = TensorDataset(val_enc_inp, val_dec_inp, val_targets)

# Create DataLoader for efficient batching
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


# # Initialise model
if WITH_CNN:
    model = Seq2SeqLSTMEncDec_CNN()
else:
    model = Seq2SeqLSTMEncDec()
print("Training model")

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.MSELoss()

# Cosine Annealing Learning Rate Scheduler
# T_max is the maximum number of iterations (or epochs), eta_min is the minimum learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=0)

min_val_loss = float('inf')  # Track best validation loss
epochs_no_improve = 0  # Counter for early stopping


model.train()
for epoch in range(config.NUM_EPOCHS):
    # Training
    epoch_train_loss = 0
    for enc_inps, dec_inps, targets in train_loader:
        optimizer.zero_grad()
        output = model(enc_inps, dec_inps)  # Forward pass
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    avg_train_loss = epoch_train_loss / len(train_loader)


    # Validation
    model.eval()  # Set model to evaluation mode
    epoch_val_loss = 0
    with torch.no_grad():
        for enc_inps, dec_inps, targets in val_loader:
            output = model(enc_inps, dec_inps)
            loss = criterion(output, targets)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}] | '
        f'Train Loss: {avg_train_loss:.5f} | '
        f'Val Loss: {avg_val_loss:.5f}'
        f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
    
    # Check for validation loss improvement
    if avg_val_loss < min_val_loss:
        print(f"Validation loss improved ({min_val_loss:.5f} → {avg_val_loss:.5f}). Saving model...")
        min_val_loss = avg_val_loss
        epochs_no_improve = 0  # Reset counter
        torch.save(model.state_dict(), best_model_path)  # Save best model
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE} epochs.")

    # Early stopping condition
    if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered! Best model saved at '{best_model_path}' with val loss {min_val_loss:.5f}")
        break

    model.train()  # Switch back to training mode


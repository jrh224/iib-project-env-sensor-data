import sys
import os

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

import config
from models import LSTMModel
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data_train = sensor_data.filter_by_date(start_date=config.start_date_train, end_date=config.end_date_train, in_place=False)

train_matrix = sensor_data_train.create_pytorch_matrix(lat=config.LAT, long=config.LONG)

# Split into train and validation dataset
split_idx = int(len(train_matrix) * config.TRAIN_SPLIT)
train_matrix, val_matrix = train_matrix[:split_idx], train_matrix[split_idx:]

# Scale each column using a different scaler
scalers = []
for i in range(train_matrix.shape[1]): # for each feature column
    scaler = MinMaxScaler()
    train_matrix[:, i] = scaler.fit_transform(train_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time
    val_matrix[:, i] = scaler.transform(val_matrix[:, i].reshape(-1, 1)).flatten()  # Transform validation data
    scalers.append(scaler)
joblib.dump(scalers, 'scalers.gz') # Store all the scalers in order to be used when predicting


X_train, y_train = create_sequences(train_matrix, config.LOOKBACK, target_col=0)
X_val, y_val = create_sequences(val_matrix, config.LOOKBACK, target_col=0)

# Create DataLoader directly from tensors
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=config.BATCH_SIZE, shuffle=True) # Automatically deals with batching etc -> more efficient training
val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=config.BATCH_SIZE, shuffle=True)


# Initialise model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

min_val_loss = float('inf')  # Track best validation loss
epochs_no_improve = 0  # Counter for early stopping
best_model_path = '14feb1518.pth'  # Path to save best model

model.train()
for epoch in range(config.NUM_EPOCHS):
    # Training
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)

    # Validation
    model.eval()  # Set model to evaluation mode
    epoch_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}] | '
          f'Train Loss: {avg_train_loss:.5f} | '
          f'Val Loss: {avg_val_loss:.5f}')
    
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

torch.save(model.state_dict(), best_model_path)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.CustomDataframe import CustomDataframe
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.openmeteocustom import get_external_temp, interpolate_ext_temp

# Configuration
SEQUENCE_LENGTH = 20
INPUT_SIZE = 2 # int and ext temp
BATCH_SIZE = 32
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

start_date = pd.to_datetime("10/09/2024 14:00")
end_date = pd.to_datetime("10/16/2024 14:00")

sensor_data_train = sensor_data.filter_by_date(start_date=start_date, end_date=end_date, in_place=False)
# sensor_data_test = sensor_data.filter_by_date(start_date="10/16/2024 14:00", end_date="10/23/2024 14:00", in_place=False)

# Add the external temperature column to the dataframe
sensor_data_train.add_ext_temp_column()

# Load internal and external temperatures
internal_temp = sensor_data_train.df["T"].values.astype(np.float32).reshape(-1, 1)
external_temp = sensor_data_train.df["temperature_2m"].values.astype(np.float32).reshape(-1, 1)

# Scale data using separate scalers
scaler_internal = MinMaxScaler()
scaler_external = MinMaxScaler()

train_internal_scaled = scaler_internal.fit_transform(internal_temp)
train_external_scaled = scaler_external.fit_transform(external_temp)

# Combine scaled features into a 2D array
train_scaled = np.hstack((train_internal_scaled, train_external_scaled))

print(train_scaled)

# Create sequence datasets
def create_sequences(data, seq_length, target_col=0):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, target_col]) # only add the internal temperature to the y vector
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)


X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH, target_col=0)

# Create DataLoader directly from tensors
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True) # Automatically deals with batching etc -> more efficient training

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size) # Apparently don't need a ReLU layer because this is a regression task
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.linear(last_time_step)
    
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    # # Validation
    # model.eval()
    # epoch_val_loss = 0
    # with torch.no_grad():
    #     for inputs, targets in val_loader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         epoch_val_loss += loss.item()
    
    # Calculate average losses
    avg_train_loss = epoch_train_loss / len(train_loader)
    # avg_val_loss = epoch_val_loss / len(val_loader)
    
    train_losses.append(avg_train_loss)
    # val_losses.append(avg_val_loss)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
          f'Train Loss: {avg_train_loss:.5f} | ')
    
    # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
    #       f'Train Loss: {avg_train_loss:.5f} | '
    #       f'Val Loss: {avg_val_loss:.5f}')

# Plot training history
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'model_5thfeb1.pth')


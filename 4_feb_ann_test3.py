import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.CustomDataframe import CustomDataframe

# Configuration
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size) # Apparently don't need a ReLU layer because this is a regression task
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.linear(last_time_step)

# Get the data

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data_train = sensor_data.filter_by_date(start_date="10/09/2024 14:00", end_date="10/16/2024 14:00", in_place=False)
sensor_data_test = sensor_data.filter_by_date(start_date="10/16/2024 14:00", end_date="10/23/2024 14:00", in_place=False)

# Load and prepare test data
test_temps = sensor_data_test.df["T"].values.astype(np.float32)

# Get training data for scaler fitting
train_temps = sensor_data_train.df["T"].values.astype(np.float32)

# Initialize and fit scaler on training data - IMPORTANT so that it matches the scaling applied before training
scaler = MinMaxScaler()
scaler.fit(train_temps.reshape(-1, 1))

# Scale test data using training scaler
test_scaled = scaler.transform(test_temps.reshape(-1, 1))

# Create sequence datasets
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)

# Create test sequences
X_test, y_test = create_sequences(test_scaled, SEQUENCE_LENGTH)


# Make predictions
model = LSTMModel()
model.load_state_dict(torch.load('model_4thfeb1.pth', weights_only=True))
model.eval()

with torch.no_grad():
    test_predictions = model(X_test)

# Inverse transform predictions and actual values
predictions_actual = scaler.inverse_transform(test_predictions.numpy())
targets_actual = scaler.inverse_transform(y_test.reshape(-1, 1)) # since this derives from the test_scaled variable, and so needs to be inverse scaled

print(predictions_actual)
print(targets_actual)

# Plot results
plt.figure(figsize=(15, 6))
plt.plot(range(len(targets_actual)), targets_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o", alpha=0.7)
plt.plot(range(len(predictions_actual)), predictions_actual, label='Predicted Temperature', color="red", linestyle="--", marker="x", alpha=0.7)
plt.title('Temperature Prediction vs Actual Values')
plt.ylabel('Temperature')
plt.xlabel('Timestep')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

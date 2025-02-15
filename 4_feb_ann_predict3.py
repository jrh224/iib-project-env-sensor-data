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
    

model = LSTMModel()
model.load_state_dict(torch.load('model_4thfeb1.pth', weights_only=True))
model.eval()

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

def autoregressive_predict(model, scaler, initial_sequence, num_predictions):
    """Make autoregressive predictions using the trained model"""
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(num_predictions):
        # Convert to tensor and add batch dimension (so that the input fits the size requirement)
        input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predicted = model(input_tensor)
            
        # Store prediction
        predictions.append(predicted.item())
        
         # Update sequence: remove oldest, add new prediction
        current_sequence = np.vstack([
            current_sequence[1:],  # Remove oldest entry
            predicted.item()       # Add new prediction
        ])
        
    return predictions

# Get initial sequence (first SEQUENCE_LENGTH points from test data)
initial_sequence = test_scaled[:SEQUENCE_LENGTH]

# Calculate number of predictions to make (remainder of test data)
num_predictions = SEQUENCE_LENGTH * 5

# Generate autoregressive predictions
predictions_scaled = autoregressive_predict(model, scaler, initial_sequence, num_predictions)

# Inverse transform predictions and actual values
predictions_actual = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
actual_actual = scaler.inverse_transform(test_scaled[0:len(predictions_actual)+SEQUENCE_LENGTH])

# Create x-axis values for the actual temperature
x_actual = range(0, len(actual_actual))

# Create x-axis values for the predictions
x_predictions = range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + len(predictions_actual))

# Plot results
plt.figure(figsize=(15, 6))
plt.plot(x_actual, actual_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
plt.plot(x_predictions, predictions_actual, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
plt.title('Autoregressive Temperature Predictions vs Actual Values')
plt.ylabel('Temperature')
plt.xlabel('Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
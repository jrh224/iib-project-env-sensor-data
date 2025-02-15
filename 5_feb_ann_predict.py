import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.CustomDataframe import CustomDataframe
from utils.openmeteocustom import *

# Configuration
SEQUENCE_LENGTH = 20
INPUT_SIZE = 2 # int and ext temp
BATCH_SIZE = 32
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8

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
model.load_state_dict(torch.load('5feb1505.pth', weights_only=True))
model.eval()

# Get the data

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

start_date_train = pd.to_datetime("10/09/2024 14:00")
end_date_train = pd.to_datetime("10/16/2024 14:00")
start_date_test = pd.to_datetime("10/16/2024 14:00")
end_date_test = pd.to_datetime("10/23/2024 14:00")

sensor_data_train = sensor_data.filter_by_date(start_date=start_date_train, end_date=end_date_train, in_place=False)
sensor_data_test = sensor_data.filter_by_date(start_date=start_date_test, end_date=end_date_test, in_place=False)

# Add the external temperature column to the train dataframe
hourly_extt_dataframe_train = get_external_temp(start_date=start_date_train.strftime('%Y-%m-%d'), end_date=end_date_train.strftime('%Y-%m-%d'))
interpolate_ext_temp(hourly_extt_dataframe_train, start_date_train, end_date_train, sensor_data_train)

# Add the external temperature column to the test dataframe
hourly_extt_dataframe_test = get_external_temp(start_date=start_date_test.strftime('%Y-%m-%d'), end_date=end_date_test.strftime('%Y-%m-%d'))
interpolate_ext_temp(hourly_extt_dataframe_test, start_date_test, end_date_test, sensor_data_test)

# Load internal and external temperatures
internal_temp_train = sensor_data_train.df["T"].values.astype(np.float32).reshape(-1, 1)
external_temp_train = sensor_data_train.df["temperature_2m"].values.astype(np.float32).reshape(-1, 1)

internal_temp_test = sensor_data_test.df["T"].values.astype(np.float32).reshape(-1, 1)
external_temp_test = sensor_data_test.df["temperature_2m"].values.astype(np.float32).reshape(-1, 1)


external_temp_test = np.full(external_temp_test.shape, float(-10)) # Force ext temp to be 10 for testing purposes



# Scale data using separate scalers
scaler_internal = MinMaxScaler()
scaler_external = MinMaxScaler()

train_internal_scaled = scaler_internal.fit_transform(internal_temp_train)
train_external_scaled = scaler_external.fit_transform(external_temp_train)

test_internal_scaled = scaler_internal.transform(internal_temp_test)
test_external_scaled = scaler_external.transform(external_temp_test)

# Combine scaled features into a 2D array
test_scaled = np.hstack((test_internal_scaled, test_external_scaled))

def autoregressive_predict(model, initial_sequence, num_predictions):
    """Make autoregressive predictions using the trained model"""
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for i in range(num_predictions):
        # Convert to tensor and add batch dimension (so that the input fits the size requirement)
        input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predicted = model(input_tensor)
            
        # Store prediction
        predictions.append(predicted.item())
        
        # Update sequence: remove oldest and append new predicted internal temperature + actual external temperature
        next_external_temp = float(external_temp_test[SEQUENCE_LENGTH + i])  # Use actual future external temp
        # new_entry = np.array([[predicted.item(), next_external_temp]])  # Shape (1, 2)
        new_entry = np.array([[predicted.item(), next_external_temp]])  # Shape (1, 2)
        
         # Update sequence: remove oldest, add new prediction
        current_sequence = np.vstack([
            current_sequence[1:],  # Remove oldest entry
            new_entry       # Add new prediction
        ])
        
    return predictions

# Get initial sequence (first SEQUENCE_LENGTH points from test data)
initial_sequence = test_scaled[:SEQUENCE_LENGTH]

# Calculate number of predictions to make (remainder of test data)
num_predictions = SEQUENCE_LENGTH * 5

# Generate autoregressive predictions
predictions_scaled = autoregressive_predict(model, initial_sequence, num_predictions)

# Inverse transform predictions and actual values
predictions_actual = scaler_internal.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
actual_actual = scaler_internal.inverse_transform(
    test_scaled[0:len(predictions_actual)+SEQUENCE_LENGTH, 0].reshape(-1, 1)
)

# Create x-axis values for the actual temperature
x_actual = range(0, len(actual_actual))
# Create x-axis values for the predictions
x_predictions = range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + len(predictions_actual))
# Create x-axis values for the external temperature
x_external = range(SEQUENCE_LENGTH + len(predictions_actual))

# Plot results
plt.figure(figsize=(15, 6))
plt.plot(x_actual, actual_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
plt.plot(x_predictions, predictions_actual, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
plt.plot(x_external, external_temp_test[0:SEQUENCE_LENGTH+len(predictions_actual)], label='External Temperature', color='green')
plt.plot()
plt.title('Autoregressive Temperature Predictions vs Actual Values')
plt.ylabel('Temperature')
plt.xlabel('Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
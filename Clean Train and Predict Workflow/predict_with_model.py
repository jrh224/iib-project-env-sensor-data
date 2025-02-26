import sys
import os

from matplotlib import pyplot as plt

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

import config
from models import LSTMModel, Seq2SeqLSTM
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME) # NB: change back to FILENAME if all in same file

sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')

# Add external temperature to sensor_data object
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
# Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

sensor_data_test, idx_blocks_test = sensor_data.filter_by_date_ranges(dates=config.TEST_RANGE, in_place=False)

test_matrix = sensor_data_test.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Test Matrix Created Successfully [Shape: {test_matrix.shape}]")

# Get the starting point for the predictions
predict_from_i = sensor_data_test.df.index.get_loc(sensor_data_test.df.index[sensor_data_test.df.index >= config.PREDICT_FROM][0])
print("Predict_from_i: " + str(predict_from_i))
print("First Prediction Timestamp:", sensor_data_test.df.index[predict_from_i])
print("Expected Start Timestamp:", config.PREDICT_FROM)
# # TEST: Force control to be 100
# forced_control = np.full(test_matrix[predict_from_i:, 2].shape, float(100))
# test_matrix[predict_from_i:, 2] = forced_control

scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time

# Initialise the model for prediction
model = Seq2SeqLSTM()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()
X_test, y_test = create_sequences(test_matrix, lookback=config.LOOKBACK, predictforward=config.OUTPUT_SIZE, step=config.SEQ_STEP, target_col=0, blocks=idx_blocks_test)
current_lookback = X_test[predict_from_i] # TRY CHANGING THIS!!
print(np.array(current_lookback).shape) # (156, 7)

input_tensor = torch.tensor(current_lookback, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    predictions = model(input_tensor)

# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
print(y_prediction)
y_real_temp = scalers[0].inverse_transform(np.array(y_test[0]).reshape(-1, 1))
print(y_real_temp)

# Create timestamps
x_prediction = sensor_data_test.df.iloc[predict_from_i+config.LOOKBACK:predict_from_i+config.LOOKBACK+predictions.shape[1]].index.to_numpy()
x_actual = sensor_data_test.df.iloc[predict_from_i:predict_from_i + config.LOOKBACK + predictions.shape[1]].index.to_numpy()
# Get actual IAT readings
y_actual = sensor_data_test.df.iloc[predict_from_i:predict_from_i + config.LOOKBACK + predictions.shape[1]]["T"].to_numpy()



plt.plot(x_actual, y_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
plt.plot(x_prediction, y_prediction, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
# plt.plot(x_actual, exttemp, label="External temperature", color="green", linestyle="-", marker="o")
plt.title('Autoregressive Temperature Predictions vs Actual Values')
plt.ylabel('Temperature')
plt.xlabel('Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
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
sensor_data = CustomDataframe(filename=config.TEST_FILENAME) # NB: change back to FILENAME if all in same file
sensor_data_test = sensor_data.filter_by_date(start_date=config.start_date_test, end_date=config.end_date_test, in_place=False)

sensor_data_test.interpolate_missing_rows()
sensor_data_test.resample(freq='15s') # Also reindexes df

test_matrix = sensor_data_test.create_pytorch_matrix(lat=config.LAT, long=config.LONG)

# Get the starting point for the predictions
predict_from_i = sensor_data_test.df.index.get_loc(sensor_data_test.df.index[sensor_data_test.df.index > config.PREDICT_FROM][0])
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
current_lookback = test_matrix[predict_from_i:config.LOOKBACK+predict_from_i]
input_tensor = torch.tensor(current_lookback, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    predictions = model(input_tensor)

# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))

# Create timestamps
x_prediction = sensor_data_test.df.iloc[predict_from_i+config.LOOKBACK:predict_from_i+config.LOOKBACK+predictions.shape[1]].index.to_numpy().reshape(-1, 1)
x_actual = sensor_data_test.df.iloc[predict_from_i:config.LOOKBACK + predictions.shape[1] + predict_from_i].index.to_numpy()
# Get actual IAT readings
y_actual = sensor_data_test.df.iloc[predict_from_i:config.LOOKBACK + predictions.shape[1] + predict_from_i]["T"].to_numpy()

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
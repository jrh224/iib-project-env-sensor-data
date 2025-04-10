import sys
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # temporary to avoid DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments.

from matplotlib import pyplot as plt
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset
import config
from models import LSTMModel, Seq2SeqLSTM, Seq2SeqLSTMEncDec
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

scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time

test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_test)
test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)

# Initialise the model for prediction
model = Seq2SeqLSTMEncDec()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Pass through train dataset, predicting one hour every half hour

# Get the starting point for the predictions
# predict_from_i = sensor_data_test.df.index[sensor_data_test.df['datetime'] > config.PREDICT_FROM].to_list()[0]

y_predictions = []
y_actuals = []

model.eval()

for enc_inp, dec_inp, target in test_dataset:
    # Make prediction
    with torch.no_grad():
        predictions = model(enc_inp, dec_inp)

    # Inverse transform predictions to get correct scale
    y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
    # Get actual IAT readings
    y_actual = scalers[0].inverse_transform(np.array(target).reshape(-1, 1))

    # Store y_prediction and y_actual for evaluation later on
    y_predictions.extend(np.array(y_prediction).reshape(-1).flatten())
    y_actuals.extend(np.array(y_actual).reshape(-1).flatten())

    # plt.clf()
    # # Ensure x_actual and x_prediction have correct shape
    # x_actual = sensor_data_test.df.iloc[predict_from_i+config.LOOKBACK:predict_from_i + config.LOOKBACK + len(y_prediction)].index.to_numpy()
    # x_prediction = sensor_data_test.df.iloc[predict_from_i + config.LOOKBACK:predict_from_i + config.LOOKBACK + len(y_prediction)].index.to_numpy()

    # plt.plot(x_actual, y_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
    # plt.plot(x_prediction, y_prediction, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
    # plt.xlabel("Time")
    # plt.ylabel("Temperature C")
    # plt.legend()
    # plt.show()

y_actuals = np.array(y_actuals)
y_predictions = np.array(y_predictions)

# Calculate MAE and R2 from y_prediction and y_actual
rmse = np.sqrt(np.mean((y_actuals - y_predictions) ** 2))

mae = np.mean(np.abs(y_actuals - y_predictions))
y_mean = np.mean(y_actuals)
r2 = 1 - np.sum((y_actuals - y_predictions)**2) / np.sum((y_actuals - y_mean)**2)

print("MAE (mean absolute error): " + str(mae))
print("R2 (coefficient of determination): " + str(r2))
print("RMSE: " + str(rmse))
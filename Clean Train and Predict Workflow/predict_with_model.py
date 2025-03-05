import sys
import os

from matplotlib import pyplot as plt

# Add the parent directory of 'my_package' to sys.path
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

# # Import sensor data into CustomDataframe object
# sensor_data = CustomDataframe(filename=config.FILENAME) # NB: change back to FILENAME if all in same file

# sensor_data.interpolate_missing_rows()
# sensor_data.resample(freq='5Min')

# # Add external temperature to sensor_data object
# sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
# # Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
# sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

# sensor_data_test, idx_blocks_test = sensor_data.filter_by_date_ranges(dates=config.TEST_RANGE, in_place=False)

# test_matrix = sensor_data_test.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
# print(f"Test Matrix Created Successfully [Shape: {test_matrix.shape}]")

# # Get the starting point for the predictions
# predict_from_i = sensor_data_test.df.index.get_loc(sensor_data_test.df.index[sensor_data_test.df.index >= config.PREDICT_FROM][0])
# print("Predict_from_i: " + str(predict_from_i))
# print("First Prediction Timestamp:", sensor_data_test.df.index[predict_from_i])


t = np.linspace(2061, 4220, 25920) # Roughly based no 2160 hours in 3 months, with 12x as many datapoints
x1 = 3 * np.sin((2 * np.pi * (t - 12)) / 24) + 7
x2 = np.cos((2 * np.pi * (t+4)) / 4)
x3 = 0.5 * np.sin((2 * np.pi * (t-7)) / 9)
x4 = 3 * np.sin((2 * np.pi * (t+1)) / 5) + 2
x5 = 2 * np.cos((2 * np.pi * (t-3)) / 2)
x6 = 2 * np.cos((2 * np.pi * (t-3)) / 3)
y = x1 + x2 + x3 + x4 + x5 + x6

x1 = x1.reshape(-1, 1)
x2 = x2.reshape(-1, 1)
x3 = x3.reshape(-1, 1)
x4 = x4.reshape(-1, 1)
x5 = x5.reshape(-1, 1)
x6 = x6.reshape(-1, 1)
y = y.reshape(-1, 1)

# train_split = config.TRAIN_SPLIT # 0.8
# train_split_i = int(np.floor(len(t) * 0.8))
# print(train_split_i)

test_matrix = np.hstack((y, x1, x2, x3, x4, x5, x6))
# train_matrix = full_matrix[:train_split_i, :]
# val_matrix = full_matrix[train_split_i:, :]


scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time


test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=None)
test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)


print(f"EncDec inputs successfully generated. EncInp: {test_enc_inp.shape}, DecInp: {test_dec_inp.shape}, Targets: {test_targets.shape}")

# Initialise the model for prediction
model = Seq2SeqLSTMEncDec()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()

# enc_inp, dec_inp, target = test_dataset[predict_from_i]
enc_inp, dec_inp, target = test_dataset[0]

# # Force heating to 100 for simulation
# dec_inp = np.array(dec_inp)
# print(dec_inp.shape)
# print(dec_inp[:, 1])
# dec_inp[:, 1] = scalers[2].transform(np.full_like(dec_inp.shape[0], 100).reshape(-1,1))
# print(dec_inp[:, 1])
# dec_inp = torch.tensor(dec_inp, dtype=torch.float32)

# print(enc_inp)
# print(dec_inp)


with torch.no_grad():
    predictions = model(enc_inp, dec_inp)

# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
# print(y_prediction)
y_real_temp = scalers[0].inverse_transform(np.array(target).reshape(-1, 1))
# print(y_real_temp)

# # Create timestamps
# x_prediction = sensor_data_test.df.iloc[predict_from_i + config.LOOKBACK : predict_from_i + config.LOOKBACK + predictions.shape[0]].index.to_numpy()
# x_actual = sensor_data_test.df.iloc[predict_from_i:predict_from_i + config.LOOKBACK + predictions.shape[0]].index.to_numpy()
# # Get actual IAT readings
# y_actual = sensor_data_test.df.iloc[predict_from_i:predict_from_i + config.LOOKBACK + predictions.shape[0]]["T"].to_numpy()

# # Get control readings
# control_actual = sensor_data_test.df.iloc[predict_from_i:predict_from_i + config.LOOKBACK + predictions.shape[0]]["C"].to_numpy() / 200 + 20

x_actual = t[0 : config.LOOKBACK + predictions.shape[0]]
x_prediction = t[config.LOOKBACK : config.LOOKBACK + predictions.shape[0]]
y_actual = y[0 : config.LOOKBACK + predictions.shape[0]]

print(x_actual.shape)
print(x_prediction.shape)
print(y_actual.shape)
print(y_prediction.shape)


plt.plot(x_actual, y_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
plt.plot(x_prediction, y_prediction, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
# plt.plot(x_actual, exttemp, label="External temperature", color="green", linestyle="-", marker="o")
# plt.plot(x_actual, control_actual, label="Control signal", color="green", linestyle="--")
plt.title('Autoregressive Temperature Predictions vs Actual Values')
plt.ylabel('Temperature')
plt.xlabel('Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
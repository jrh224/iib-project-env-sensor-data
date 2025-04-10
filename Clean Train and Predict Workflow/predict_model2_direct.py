import sys
import os

from matplotlib import pyplot as plt

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import config
from models import LSTMModel, Seq2SeqLSTM, Seq2SeqLSTMEncDec, Seq2SeqLSTMEncDec_CNN
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *
from utils.fake_data_gen import *


# length = 25920
# hours = 2160
# t = np.linspace(0, hours, length)
# test_matrix = gen_sum_of_consts(hours=hours, length=length, no_covariates=6, seed=51)
# test_matrix = gen_sum_of_consts_w_lag(hours=2160, length=25920, no_covariates=6, seed=51)
# test_matrix = gen_r2c2_w_irregular_heating_real_meteo(hours=2160, length=25920, seed=51) # 3 covariates

WITH_CNN = True

# scaler_file = "scalers_B0A732CC4A38.gz"

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')

# Add external temperature to sensor_data object
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
# Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

sensor_data_test, idx_blocks_test = sensor_data.filter_by_date_ranges(dates=config.TEST_RANGE, in_place=False)
test_matrix = sensor_data_test.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Test Matrix Created Successfully [Shape: {test_matrix.shape}]")

test_matrix_unscaled = test_matrix.copy()


scalers = joblib.load(config.SCALER_FILE) # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time


test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=None)

# Create dataset for training
test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)
print(f"EncDec inputs successfully generated. EncInp: {test_enc_inp.shape}, DecInp: {test_dec_inp.shape}, Targets: {test_targets.shape}")

# Initialise the model for prediction
if WITH_CNN:
    model = Seq2SeqLSTMEncDec_CNN()
else:
    model = Seq2SeqLSTMEncDec()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()

start_time = pd.to_datetime("2024-12-08 04:30:00").tz_localize("Europe/London").tz_convert("UTC")
# start_time = pd.to_datetime("2025-02-22 11:00:00").tz_localize("Europe/London").tz_convert("UTC")
start_point=sensor_data_test.df.index.get_loc(start_time)
print("Start index: ", start_point)

tlookback = start_point
t0 = tlookback+config.LOOKBACK
thorizon = t0+config.HORIZON

enc_inp = torch.tensor(test_matrix[tlookback:t0, :], dtype=torch.float32) # All features, before t=0 (start_point+lookback)
dec_inp = torch.tensor(test_matrix[t0:thorizon, 1:], dtype=torch.float32) # Not including feature 0 (the target)

predictions = model(enc_inp, dec_inp)
predictions = predictions.detach().cpu().numpy()  # Convert to NumPy array


# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
# print(y_prediction)

y_actual = test_matrix_unscaled[tlookback:thorizon, 0] # column 0 = targets
# print(y_actual)

# x_actual = t[tlookback : thorizon]
# x_prediction = t[t0 : thorizon]
x_actual = sensor_data_test.df.iloc[tlookback : thorizon].index
x_prediction = sensor_data_test.df.iloc[t0:thorizon].index


# print(x_actual.shape)
# print(x_prediction.shape)
# print(y_actual.shape)
# print(y_prediction.shape)

fontsize = 15
labelsize = 13
plt.plot(x_actual, y_actual, label='Ground truth temperature', color="blue", linestyle="-", marker="o")
plt.plot(x_prediction, y_prediction, label='Predicted temperature', color="red", linestyle="--", marker="x")
# plt.plot(x_actual, exttemp, label="External temperature", color="green", linestyle="-", marker="o")
# plt.plot(x_actual, control_actual, label="Control signal", color="green", linestyle="--")
plt.title('Predicted temperature vs ground truth (Model 3: Direct + CNN)', fontsize=fontsize)
plt.gca().set_xlabel("Time", fontsize=fontsize)
plt.gca().set_ylabel("Temperature °C", fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
plt.gca().tick_params(axis='y', labelsize=labelsize)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(fontsize=labelsize)
plt.gcf().set_tight_layout(True)
plt.gcf().set_figheight(5)
plt.gcf().set_figwidth(8)
plt.show()
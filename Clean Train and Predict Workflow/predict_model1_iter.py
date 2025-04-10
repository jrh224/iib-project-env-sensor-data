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
from models import LSTMModel, LSTM_CNN_Model, Seq2SeqLSTM, Seq2SeqLSTMEncDec
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *
from utils.fake_data_gen import *


# length = 25920
# hours = 2160
# t = np.linspace(0, hours, length)
# # test_matrix = gen_sum_of_consts_w_lag(hours=2160, length=25920, no_covariates=6, seed=51)
# test_matrix = gen_r2c2_w_irregular_heating_real_meteo(hours=2160, length=25920, seed=51) # 3 covariates
# test_matrix_unscaled = test_matrix.copy()

WITH_CNN = True # decide whether to include the CNN layer or not


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


# test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=None)
X_test, y_test = create_sequences(test_matrix, lookback=config.LOOKBACK, horizon=1, stride=config.STRIDE, target_col=0, blocks=None)

# test_dataset = TensorDataset(X_test, y_test)
# train_loader = DataLoader(list(zip(X_test, y_test)), batch_size=config.BATCH_SIZE, shuffle=True)


print(f"LSTM inputs and targets successfully generated. X_train: {X_test.shape}, y_train: {y_test.shape}")

# Initialise the model for prediction
# model = Seq2SeqLSTMEncDec()
if WITH_CNN:
    model = LSTM_CNN_Model()
else:
    model = LSTMModel()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()

start_time = pd.to_datetime("2025-01-21 21:00:00").tz_localize("Europe/London").tz_convert("UTC")
start_point=sensor_data_test.df.index.get_loc(start_time)

no_predictions = 96
# no_predictions = config.HORIZON
predictions = autoregressive_predict(model, test_matrix, no_predictions, start_point=start_point)

# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
print(y_prediction)

y_actual = test_matrix_unscaled[start_point:start_point + config.LOOKBACK + y_prediction.shape[0], 0]
print(y_actual)

tlookback = start_point
t0 = tlookback+config.LOOKBACK
thorizon = t0+no_predictions

x_actual = sensor_data_test.df.iloc[tlookback : thorizon].index
x_prediction = sensor_data_test.df.iloc[t0:thorizon].index

fontsize = 15
labelsize = 13
plt.plot(x_actual, y_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
plt.plot(x_prediction, y_prediction, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
# plt.plot(x_actual, exttemp, label="External temperature", color="green", linestyle="-", marker="o")
# plt.plot(x_actual, control_actual, label="Control signal", color="green", linestyle="--")
plt.title('Autoregressive Temperature Predictions vs Actual Values', fontsize=fontsize)
plt.gca().set_xlabel("Time", fontsize=fontsize)
plt.gca().set_ylabel("Temperature °C", fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
plt.gca().tick_params(axis='y', labelsize=labelsize)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(fontsize=labelsize)
plt.tight_layout()
plt.gcf().set_figheight(5)
plt.gcf().set_figwidth(8)
plt.show()
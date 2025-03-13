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
from models import LSTMModel, Seq2SeqLSTM, Seq2SeqLSTMEncDec
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *
from utils.fake_data_gen import *


length = 25920
hours = 2160
t = np.linspace(0, hours, length)
# test_matrix = gen_sum_of_consts_w_lag(hours=2160, length=25920, no_covariates=6, seed=51)
test_matrix = gen_sum_of_exp(hours=2160, length=25920, no_covariates=6, seed=51)
test_matrix_unscaled = test_matrix.copy()


scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
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
model = LSTMModel()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()

start_point=4510
predictions = autoregressive_predict(model, test_matrix, config.HORIZON, start_point=start_point)


# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
print(y_prediction)

y_actual = test_matrix_unscaled[start_point:start_point + config.LOOKBACK + y_prediction.shape[0], 0]
print(y_actual)

x_actual = t[start_point : start_point + config.LOOKBACK + y_prediction.shape[0]]
x_prediction = t[start_point + config.LOOKBACK : start_point + config.LOOKBACK + y_prediction.shape[0]]


# print(x_actual.shape)
# print(x_prediction.shape)
# print(y_actual.shape)
# print(y_prediction.shape)


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
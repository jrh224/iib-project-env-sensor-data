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
# test_matrix = gen_sum_of_consts(hours=hours, length=length, no_covariates=6, seed=51)
# test_matrix = gen_sum_of_consts_w_lag(hours=2160, length=25920, no_covariates=6, seed=51)
test_matrix = gen_sum_of_exp(hours=2160, length=25920, no_covariates=6, seed=51)
test_matrix_unscaled = test_matrix.copy()


scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time


test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=None)

# Create dataset for training
test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)
print(f"EncDec inputs successfully generated. EncInp: {test_enc_inp.shape}, DecInp: {test_dec_inp.shape}, Targets: {test_targets.shape}")

# Initialise the model for prediction
model = Seq2SeqLSTMEncDec()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()

start_point=4998

tlookback = start_point
t0 = tlookback+config.LOOKBACK
thorizon = t0+config.HORIZON

enc_inp = torch.tensor(test_matrix[tlookback:t0, :], dtype=torch.float32) # All features, before t=0 (start_point+lookback)
dec_inp = torch.tensor(test_matrix[t0:thorizon, 1:], dtype=torch.float32) # Not including feature 0 (the target)


predictions = model(enc_inp, dec_inp)
predictions = predictions.detach().cpu().numpy()  # Convert to NumPy array


# Inverse transform predictions to get correct scale
y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
print(y_prediction)

y_actual = test_matrix_unscaled[tlookback:thorizon, 0] # column 0 = targets
print(y_actual)

x_actual = t[tlookback : thorizon]
x_prediction = t[t0 : thorizon]


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
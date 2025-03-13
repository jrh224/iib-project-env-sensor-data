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
from utils.fake_data_gen import *


length = 25920
hours = 2160
t = np.linspace(0, hours, length)
# test_matrix = gen_sum_of_consts(hours=hours, length=length, no_covariates=6, seed=51)
# test_matrix = gen_sum_of_consts_w_lag(hours=hours, length=length, no_covariates=6, seed=51)
test_matrix = gen_sum_of_exp(hours=hours, length=length, no_covariates=6, seed=51)
test_matrix_unscaled = test_matrix.copy()


scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time


# Initialise the model for prediction
model = LSTMModel()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Pass through train dataset, predicting one hour every half hour


y_predictions = []
y_actuals = []

# assuming i_start is the beginning of the lookback period
for i_start in range(0, test_matrix.shape[0] - config.LOOKBACK - config.HORIZON, config.STRIDE): # stride = 1
    y_prediction = autoregressive_predict(model, test_matrix, config.HORIZON, i_start)
    y_prediction = scalers[0].inverse_transform(np.array(y_prediction).reshape(-1, 1))
    y_predictions.extend(np.array(y_prediction).reshape(-1).flatten())

    y_actual = test_matrix_unscaled[i_start+config.LOOKBACK:i_start+config.LOOKBACK+config.HORIZON, 0]
    y_actuals.extend(np.array(y_actual).reshape(-1).flatten())


y_predictions = np.array(y_predictions)
y_actuals = np.array(y_actuals)

print(y_predictions[0])
print(y_actuals[0])

# Calculate MAE and R2 from y_prediction and y_actual
rmse = np.sqrt(np.mean((y_actuals - y_predictions) ** 2))

mae = np.mean(np.abs(y_actuals - y_predictions))
y_mean = np.mean(y_actuals)
r2 = 1 - np.sum((y_actuals - y_predictions)**2) / np.sum((y_actuals - y_mean)**2)

print("MAE (mean absolute error): " + str(mae))
print("R2 (coefficient of determination): " + str(r2))
print("RMSE: " + str(rmse))
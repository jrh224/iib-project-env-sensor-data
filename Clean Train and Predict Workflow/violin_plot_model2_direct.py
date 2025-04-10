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
from models import LSTMModel, LSTM_CNN_Model, Seq2SeqLSTM, Seq2SeqLSTMEncDec, Seq2SeqLSTMEncDec_CNN
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *
from utils.fake_data_gen import *

import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_test)

# Create dataset for training
test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)
print(f"EncDec inputs successfully generated. EncInp: {test_enc_inp.shape}, DecInp: {test_dec_inp.shape}, Targets: {test_targets.shape}")


# # Initialise model
if WITH_CNN:
    model = Seq2SeqLSTMEncDec_CNN()
else:
    model = Seq2SeqLSTMEncDec()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Pass through train dataset, predicting one hour every half hour


y_predictions = []
y_actuals = []

for enc_inp, dec_inp, target in test_dataset:
    # Make prediction
    with torch.no_grad():
        predictions = model(enc_inp, dec_inp)

    # Inverse transform predictions to get correct scale
    y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # y_prediction = np.array(predictions).reshape(-1, 1)
    # Get actual IAT readings
    y_actual = scalers[0].inverse_transform(np.array(target).reshape(-1, 1)).flatten()
    # y_actual = np.array(target).reshape(-1, 1)

    # Store y_prediction and y_actual for evaluation later on
    y_predictions.append(y_prediction)
    y_actuals.append(y_actual)


y_predictions = np.array(y_predictions)
y_actuals = np.array(y_actuals)

errors = np.abs(y_predictions - y_actuals)

print(y_predictions.shape)
print(y_actuals.shape)

print(y_predictions[0])
print(y_actuals[0])

# Convert to DataFrame for seaborn
df = pd.DataFrame(errors, columns=[f"T+{i+1}" for i in range(12)])

# Convert to long format
df_long = df.melt(var_name="Timestep", value_name="Error")

# Plot violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="Timestep", y="Error", data=df_long, inner="quartile", cut=0, bw_adjust=0.5)
fontsize = 15
labelsize = 13
plt.title("MAE Distribution Across Timesteps (Model 3: Direct + CNN)", fontsize=fontsize)
plt.gca().set_xlabel("Timestep into the future", fontsize=fontsize)
plt.gca().set_ylabel("MAE °C", fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
plt.gca().tick_params(axis='y', labelsize=labelsize)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gcf().set_tight_layout(True)

plt.show()

maes = np.mean(errors, axis=0)
print(maes)

stds = np.std(errors, axis=0)
print(stds)


# # Calculate MAE and R2 from y_prediction and y_actual
# rmse = np.sqrt(np.mean((y_actuals - y_predictions) ** 2))

# mae = np.mean(np.abs(y_actuals - y_predictions))
# y_mean = np.mean(y_actuals)
r2 = 1 - np.sum((y_actuals.flatten() - y_predictions.flatten())**2) / np.sum((y_actuals.flatten() - np.mean(y_actuals.flatten()))**2)
print(r2)


# print("MAE (mean absolute error): " + str(mae))
# print("R2 (coefficient of determination): " + str(r2))
# print("RMSE: " + str(rmse))
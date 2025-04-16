# Load trained model
# Run trained model to create lots of one hour predictions
# Use these one hour predictions



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

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

WITH_CNN = False # decide whether to include the CNN layer or not

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


past_covariates = []
residuals = []

for i_start in range(0, test_matrix.shape[0] - config.LOOKBACK - config.HORIZON, config.STRIDE): # stride = 1

    current_lookback = test_matrix[i_start:config.LOOKBACK+i_start, :] # Features being used for predictions. Shape: (lookback, # features) e.g. (12, 7)
    past_covariates.append(current_lookback)


    y_prediction = autoregressive_predict(model, test_matrix, config.HORIZON, i_start)
    y_prediction = scalers[0].inverse_transform(np.array(y_prediction).reshape(-1, 1))
    y_prediction = np.array(y_prediction).reshape(-1)
    
    y_actual = test_matrix_unscaled[i_start+config.LOOKBACK:i_start+config.LOOKBACK+config.HORIZON, 0]
    y_actual = np.array(y_actual).reshape(-1)

    residual = np.abs(y_actual - y_prediction)
    residuals.append(residual)

past_covariates = np.array(past_covariates) # (# samples, lookback, # features) e.g. (2568, 12, 7)
residuals = np.array(residuals) # (# samples, # datapoints) e.g. (2568, 12)



# Now train XGBoost using this:

# Flatten the past covariates
X = past_covariates.reshape(past_covariates.shape[0], -1)  # (2568, 84)
y = residuals  # (2568, 12)

# Optional: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

# Wrap with MultiOutputRegressor to predict multiple outputs (12 residuals)
multi_output_model = MultiOutputRegressor(xgb_model)

# Train the model
multi_output_model.fit(X_train, y_train)

# Evaluate
y_pred = multi_output_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE on Test Set:", mae) # find mae

# Measure the magnitude of actual residuals (from the base model)
true_residual_magnitudes = np.mean(np.abs(y_test), axis=1)  # (samples,)

# Get indices of samples with lowest and highest errors
num_samples = 2  # adjust how many to plot

lowest_residual_indices = np.argsort(true_residual_magnitudes)[:num_samples]
highest_residual_indices = np.argsort(true_residual_magnitudes)[-num_samples:]

fontsize = 15
labelsize = 13

# def plot_samples(indices, title_prefix):
#     for i, idx in enumerate(indices):
#         true_residual = y_test[idx]
#         predicted_residual = y_pred[idx]
#         mean_true = np.mean(np.abs(true_residual))
#         mean_pred = np.mean(np.abs(predicted_residual))

#         plt.subplot(len(indices), 1, i + 1)
#         plt.plot(true_residual, label='True Residuals', marker='o')
#         plt.plot(predicted_residual, label='Predicted Residuals', marker='x')
#         print(f'{title_prefix} | Mean of true residual: {mean_true:.4f} | Mean of predicted residual: {mean_pred:.4f}')
#         # plt.title(
#         #     f'{title_prefix}',
#         #     fontsize=fontsize
#         # )
#         plt.gca().set_xlabel("Timestep", fontsize=fontsize)
#         plt.gca().set_ylabel("Residual °C", fontsize=fontsize)
#         plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
#         plt.gca().tick_params(axis='y', labelsize=labelsize)
#         plt.grid(True)
#         # Add legend only to the top subplot
#         if i == 0:
#             plt.legend(loc='upper left', fontsize=labelsize)  # legend in bottom right
#     plt.gcf().set_figheight(5)
#     plt.gcf().set_figwidth(10)
#     plt.tight_layout()
#     plt.show()

# # Plot lowest-residual (base model was accurate) samples
# plot_samples(lowest_residual_indices, title_prefix="Smallest Base Residuals")

# # Plot highest-residual (base model failed) samples
# plot_samples(highest_residual_indices, title_prefix="Largest Base Residuals")






# Compute mean absolute residual per sample
true_means = np.mean(np.abs(y_test), axis=1)
predicted_means = np.mean(np.abs(y_pred), axis=1)

# Sort samples by true residual mean
sorted_indices = np.argsort(true_means)
sorted_true = true_means[sorted_indices]
sorted_predicted = predicted_means[sorted_indices]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(sorted_true, label='Mean of true residual|', marker='', linewidth=2)
plt.plot(sorted_predicted, label='Mean of predicted residual|', marker='', linewidth=2)
plt.gca().set_xlabel("Sample Index (sorted by true residual)", fontsize=fontsize)
plt.gca().set_ylabel("Mean Absolute Residual °C", fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
plt.gca().tick_params(axis='y', labelsize=labelsize)
plt.title('True vs Predicted Residual Magnitudes Across Samples', fontsize=fontsize)
plt.legend(loc='upper left', fontsize=labelsize)
plt.grid(True)
plt.tight_layout()
plt.show()


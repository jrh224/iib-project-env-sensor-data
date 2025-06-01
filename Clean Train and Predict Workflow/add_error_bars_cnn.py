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
from models import Seq2SeqLSTMEncDec, Seq2SeqLSTMEncDec_CNN
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *
from utils.fake_data_gen import *

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
# X_test, y_test = create_sequences(test_matrix, lookback=config.LOOKBACK, horizon=1, stride=config.STRIDE, target_col=0, blocks=None)

test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)
# train_loader = DataLoader(list(zip(X_test, y_test)), batch_size=config.BATCH_SIZE, shuffle=True)

# print(f"LSTM inputs and targets successfully generated. X_train: {X_test.shape}, y_train: {y_test.shape}")
print(f"EncDec inputs successfully generated. EncInp: {test_enc_inp.shape}, DecInp: {test_dec_inp.shape}, Targets: {test_targets.shape}")


# Initialise the model for prediction
if WITH_CNN:
    model = Seq2SeqLSTMEncDec_CNN()
else:
    model = Seq2SeqLSTMEncDec()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Perform predictions
model.eval()


past_covariates = []
residuals = []
# Also collect predictions and targets to support error bar plotting
y_predictions = []
y_actuals = []

# Ensure model is in evaluation mode and on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# No gradients needed during inference
with torch.no_grad():
    for i, _ in enumerate(test_dataset):
        enc_inp, dec_inp, target = test_dataset[i]

        # Move tensors to device
        enc_inp = enc_inp.unsqueeze(0).to(device)  # Add batch dimension
        dec_inp = dec_inp.unsqueeze(0).to(device)
        target = target.cpu().numpy()

        # Flatten and save encoder input as past covariates
        flattened_input = enc_inp.cpu().numpy().flatten()
        past_covariates.append(flattened_input)

        # Run prediction
        prediction = model(enc_inp, dec_inp)
        prediction = prediction.squeeze().cpu().numpy()

        # If prediction was scaled, inverse transform it
        prediction_unscaled = scalers[0].inverse_transform(prediction.reshape(-1, 1)).flatten()
        target_unscaled = scalers[0].inverse_transform(target.reshape(-1, 1)).flatten()

        y_predictions.append(prediction_unscaled)
        y_actuals.append(target_unscaled)

        # Compute residuals
        residual = np.abs(target_unscaled - prediction_unscaled)
        residuals.append(residual)

# Convert lists to numpy arrays
past_covariates = np.array(past_covariates)  # shape: (num_samples, enc_inp_features * lookback)
residuals = np.array(residuals)              # shape: (num_samples, horizon)
y_predictions = np.array(y_predictions)
y_actuals = np.array(y_actuals)



# Now train XGBoost using this:

# Flatten the past covariates
X = past_covariates.reshape(past_covariates.shape[0], -1)  # (2568, 84)
y = residuals  # (2568, 12)


# Create indices for tracking
original_indices = np.arange(X.shape[0])

# Split data and indices together
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, original_indices, test_size=0.2, random_state=42
)


# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
multi_output_model = MultiOutputRegressor(xgb_model)

# Train the model
multi_output_model.fit(X_train, y_train)

# Use the XGBoost model to predict residuals for all test samples
predicted_residuals = multi_output_model.predict(X_test)  # shape: (n_samples, horizon)
mae = mean_absolute_error(y_test, predicted_residuals)
print("MAE on Test Set:", mae) # find mae

# Measure the magnitude of actual residuals (from the base model)
true_residual_magnitudes = np.mean(np.abs(y_test), axis=1)  # (samples,)

# Get indices of samples with lowest and highest errors
num_samples = 3

lowest_residual_indices = np.argsort(true_residual_magnitudes)[:num_samples]
highest_residual_indices = np.argsort(true_residual_magnitudes)[-num_samples:]


fontsize = 15
labelsize = 13

def plot_with_predicted_residuals(y_preds_all, y_actuals_all, predicted_residuals, original_test_indices, selected_test_indices, title_prefix="Sample", multiplier=1.5):
    for i, local_test_idx in enumerate(selected_test_indices):
        # Map to full dataset index
        full_idx = original_test_indices[local_test_idx]

        y_pred = y_predictions[full_idx]
        y_act = y_actuals[full_idx]
        y_err = predicted_residuals[local_test_idx] * multiplier

        plt.figure(figsize=(10, 5))
        plt.errorbar(np.arange(len(y_pred)), y_pred, yerr=y_err, fmt='-o', label='Predicted Temp + Error Bars', capsize=5)
        plt.plot(np.arange(len(y_act)), y_act, '-x', label='Actual Temp')
        plt.title(f"{title_prefix} {i+1}", fontsize=fontsize)
        plt.gca().set_xlabel("Timestep (5 mins / interval)", fontsize=fontsize)
        plt.gca().set_ylabel("Temperature °C", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=labelsize)
        plt.gca().tick_params(axis='y', labelsize=labelsize)
        plt.legend(fontsize=labelsize)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

multiplier = 2

# plot_with_predicted_residuals(
#     y_predictions,
#     y_actuals,
#     predicted_residuals,
#     original_test_indices=idx_test,
#     selected_test_indices=lowest_residual_indices,
#     title_prefix="Lowest Residual Test Sample",
#     multiplier=multiplier
# )

plot_with_predicted_residuals(
    y_predictions,
    y_actuals,
    predicted_residuals,
    original_test_indices=idx_test,
    selected_test_indices=highest_residual_indices,
    title_prefix="Highest Residual Test Sample",
    multiplier=multiplier
)



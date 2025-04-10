import sys
import os
# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.CustomDataframe import CustomDataframe
import config
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from utils.helper_functions import *
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import TensorDataset
from sklearn.metrics import root_mean_squared_error
import optuna

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')

# Add external temperature to sensor_data object
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
# Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

sensor_data_train, idx_blocks_train = sensor_data.filter_by_date_ranges(dates=config.TRAIN_RANGE, in_place=False)
train_matrix = sensor_data_train.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Train Matrix Created Successfully [Shape: {train_matrix.shape}]")

sensor_data_val, idx_blocks_val = sensor_data.filter_by_date_ranges(dates=config.VALID_RANGE, in_place=False)
val_matrix = sensor_data_val.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Validation Matrix Created Successfully [Shape: {val_matrix.shape}]")

# Scale each column using a different scaler
scalers = []
for i in range(train_matrix.shape[1]): # for each feature column
    scaler = MinMaxScaler()
    train_matrix[:, i] = scaler.fit_transform(train_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time
    val_matrix[:, i] = scaler.transform(val_matrix[:, i].reshape(-1, 1)).flatten() 
    scalers.append(scaler)
joblib.dump(scalers, 'real_data_scalers.gz') # Store all the scalers in order to be used when predicting


train_enc_inp, train_dec_inp, train_targets = get_encdec_inputs(train_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_train)
val_enc_inp, val_dec_inp, val_targets = get_encdec_inputs(val_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_val)
print(f"EncDec inputs successfully generated. EncInp: {train_enc_inp.shape}, DecInp: {train_dec_inp.shape}, Targets: {train_targets.shape}")

# Create dataset for training
train_dataset = TensorDataset(train_enc_inp, train_dec_inp, train_targets)
val_dataset = TensorDataset(val_enc_inp, val_dec_inp, val_targets)

X_train, y_train = get_xgboost_inputs(train_dataset)
X_val, y_val = get_xgboost_inputs(val_dataset)
# X_train.shape = (num_samples, num_features)
# y_train.shape = (num_samples, num_targets)
print("X_train generated with shape: ", X_train.shape)
print("y_train generated with shape: ", y_train.shape)
print("X_val generated with shape: ", X_val.shape)
print("y_val generated with shape: ", y_val.shape)


# Perform hyperparameter optimisation
# Define the objective function
def objective(trial):
    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # Print the hyperparameters being tried
    print(f"Trial {trial.number}: {params}")

    # Create base model
    base_model = xgb.XGBRegressor(objective="reg:squarederror", **params)
    # Wrap it in MultiOutputRegressor
    model = MultiOutputRegressor(base_model)
    # Train model
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)
    # Calculate mean squared error
    score = root_mean_squared_error(y_val, y_pred)  # RMSE
    return score


# Run Optuna Optimization
study = optuna.create_study(direction="minimize")  # Minimize RMSE
study.optimize(objective, n_trials=50)




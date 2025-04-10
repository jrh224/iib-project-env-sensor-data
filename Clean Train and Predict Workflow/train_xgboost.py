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

save_model_as = "optimal_multioutput_xgb_model.pkl"

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
print(f"Train EncDec inputs successfully generated. EncInp: {train_enc_inp.shape}, DecInp: {train_dec_inp.shape}, Targets: {train_targets.shape}")

# Create dataset for training
train_dataset = TensorDataset(train_enc_inp, train_dec_inp, train_targets)

X_train, y_train = get_xgboost_inputs(train_dataset)
# X_train.shape = (num_samples, num_features)
# y_train.shape = (num_samples, num_targets)
print("X_train generated with shape: ", X_train.shape)
print("y_train generated with shape: ", y_train.shape)

# Define hyperparameters for training xgboost (can be found with hp optimiser)
params = {'n_estimators': 684, 'max_depth': 3, 'learning_rate': 0.09286780707141348, 'subsample': 0.9413904311355064, 'colsample_bytree': 0.978833768039749, 'reg_alpha': 0.005547173478936673, 'reg_lambda': 6.406223715025617e-06}

# Train XGBoost model with MultiOutputRegressor
base_model = xgb.XGBRegressor(objective='reg:squarederror', **params)
trained_model = MultiOutputRegressor(base_model).fit(X_train, y_train)

# Save the entire MultiOutputRegressor model
joblib.dump(trained_model, save_model_as)



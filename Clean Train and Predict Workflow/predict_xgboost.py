import sys
import os
# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.CustomDataframe import CustomDataframe
import config
import joblib
import numpy as np
from utils.helper_functions import *
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt

model_filepath = "optimal_multioutput_xgb_model.pkl"

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
print(f"Train Matrix Created Successfully [Shape: {test_matrix.shape}]")

scalers = joblib.load('real_data_scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time

test_enc_inp, test_dec_inp, test_targets = get_encdec_inputs(test_matrix, lookback=config.LOOKBACK, horizon=config.HORIZON, stride=config.STRIDE, target_col=0, blocks=idx_blocks_test)

# Create dataset for training
test_dataset = TensorDataset(test_enc_inp, test_dec_inp, test_targets)
print(f"EncDec inputs successfully generated. EncInp: {test_enc_inp.shape}, DecInp: {test_dec_inp.shape}, Targets: {test_targets.shape}")

X_test, y_test = get_xgboost_inputs(test_dataset)
print("X_test generated with shape: ", X_test.shape)
print("y_test generated with shape: ", y_test.shape)

# Load the saved model
loaded_model = joblib.load(model_filepath)

# Predict for all samples in one go
y_pred = loaded_model.predict(X_test)

print(y_pred.shape)
y_pred = scalers[0].inverse_transform(np.array(y_pred))

for i in range(0, len(y_pred), 6): # Increase in half hour increments
    plt.clf()
    x_predict = sensor_data_test.df.iloc[i + config.LOOKBACK:i + config.LOOKBACK + config.HORIZON].index.to_numpy()
    x_actual = sensor_data_test.df.iloc[i:i + config.LOOKBACK + config.HORIZON].index.to_numpy()
    y_actual = sensor_data_test.df.iloc[i:i + config.LOOKBACK + config.HORIZON]["T"].to_numpy()

    fontsize = 14
    labelsize = 13
    plt.title("XGBoost: Prediction vs ground truth IAT", fontsize=fontsize)
    plt.gca().set_xlabel("Time", fontsize=fontsize)
    plt.gca().set_ylabel("Temperature C", fontsize=fontsize)
    plt.plot(x_actual, y_actual, label="Actual temperature", color="blue", linestyle="-", marker="o")
    plt.plot(x_predict, y_pred[i], label="Predicted temperature", color="red", linestyle="--", marker="x")
    plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
    plt.gca().tick_params(axis='y', labelsize=labelsize)
    plt.xticks(rotation=45)
    plt.legend(fontsize=labelsize)
    plt.gcf().set_tight_layout(True)
    plt.show()
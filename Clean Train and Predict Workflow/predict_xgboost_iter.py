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
import pandas as pd

model_filepath = "optimal_singleoutput_xgb_model_room1.pkl"

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

scalers = joblib.load(config.SCALER_FILE) # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time


# # Load data and create sequences
# X_test, y_test = create_sequences(test_matrix, config.LOOKBACK, 1, config.STRIDE, 0, blocks=idx_blocks_test)
# # flatten to gboost format
# X_test, y_test = get_xgboost_inputs_iter(X_test, y_test)


# Load the saved model
loaded_model = joblib.load(model_filepath)

start_time = pd.to_datetime("2024-12-07 00:05:00").tz_localize("Europe/London").tz_convert("UTC")
start_point_loop=sensor_data_test.df.index.get_loc(start_time)

for i in range(start_point_loop, 10000, 6):
    start_point=i

    no_predictions = 12
    y_pred = xgboost_autoregressive_predict(loaded_model, test_matrix, no_predictions, start_point)

    y_pred = scalers[0].inverse_transform(np.array(y_pred).reshape(-1, 1)).reshape(-1)

    y_actual = sensor_data_test.df.iloc[start_point:start_point + config.LOOKBACK + no_predictions]["T"].to_numpy()
    x_predict = sensor_data_test.df.iloc[start_point + config.LOOKBACK:start_point + config.LOOKBACK + no_predictions].index.to_numpy()
    x_actual = sensor_data_test.df.iloc[start_point:start_point + config.LOOKBACK + no_predictions].index.to_numpy()


    plt.figure(figsize=(8, 5))
    fontsize = 15
    labelsize = 13
    plt.title("XGBoost: Prediction vs ground truth IAT", fontsize=fontsize)
    plt.gca().set_xlabel("Time", fontsize=fontsize)
    plt.gca().set_ylabel("Temperature C", fontsize=fontsize)
    plt.plot(x_actual, y_actual, label="Actual temperature", color="blue", linestyle="-", marker="o")
    plt.plot(x_predict, y_pred, label="Predicted temperature", color="red", linestyle="--", marker="x")
    plt.gca().tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks
    plt.gca().tick_params(axis='y', labelsize=labelsize)
    plt.xticks(rotation=45)
    plt.legend(fontsize=labelsize)
    plt.gcf().set_tight_layout(True)
    plt.grid(True)
    plt.show()

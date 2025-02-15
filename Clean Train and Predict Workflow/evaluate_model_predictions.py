import sys
import os

from matplotlib import pyplot as plt
# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import config
from models import LSTMModel
from utils.CustomDataframe import CustomDataframe
from utils.helper_functions import *


# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.TEST_FILENAME) # NB: change back to FILENAME if all in same file
sensor_data_test = sensor_data.filter_by_date(start_date=config.start_date_test, end_date=config.end_date_test, in_place=False)

test_matrix = sensor_data_test.create_pytorch_matrix(lat=config.LAT, long=config.LONG)

scalers = joblib.load('scalers.gz') # Load up the previously trained scalers
for i in range(test_matrix.shape[1]): # for each feature column
    scaler = scalers[i] # Use the appropriate scaler for each column
    test_matrix[:, i] = scaler.transform(test_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time

# Initialise the model for prediction
model = LSTMModel()
model.load_state_dict(torch.load(config.PREDICT_MODEL, weights_only=True))

# Pass through train dataset, predicting one hour every half hour

# Get the starting point for the predictions
# predict_from_i = sensor_data_test.df.index[sensor_data_test.df['datetime'] > config.PREDICT_FROM].to_list()[0]
predict_from_i = 0 # start from beginning of test set
df_length = sensor_data_test.df.shape[0]
rmse_values = []

while predict_from_i < df_length - config.NUM_PREDICTIONS - config.LOOKBACK:
    # Perform predictions
    predictions = autoregressive_predict(model, test_matrix, num_predictions=config.NUM_PREDICTIONS, start_point=predict_from_i)
    # Inverse transform predictions to get correct scale
    y_prediction = scalers[0].inverse_transform(np.array(predictions).reshape(-1, 1))
    # Get actual IAT readings
    y_actual = sensor_data_test.df.iloc[predict_from_i:config.LOOKBACK + config.NUM_PREDICTIONS + predict_from_i]["T"].to_numpy()
    rmse_values.append(np.sqrt(np.mean(y_actual - y_prediction) ** 2))

    # plt.clf()
    # x_actual = sensor_data_test.df.iloc[predict_from_i:config.LOOKBACK + len(predictions) + predict_from_i]['datetime'].to_numpy()
    # x_prediction = sensor_data_test.df.iloc[config.LOOKBACK + predict_from_i:config.LOOKBACK + len(predictions) + predict_from_i]['datetime'].to_numpy()
    # plt.plot(x_actual, y_actual, label='Actual Temperature', color="blue", linestyle="-", marker="o")
    # plt.plot(x_prediction, y_prediction, label='Autoregressive Predictions', color="red", linestyle="--", marker="x")
    # plt.show()

    predict_from_i += 120 # Add half an hour each time

rmse_values = np.array(rmse_values)
overall_rmse = np.sqrt(np.mean(rmse_values**2))
print("Overall RMSE:", overall_rmse)
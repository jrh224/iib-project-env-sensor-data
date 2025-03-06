import sys
import os

from matplotlib import pyplot as plt
import pickle

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from darts.models import RNNModel, BlockRNNModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pandas as pd

from utils.CustomDataframe import CustomDataframe
import config

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)
train_dataframes = sensor_data.filter_by_date_ranges(dates=config.TRAIN_RANGE, return_separate=True)
val_dataframes = sensor_data.filter_by_date_ranges(dates=config.VALID_RANGE, return_separate=True)

test_dataframes = sensor_data.filter_by_date_ranges(dates=config.TEST_RANGE, return_separate=True)

# Define the target column and covariate columns
target_column = "T" 
covariate_columns = ["C", "temperature_2m"] 

test_targets = TimeSeries.from_dataframe(test_dataframes[0], value_cols=target_column)
test_covariates = TimeSeries.from_dataframe(test_dataframes[0], value_cols=covariate_columns)

# Load scalers
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# Retrieve them
target_scaler = scalers["target_scaler"]
covariate_scaler = scalers["covariate_scaler"]

test_targets_scaled = target_scaler.transform(test_targets)
test_covariates_scaled = covariate_scaler.transform(test_covariates)
test_targets_scaled = test_targets_scaled.astype("float32")
test_covariates_scaled = test_covariates_scaled.astype("float32")


model = RNNModel.load("RNNModel_2025-03-06_14_58_30.pt")

# Force heating to max from point of prediction
df_covariates = test_covariates_scaled.pd_dataframe()
df_covariates.iloc[600:, 0] = 1  # Setting the control column (index 0) values after row 600 to 1
test_covariates_scaled = TimeSeries.from_dataframe(df_covariates)

# Make Predictions
prediction = model.predict(
    n=6, 
    # past_covariates = test_covariates_scaled,
    series = test_targets_scaled[:600],
    future_covariates = test_covariates_scaled)

test_targets_scaled.plot(label="actual")
prediction.plot(label="forecast")
plt.legend()
plt.show()
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


# Assuming `blocks_dfs` is the list of DataFrames you obtained from filter_by_date_ranges
# for i, block_df in enumerate(train_dataframes):
i=1
block_df = train_dataframes[0]
print(f"Training on Block {i+1}...")

# Convert each block into a TimeSeries object
block_target_series = TimeSeries.from_dataframe(block_df, value_cols=target_column)
block_covariate_series = TimeSeries.from_dataframe(block_df, value_cols=covariate_columns)

# Normalize data if needed
target_scaler = Scaler()
covariate_scaler = Scaler()
block_target_scaled = target_scaler.fit_transform(block_target_series)
block_covariate_scaled = covariate_scaler.fit_transform(block_covariate_series)


# Save both scalers (target and covariate)
with open("scalers.pkl", "wb") as f:
    pickle.dump({"target_scaler": target_scaler, "covariate_scaler": covariate_scaler}, f)

# Convert the scaled series to float32
block_target_scaled = block_target_scaled.astype("float32")
block_covariate_scaled = block_covariate_scaled.astype("float32")

# Initialize model (LSTM)
model = RNNModel(
    model="LSTM",
    input_chunk_length=12,  # Example: use past 12 time steps for training
    output_chunk_length=6,  # Example: predict the next 6 time steps
    random_state=42
)

# Train the model on the block data
# Train the model on the block data, using the target and covariates
model.fit(
    series=block_target_scaled,        # The target series
    # past_covariates = block_covariate_scaled 
    future_covariates=block_covariate_scaled  # The covariate series
)

model.save()


test_targets = TimeSeries.from_dataframe(test_dataframes[0], value_cols=target_column)
test_covariates = TimeSeries.from_dataframe(test_dataframes[0], value_cols=covariate_columns)

print(test_dataframes[0])

# ??? ValueError: For the given forecasting case, the provided future covariates at dataset index `0` do not extend far enough into the past. The future covariates must start at time step `2024-11-30 23:05:00`, whereas now they start at time step `2024-12-07 00:05:00`.

test_targets_scaled = target_scaler.transform(test_targets) # NB: Uses the last trained scaler
test_covariates_scaled = covariate_scaler.transform(test_covariates)

test_targets_scaled = test_targets_scaled.astype("float32")
test_covariates_scaled = test_covariates_scaled.astype("float32")


n = 6  # Number of time steps to drop from targets to predict (same as output chunk length)
test_targets_cut = test_targets_scaled.drop_after(test_targets_scaled.end_time() - n * test_targets_scaled.freq)

# Make Predictions
prediction = model.predict(
    n=n, 
    # past_covariates = test_covariates_scaled,
    series = test_targets_cut,
    future_covariates = test_covariates_scaled)

test_targets_scaled.plot(label="actual")
prediction.plot(label="forecast")
plt.legend()
plt.show()
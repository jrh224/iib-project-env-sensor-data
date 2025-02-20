import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './Clean Train and Predict Workflow/')))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from utils.CustomDataframe import CustomDataframe
import config

sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data_train = sensor_data.filter_by_date(start_date=config.start_date_train, end_date=config.end_date_train, in_place=False)

sensor_data_train.interpolate_missing_rows()
sensor_data_train.resample()

sensor_data_train.add_ext_temp_column(lat=config.LAT, long=config.LONG)
sensor_data_train.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

sensor_data_train.df.drop(['H', 'Dd'], axis=1, inplace=True)

print(sensor_data_train.df)

# Function to create windowed dataset with future covariates
def create_windowed_data_with_future_covariates(df, target_col, window_size, horizon):
    X, y = [], []
    # Identify covariates (all columns except the target column)
    covariate_cols = [col for col in df.columns if col != target_col]

    for t in range(len(df) - window_size - horizon): # repeat for each prediction window
        past_values = df[target_col].iloc[t:t + window_size].values  # Past target values
        past_covariates = df[covariate_cols].iloc[t:t + window_size].values.flatten()  # Past external features
        future_covariates = df[covariate_cols].iloc[t + window_size:t + window_size + horizon].values.flatten()  # Future external features
        future_values = df[target_col].iloc[t + window_size:t + window_size + horizon].values  # Future target values
        
        X.append(np.concatenate([past_values, past_covariates, future_covariates]))  # Concatenate all inputs
        y.append(future_values)  # Future target values
        
    return np.array(X), np.array(y)

# Create features and labels
X_train, y_train = create_windowed_data_with_future_covariates(sensor_data_train.df, target_col='T', window_size=config.LOOKBACK, horizon=config.OUTPUT_SIZE)

print(X_train.shape)
print(y_train.shape)

# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Train XGBoost model
# model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
# model.fit(X_train, y_train)

# Train XGBoost model with MultiOutputRegressor
base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
trained_model = MultiOutputRegressor(base_model).fit(X_train, y_train)




# Now load test data
sensor_data = CustomDataframe(filename=config.TEST_FILENAME)
sensor_data_test = sensor_data.filter_by_date(start_date=config.start_date_test, end_date=config.end_date_test, in_place=False)

sensor_data_test.interpolate_missing_rows()
sensor_data_test.resample()

sensor_data_test.add_ext_temp_column(lat=config.LAT, long=config.LONG)
sensor_data_test.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)

sensor_data_test.df.drop(['H', 'Dd'], axis=1, inplace=True)

print(sensor_data_test.df)

# Create features and labels
X_test, y_test = create_windowed_data_with_future_covariates(sensor_data_test.df, target_col='T', window_size=config.LOOKBACK, horizon=config.OUTPUT_SIZE)


# Predict multiple future steps in one go
y_pred = trained_model.predict(X_test)

for i in range(0, len(y_pred), 6): # Increase in half hour increments
    plt.clf()
    x = sensor_data_test.df.iloc[i + config.LOOKBACK:i + config.LOOKBACK + config.OUTPUT_SIZE].index.to_numpy()
    plt.xlabel("Time")
    plt.ylabel("Temperature C")
    plt.plot(x, y_test[i], label="Actual temperature", color="blue", linestyle="-", marker="o")
    plt.plot(x, y_pred[i], label="Predicted temperature", color="red", linestyle="--", marker="x")
    plt.legend()
    plt.show()

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.4f}')

# Evaluate model
r2 = r2_score(y_test, y_pred)
print(f'Coefficient of determination: {r2:.4f}')
import pandas as pd

# Configuration
LOOKBACK = 20 # 20 = 5 minutes
INPUT_SIZE = 5 # int tempt, ext temp, heating control, re, daylight
BATCH_SIZE = 32
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
TRAIN_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 5

# Model to use for prediction
PREDICT_MODEL = '14feb1518.pth'
NUM_PREDICTIONS = 240
PREDICT_FROM = pd.to_datetime("2024/11/14 00:00:00").tz_localize('UTC') # Start date

# Determine the train / test data source
FILENAME = 'wadham college/2025-02-07-18-26_influxdb_dataP1.csv' 
TEST_FILENAME = 'wadham college/2025-02-07-18-27_influxdb_dataP2.csv' # Split train and test here temporarily

# Define the start and end dates of the training and test data
start_date_train = pd.to_datetime("2024/11/01 00:00:15")
end_date_train = pd.to_datetime("2024/11/14 00:00:00")
start_date_test = pd.to_datetime("2024/11/14 00:00:15")
end_date_test = pd.to_datetime("2024/11/30 00:00:00")

# Define coordinates of room location
LAT = 51.756
LONG = 1.254
import pandas as pd

# Configuration
LOOKBACK = 288 # 288 = 24 hrs / 5 min timesteps
INPUT_SIZE = 7 # int tempt, ext temp, heating control, re, daylight, sin24hr, cos24hr
OUTPUT_SIZE = 1 # 12 * 5 min = 1 hour of predictions
BATCH_SIZE = 32
HIDDEN_SIZE = 512
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 50 # Paper suggests 60
TRAIN_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 5

# Model to use for prediction
PREDICT_MODEL = '15feb1757.pth'
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
import pandas as pd

# Configuration
LOOKBACK = 144 # 288 = 24 hrs / 5 min timesteps // set this to be 144 next time
INPUT_SIZE = 7 # int tempt, ext temp, heating control, re, daylight, sin24hr, cos24hr
OUTPUT_SIZE = 12 # 12 * 5 min = 1 hour of predictions
BATCH_SIZE = 32
HIDDEN_SIZE = 512
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 50 # Paper suggests 60
TRAIN_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 5
SEQ_STEP = 1 # =1 means no skipping start points in the data

# Model to use for prediction
PREDICT_MODEL = '26feb1433.pth'
NUM_PREDICTIONS = 240 # Not really used anymore
PREDICT_FROM = pd.to_datetime("2024/12/07 00:00:15").tz_localize('UTC') # Start date of lookback period

# Determine the train / test data source
FILENAME = 'wadham college/Dom Bursar 24Nov2024 to Feb2025.csv' 
TEST_FILENAME = 'wadham college/Dom Bursar 24Nov2024 to Feb2025.csv'

# Define the start and end dates of the training and test data
start_date_train = pd.to_datetime("2024/11/01 00:00:15")
end_date_train = pd.to_datetime("2024/11/14 00:00:00")
start_date_test = pd.to_datetime("2024/11/14 00:00:15")
end_date_test = pd.to_datetime("2024/11/30 00:00:00")

# FOR USE WITH 3 LARGER DATA SETS
TRAIN_RANGE = [
        ("2024-11-24 00:00:15", "2024-12-01 00:00:00"),
        ("2024-12-10 00:00:15", "2025-01-16 00:00:00"),
        ("2025-01-24 00:00:15", "2025-02-15 00:00:00")]
VALID_RANGE = [
    ("2024-12-01 00:00:15", "2024-12-07 00:00:00"),
    ("2025-01-16 00:00:15", "2025-01-21 00:00:00"),
    ("2025-02-15 00:00:15", "2025-02-21 00:00:00")
]
TEST_RANGE = [
    ("2024-12-07 00:00:15", "2024-12-10 00:00:00"),
    ("2025-01-21 00:00:15", "2025-01-24 00:00:00"),
    ("2025-02-21 00:00:15", "2025-02-24 00:00:00")
]

# Define coordinates of room location
LAT = 51.756
LONG = 1.254
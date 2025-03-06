# Author: Laura Kulowski

'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''



import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys
import os

from sklearn.preprocessing import MinMaxScaler

# Add the parent directory of 'my_package' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import generate_dataset
import lstm_encoder_decoder
import plotting
from utils.CustomDataframe import CustomDataframe 
from utils.helper_functions import get_encdec_inputs, my_windowed_dataset

matplotlib.rcParams.update({'font.size': 17})

#----------------------------------------------------------------------------------------------------------------
# # generate dataset for LSTM
# t, y = generate_dataset.synthetic_data()
# print(t.shape)
# print(y.shape)

# Import sensor data into CustomDataframe object
sensor_data = CustomDataframe(filename=config.FILENAME)
sensor_data.interpolate_missing_rows()
sensor_data.resample(freq='5Min')
sensor_data.add_ext_temp_column(lat=config.LAT, long=config.LONG)
sensor_data.add_sunrise_sunset_column(lat=config.LAT, long=config.LONG)
sensor_data_train, idx_blocks_train = sensor_data.filter_by_date_ranges(dates=config.TRAIN_RANGE, in_place=False)
train_matrix = sensor_data_train.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Train Matrix Created Successfully [Shape: {train_matrix.shape}]")

sensor_data_val, idx_blocks_val = sensor_data.filter_by_date_ranges(dates=config.VALID_RANGE, in_place=False)
val_matrix = sensor_data_val.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Validation Matrix Created Successfully [Shape: {val_matrix.shape}]")

sensor_data_test, idx_blocks_test = sensor_data.filter_by_date_ranges(dates=config.TEST_RANGE, in_place=False)
test_matrix = sensor_data_test.create_pytorch_matrix(lat=config.LAT, long=config.LONG)
print(f"Test Matrix Created Successfully [Shape: {test_matrix.shape}]")

# Scale each column using a different scaler
scalers = []
for i in range(train_matrix.shape[1]): # for each feature column
    scaler = MinMaxScaler()
    train_matrix[:, i] = scaler.fit_transform(train_matrix[:, i].reshape(-1, 1)).flatten() # scale one column at a time
    val_matrix[:, i] = scaler.transform(val_matrix[:, i].reshape(-1, 1)).flatten()  # Transform validation data
    scalers.append(scaler)
joblib.dump(scalers, 'scalers.gz') # Store all the scalers in order to be used when predicting



# t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, y, split = 0.8)

# # plot time series 
# plt.figure(figsize = (18, 6))
# plt.plot(t, y, color = 'k', linewidth = 2)
# plt.xlim([t[0], t[-1]])
# plt.xlabel('$t$')
# plt.ylabel('$y$')
# plt.title('Synthetic Time Series')
# import os

# # Create 'plots' directory if it doesn't exist
# if not os.path.exists("plots"):
#     os.makedirs("plots")

# plt.savefig('plots/synthetic_time_series.png')

# # plot time series with train/test split
# plt.figure(figsize = (18, 6))
# plt.plot(t_train, y_train, color = '0.4', linewidth = 2, label = 'Train') 
# plt.plot(np.concatenate([[t_train[-1]], t_test]), np.concatenate([[y_train[-1]], y_test]),
#          color = (0.74, 0.37, 0.22), linewidth = 2, label = 'Test')
# plt.xlim([t[0], t[-1]])
# plt.xlabel(r'$t$')
# plt.ylabel(r'$y$')
# plt.title('Time Series Split into Train and Test Sets')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.tight_layout
# plt.savefig('plots/train_test_split.png')

#----------------------------------------------------------------------------------------------------------------
# window dataset

# set size of input/output windows 
iw = 72 # 6 hrs
ow = 12 # 1 hr
s = 2 # every 10 mins

# generate windowed training/test datasets
Xtrain, Ytrain= generate_dataset.windowed_dataset(y_train, input_window = iw, output_window = ow, stride = s)
Xtest, Ytest = generate_dataset.windowed_dataset(y_test, input_window = iw, output_window = ow, stride = s)

Xtrain, Ytrain = my_windowed_dataset(test_matrix, lookback = iw, horizon = ow, stride = s, blocks=idx_blocks_test)

# # plot example of windowed data  
# plt.figure(figsize = (10, 6)) 
# plt.plot(np.arange(0, iw), Xtrain[:, 0, 0], 'k', linewidth = 2.2, label = 'Input')
# plt.plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, 0, 0]], Ytrain[:, 0, 0]]),
#          color = (0.2, 0.42, 0.72), linewidth = 2.2, label = 'Target')
# plt.xlim([0, iw + ow - 1])
# plt.xlabel(r'$t$')
# plt.ylabel(r'$y$')
# plt.title('Example of Windowed Training Data')
# plt.legend(bbox_to_anchor=(1.3, 1))
# plt.tight_layout() 
# plt.savefig('plots/windowed_data.png')

# #----------------------------------------------------------------------------------------------------------------
# # LSTM encoder-decoder

# # convert windowed data from np.array to PyTorch tensor
# X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# # specify model parameters and train
# model = lstm_encoder_decoder.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 15)
# loss = model.train_model(X_train, Y_train, n_epochs = 50, target_len = ow, batch_size = 5, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

# # plot predictions on train/test data
# plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)

# plt.close('all')


# 15th December - Adding RNN to try to learn how the temp decay curves behave, using start_temp and outside temperature
# Include an LTSM layer to allow the model to remember previous states (prevents the model from being a markov chain)

import numpy as np
import pandas as pd
from datetime import datetime
from CustomDataframe import CustomDataframe
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from openmeteocustom import get_external_temp

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data.filter_by_date(days=3)

# 40 exceptions = 10 minutes allowed exception ; 240 = one hour decay curves minimum
all_sequences = sensor_data.get_temp_curves_for_training(column='T', pos_or_neg='neg', max_exceptions=40, response_duration=10800, plot_sequences=True)

hourly_ext_temps = get_external_temp(start_date=all_sequences[0][0].date().isoformat(), end_date=all_sequences[-1][1].date().isoformat())

start_temps = []
X = []  # Input for RNN
y = []  # Target decay curves (labels)
for sequence in all_sequences:
    # Extract T_smoothed (dynamic temperature curve) for this sequence
    temp_curve = sensor_data.df[(sensor_data.df["datetime"] >= sequence[0]) & 
                                (sensor_data.df["datetime"] <= sequence[1])]["T_smoothed"].values
    
    hours = sensor_data.df[(sensor_data.df["datetime"] >= sequence[0]) & 
                                (sensor_data.df["datetime"] <= sequence[1])]["datetime"].to_numpy()
    floored_hours = [pd.Timestamp(dt).floor('h') for dt in hours]

    # Extract corresponding external temperatures
    ext_temps = np.array(hourly_ext_temps.set_index("date").reindex(floored_hours)["temperature_2m"].values)


    assert len(temp_curve) == len(ext_temps), "Mismatch in sequence lengths!"

    # Create input features: concatenate static (start_temp) with dynamic (external_temp)
    start_temp = temp_curve[0]  # Static feature
    input_features = np.column_stack([np.full_like(ext_temps, start_temp), ext_temps])  # Shape: (sequence_length, 2)

    # Store input sequence and corresponding target
    X.append(input_features)  
    y.append(temp_curve)



# Assuming X is a list of sequences where each sequence is a tensor with the first column being initial temperature
# and the remaining columns are the external temperatures.
X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
# Pad sequences with NaN instead of 0
X_tensor = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=float('nan'))


y = [torch.tensor(seq, dtype=torch.float32) for seq in y]
y_tensor = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=float('nan'))

_, max_sequence_length = y_tensor.size()

# Train-test split
n_curves = len(all_sequences)
train_size = int(0.8 * n_curves)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]


# Define the RNN model
class TempRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TempRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # Ignore hidden state
        out = self.fc(out[:, -1, :])  # Only take the output of the last time step
        return out

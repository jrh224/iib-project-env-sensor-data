import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.CustomDataframe import CustomDataframe


sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data_train = sensor_data.filter_by_date(start_date="10/09/2024 14:00", end_date="10/16/2024 14:00", in_place=False)
sensor_data_test = sensor_data.filter_by_date(start_date="10/16/2024 14:00", end_date="10/23/2024 14:00", in_place=False)

scaler = StandardScaler()
scaler.fit(sensor_data_train.df[["T"]])  # Fit on the training temperature data

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=16, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = LSTMModel(input_size=1)
model.load_state_dict(torch.load('saved_model_5.pth', weights_only=True))
model.eval()

# Now let's test the model
seq_size = 20
x_data = []
y_data = []
for i in range(sensor_data_test.df.shape[0] - seq_size):
    window = [[x] for x in sensor_data_test.df.iloc[i:i+seq_size]["T"].to_numpy()]
    after_window = sensor_data_test.df.iloc[i + seq_size]["T"]
    x_data.append(window)
    y_data.append(after_window)
x_test = torch.tensor(x_data, dtype=torch.float32)
y_test = torch.tensor(y_data, dtype=torch.float32)

outputs = model(x_test)
predictions = outputs.squeeze().tolist()

rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
print(f"Score (RMSE): {rmse:.4f}")

# x = np.arange(0, len(x_test[0])+1, 1)
# plt.scatter(x[:len(x_test[0]+1)], x_test[0], label="Actual temperature input", color="blue")
# plt.scatter(x[len(x_test[0]+1)], y_test[0], label="Actual next timestep", color="green")
# plt.scatter(x[len(x_test[0]+1)], predictions[0], label="Predicted next timestep", color="red")
# plt.gca().set_xlabel("Timestep", fontsize=14)
# plt.gca().set_ylabel("Temperature", fontsize=14)
# plt.legend()
# plt.show()


# Convert predictions back to original scale
# y_test_real = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
# predictions_real = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
y_test_real = y_test.numpy().reshape(-1, 1).flatten()
predictions_real = np.array(predictions).reshape(-1, 1).flatten()



# Create the plot
plt.figure(figsize=(12, 6))

# Plot the actual temperature values
plt.plot(range(len(y_test_real)), y_test_real, label="Actual Temperature", color="blue", linestyle="-", marker="o")

# Plot the predicted temperature values
plt.plot(range(len(predictions_real)), predictions_real, label="Predicted Temperature", color="red", linestyle="--", marker="x")

# Labels and title
plt.xlabel("Timestep", fontsize=14)
plt.ylabel("Temperature", fontsize=14)
plt.title("Temperature Predictions vs Actual Values", fontsize=16)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
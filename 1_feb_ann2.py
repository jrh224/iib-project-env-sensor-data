import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.CustomDataframe import CustomDataframe
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data_train = sensor_data.filter_by_date(start_date="10/09/2024 14:00", end_date="10/16/2024 14:00", in_place=False)
sensor_data_test = sensor_data.filter_by_date(start_date="10/16/2024 14:00", end_date="10/23/2024 14:00", in_place=False)


scaler = StandardScaler()
scaler.fit(sensor_data_train.df[["T"]])  # Fit on the training temperature data

# Dimensions of the training set ( x ):

# Axis 1: Training set elements (sequences) (must be of the same size as  y  size)
# Axis 2: Members of sequence
# Axis 3: Features in data (like input neurons)

x_data = []
y_data = []
seq_size = 20 # 5 minutes of data for each prediction
for i in range(sensor_data_train.df.shape[0] - seq_size):
    window = [[x] for x in sensor_data_train.df.iloc[i:i+seq_size]["T"].to_numpy()]
    # window = scaler.transform(window)  # Scale input window
    after_window = sensor_data_train.df.iloc[i + seq_size]["T"]

    # after_window = scaler.transform(np.array([[after_window]]))[0, 0]

    x_data.append(window)
    y_data.append(after_window)
x = torch.tensor(x_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32)

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

# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 1000  # Number of times the model will see the full dataset

# Not doing any validation here... could add in the future
for epoch in range(num_epochs):
    if epoch == 500:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    optimizer.zero_grad()  # Reset gradients

    outputs = model(x).squeeze()  # Forward pass
    loss = criterion(outputs, y)  # Compute loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")

# # Now let's test the model
# x_data = []
# y_data = []
# for i in range(sensor_data_test.df.shape[0] - seq_size):
#     window = [[x] for x in sensor_data_test.df.iloc[i:i+seq_size]["T"].to_numpy()]
#     after_window = sensor_data_test.df.iloc[i + seq_size]["T"]
#     x_data.append(window)
#     y_data.append(after_window)
# x_test = torch.tensor(x_data, dtype=torch.float32)
# y_test = torch.tensor(y_data, dtype=torch.float32)


# # Make sure the model is in evaluation mode (disables dropout)
# model.eval()
# outputs = model(x_test)
# predictions = outputs.squeeze().tolist()

# rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
# print(f"Score (RMSE): {rmse:.4f}")

# Save the model to be tested separately
torch.save(model.state_dict(), 'saved_model_.pth')


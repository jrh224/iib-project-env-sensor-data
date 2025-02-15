import torch
import torch.nn as nn
import torch.optim as optim
from utils.CustomDataframe import CustomDataframe

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data_train = sensor_data.filter_by_date(start_date="10/09/2024 14:00", end_date="10/16/2024 14:00", inplace=False)
sensor_data_test = sensor_data.filter_by_date(start_date="10/16/2024 14:00", end_date="10/23/2024 14:00", inplace=False)

print(sensor_data.df)

sensor_data.plot(column="T")


# Dimensions of the training set ( x ):

# Axis 1: Training set elements (sequences) (must be of the same size as  y  size)
# Axis 2: Members of sequence
# Axis 3: Features in data (like input neurons)

x_data = [
    [[x] for x in sensor_data.df.iloc[0:20]["T"].to_numpy()], # not inclusive i.e. need to predict index 20
    [[x] for x in sensor_data.df.iloc[1:21]["T"].to_numpy()],
    [[x] for x in sensor_data.df.iloc[2:22]["T"].to_numpy()],
    [[x] for x in sensor_data.df.iloc[3:23]["T"].to_numpy()],
    [[x] for x in sensor_data.df.iloc[4:24]["T"].to_numpy()],
    [[x] for x in sensor_data.df.iloc[5:25]["T"].to_numpy()],
    [[x] for x in sensor_data.df.iloc[6:26]["T"].to_numpy()]
]
x = torch.tensor(x_data, dtype=torch.float32)

y_data = sensor_data.df.iloc[20:27]["T"].to_numpy()
y = torch.tensor(y_data, dtype=torch.float32)


# Model definition
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
model = LSTMModel()

# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 100  # Number of times the model will see the full dataset

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients

    outputs = model(x).squeeze()  # Forward pass
    loss = criterion(outputs, y)  # Compute loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")

# Now let's test the model
test_input_data = [
    [[x] for x in sensor_data.df.iloc[7:27]["T"].to_numpy()]
    ]
test_input = torch.tensor(test_input_data, dtype=torch.float32)

# Make sure the model is in evaluation mode (disables dropout)
model.eval()

# Get the prediction
with torch.no_grad():  # No gradients needed for inference
    predicted_value = model(test_input).item()  # Convert tensor to scalar

print(f"Predicted next temperature: {predicted_value:.2f}")
print("Actual next temperature: " + str(sensor_data.df.iloc[27]["T"]))


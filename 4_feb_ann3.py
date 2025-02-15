import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.CustomDataframe import CustomDataframe
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configuration
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data_train = sensor_data.filter_by_date(start_date="10/09/2024 14:00", end_date="10/16/2024 14:00", in_place=False)
sensor_data_test = sensor_data.filter_by_date(start_date="10/16/2024 14:00", end_date="10/23/2024 14:00", in_place=False)


# Load and prepare data
train_data = sensor_data_train.df["T"].values.astype(np.float32)

# Scale data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

# Create sequence datasets
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)


X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)

# Create DataLoader directly from tensors
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True) # Automatically deals with batching etc -> more efficient training

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size) # Apparently don't need a ReLU layer because this is a regression task
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.linear(last_time_step)
    
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    # # Validation
    # model.eval()
    # epoch_val_loss = 0
    # with torch.no_grad():
    #     for inputs, targets in val_loader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         epoch_val_loss += loss.item()
    
    # Calculate average losses
    avg_train_loss = epoch_train_loss / len(train_loader)
    # avg_val_loss = epoch_val_loss / len(val_loader)
    
    train_losses.append(avg_train_loss)
    # val_losses.append(avg_val_loss)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
          f'Train Loss: {avg_train_loss:.5f} | ')
    
    # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
    #       f'Train Loss: {avg_train_loss:.5f} | '
    #       f'Val Loss: {avg_val_loss:.5f}')

# Plot training history
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'model_4thfeb1.pth')


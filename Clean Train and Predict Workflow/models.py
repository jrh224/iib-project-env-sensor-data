import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size) # Apparently don't need a ReLU layer because this is a regression task
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.linear(last_time_step)
        
# Model based on paper "Building thermal dynamics modeling with deep transfer learning using a large residential smart thermostat dataset"
# See paper for hyperparameter values
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, input):
        output, _ = self.lstm(input) # no need for _ hidden state
        last_output = output[:, -1, :] # pass in final lstm value to linear layer
        output = self.linear(last_output)
        return output
    


# Encoder/decoder architecture to allow for the inclusion of future covariates
# Based on papers linked from https://stackoverflow.com/questions/70361179/how-to-include-future-values-in-a-time-series-prediction-of-a-rnn-in-keras
class Seq2SeqLSTMEncDec(nn.Module):
    def __init__(self, input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, output_dim=OUTPUT_SIZE):
        super(Seq2SeqLSTMEncDec, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_inputs, decoder_inputs):
        # Encode the input sequence (hidden and cell states of every element in the sequence) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        _, (hidden, cell) = self.encoder_lstm(encoder_inputs)

        # Decode using an LSTM
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs, (hidden, cell))

        # Fully connected layer for final output
        output = self.fc(decoder_outputs)
        return output
import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, HORIZON

# Define LSTM model
class LSTMModel(nn.Module): # Option 1
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size) # Apparently don't need a ReLU layer because this is a regression task
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] # Get output features (hidden state) of the end of the sequence of the LSTM 
        return self.linear(last_time_step).squeeze(-1)
        
# Model based on paper "Building thermal dynamics modeling with deep transfer learning using a large residential smart thermostat dataset"
# See paper for hyperparameter values
# Not suitable for using future covariates
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, horizon=HORIZON, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, horizon)

    def forward(self, input):
        output, _ = self.lstm(input) # no need for _ hidden state
        last_output = output[:, -1, :] # pass in final lstm value to linear layer
        output = self.linear(last_output)
        return output
    


# Encoder/decoder architecture to allow for the inclusion of future covariates
# Based on papers linked from https://stackoverflow.com/questions/70361179/how-to-include-future-values-in-a-time-series-prediction-of-a-rnn-in-keras
class Seq2SeqLSTMEncDec(nn.Module): # Option 2
    def __init__(self, input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, output_dim=1):
        super(Seq2SeqLSTMEncDec, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_dim-1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_inputs, decoder_inputs):
        # Encode the input sequence (hidden and cell states of every element in the sequence) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        _, (hidden, cell) = self.encoder_lstm(encoder_inputs)

        # Decode using an LSTM
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs, (hidden, cell))

        # Fully connected layer for final output
        output = self.fc(decoder_outputs)
        return output


# # Option 3 
# class HybridLSTMEncDec(nn.Module):
#     def __init__(self, input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, output_dim=1):
#         super(HybridLSTMEncDec, self).__init__()
#         self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        

#     def forward(self, input_tensor):
#         encoder_hidden = self.encoder_model.init_hidden(batch_size=input_tensor.shape[0])
#         _, encoder_hidden = self.encoder_model(input_tensor, encoder_hidden)

#         decoder_input = input_tensor[:, -1, -1].unsqueeze(1).unsqueeze(1)
#         decoder_outputs, decoder_hidden, _ = self.decoder_model(decoder_input, encoder_hidden)

#         return decoder_outputs

# # Used in Option 3
# class DecoderLSTM(nn.Module):
#     def __init__(self, hidden_dim, output_size, batch_size, n_layers, forecasting_horizon, bidirectional=False, dropout_p=0):
#         super(DecoderLSTM, self).__init__()
#         self.hidden_size = hidden_dim
#         self.output_size = output_size
#         self.batch_size = batch_size
#         self.bidirectional = bidirectional
#         self.dropout_p = dropout_p
#         self.forecasting_horizon = forecasting_horizon

#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
#         self.out = nn.Linear(hidden_dim, output_size)

#     def forward(self, decoder_input, encoder_hidden):
#         decoder_hidden = encoder_hidden
#         decoder_outputs = []

#         for i in range(self.forecasting_horizon):
#             decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
#             decoder_outputs.append(decoder_output)
#             decoder_input = decoder_hidden[0][-1, :, :].unsqueeze(0).permute(1, 0, 2)

#         decoder_outputs = torch.cat(decoder_outputs, dim=1)
#         return decoder_outputs, decoder_hidden, None

#     def forward_step(self, X, hidden):
#         output, hidden = self.lstm(X, hidden)
#         output = self.out(output)
#         return output, hidden
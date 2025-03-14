import numpy as np
import torch
import config

def create_sequences_with_future(data, lookback, predictforward, step=1, target_col=0, blocks=None):
    """
    Data should be of shape (# timestamps, # features). (e.g. train_matrix)

    If blocks is provided, it should be in the following format:
    blocks = [
    (0, 23),
    (30, 79),
    (100, 105)
    ]
    i.e. both sides 
    """

    sequences = []
    targets = []

    if blocks is None:
        for i in range(0, len(data) - lookback - predictforward, step):
            # Create a copy of the slice of the sequence to avoid modifying the original data
            sequence = data[i:i + lookback + predictforward].copy()
            # Mask the target values in the future prediction horizon (from i + lookback to i + lookback + predictforward)
            sequence[lookback:lookback + predictforward, target_col] = -9999
            # Append the sequence (with the masked target values) to sequences
            sequences.append(sequence)

            targets.append(data[i + lookback:i + lookback + predictforward, target_col])

        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).squeeze(-1)
    
    else:
        for block in blocks:
            for i in range(block[0], block[1]+1 - lookback - predictforward, step):
                # Create a copy of the slice of the sequence to avoid modifying the original data
                sequence = data[i:i + lookback + predictforward].copy()
                # Mask the target values in the future prediction horizon (from i + lookback to i + lookback + predictforward)
                sequence[lookback:lookback + predictforward, target_col] = -9999
                # Append the sequence (with the masked target values) to sequences
                sequences.append(sequence)

                targets.append(data[i + lookback:i + lookback + predictforward, target_col])

        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).squeeze(-1)

def create_sequences(data, lookback, horizon, stride=1, target_col=0, blocks=None):
    """
    Returns:
        - sequences (# sequences, # lookback, # features)
        - targets (# sequences)

    Data should be of shape (# timestamps, # features). (e.g. train_matrix)

    If blocks is provided, it should be in the following format:
    blocks = [
    (0, 23),
    (30, 79),
    (100, 105)
    ]
    i.e. both sides 
    """

    sequences = []
    targets = []

    if blocks is None:
        for i in range(0, len(data) - lookback - horizon, stride):
            # Create a copy of the slice of the sequence to avoid modifying the original data
            sequence = data[i:i + lookback]
            # Append the sequence
            sequences.append(sequence)

            targets.append(data[i + lookback:i + lookback + horizon, target_col])

        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).squeeze(-1)
    
    else:
        for block in blocks:
            for i in range(block[0], block[1]+1 - lookback - horizon, stride):
                # Create a copy of the slice of the sequence to avoid modifying the original data
                sequence = data[i:i + lookback]
                # Append the sequence
                sequences.append(sequence)

                targets.append(data[i + lookback:i + lookback + horizon, target_col])

        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).squeeze(-1)
    
def get_encdec_inputs(matrix, lookback, horizon, stride=1, target_col=0, blocks=None):
    """
    Returns torch.tensor objects
        - encoder_ins (# samples, # lookback, # features incl. target)
        - decoder_ins (# samples, # lookback, # features excl. target)
        - targets (# samples, # horizon, 1)

    Matrix should be of shape (# timestamps, # features). (e.g. train_matrix)

    If blocks is provided, it should be in the following format:
    blocks = [
    (0, 23),
    (30, 79),
    (100, 105)
    ]
    i.e. both sides 
    """

    encoder_ins = []
    decoder_ins = []
    targets = []


    if blocks is None:
        for i in range(0, matrix.shape[0] - lookback - horizon, stride):
            tlookback = i
            t0 = i+lookback
            thorizon = i+lookback+horizon

            encoder_in = matrix[tlookback:t0, :] # All features, before t=0
            encoder_ins.append(encoder_in)

            decoder_in = matrix[t0:thorizon, :] # All features, after t=0
            decoder_in = np.delete(decoder_in, target_col, axis=1) # Remove target column
            decoder_ins.append(decoder_in)

            target = matrix[t0:thorizon, target_col] # Target column, after t=0
            targets.append(target)
    else:
        for block in blocks:
            for i in range(block[0], block[1]+1 - lookback - horizon, stride):
                tlookback = i
                t0 = i+lookback
                thorizon = i+lookback+horizon

                encoder_in = matrix[tlookback:t0, :] # All features, before t=0
                encoder_ins.append(encoder_in)

                decoder_in = matrix[t0:thorizon, :] # All features, after t=0
                decoder_in = np.delete(decoder_in, target_col, axis=1) # Remove target column
                decoder_ins.append(decoder_in)

                target = matrix[t0:thorizon, target_col] # Target column, after t=0
                targets.append(target)

    return torch.tensor(np.array(encoder_ins), dtype=torch.float32), torch.tensor(np.array(decoder_ins), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)


def my_windowed_dataset(matrix, lookback, horizon, stride, target_col=0, blocks=None):
    """
    UNUSED
    Matrix should be of shape (# timestamps, # features). (e.g. train_matrix)

    If blocks is provided, it should be in the following format:
    blocks = [
    (0, 23),
    (30, 79),
    (100, 105)
    ]
    i.e. both sides 

    Returns torch.tensor objects (encoder_ins, decoder_ins, targets)
    """

    encoder_ins = []
    decoder_ins = []
    targets = []


    if blocks is None:
        pass
        # for i in range(0, matrix.shape[0] - lookback - horizon, stride):
            # tlookback = i
            # t0 = i+lookback
            # thorizon = i+lookback+horizon

            # encoder_in = matrix[tlookback:t0, :] # All features, before t=0
            # encoder_ins.append(encoder_in)

            # decoder_in = matrix[t0:thorizon, :] # All features, after t=0
            # decoder_in = np.delete(decoder_in, target_col, axis=1) # Remove target column
            # decoder_ins.append(decoder_in)

            # target = matrix[t0:thorizon, target_col] # Target column, after t=0
            # targets.append(target)
    else:
        for block in blocks:
            for i in range(block[0], block[1]+1 - lookback - horizon, stride):
                tlookback = i
                t0 = i+lookback
                thorizon = i+lookback+horizon
                
                input = matrix[tlookback:t0, :] # All features, before t=0
                encoder_ins.append(encoder_in)

                decoder_in = matrix[t0:thorizon, :] # All features, after t=0
                decoder_in = np.delete(decoder_in, target_col, axis=1) # Remove target column
                decoder_ins.append(decoder_in)

                target = matrix[t0:thorizon, target_col] # Target column, after t=0
                targets.append(target)

    return torch.tensor(np.array(encoder_ins), dtype=torch.float32), torch.tensor(np.array(decoder_ins), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)




def autoregressive_predict(model, test_matrix, num_predictions, start_point=0):
    """
    Make autoregressive predictions using a trained model
    Returns:
        - List: predictions (# num_predictions)
    """
    model.eval()
    predictions = []

    current_lookback = test_matrix[start_point:config.LOOKBACK+start_point, :]
    
    for i in range(num_predictions):
        # Convert to tensor and add batch dimension (so that the input fits the size requirement)
        input_tensor = torch.tensor(current_lookback, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predicted = model(input_tensor)
            
        # Store prediction
        predictions.append(predicted.item())
        
        # Update sequence: assumes that column 0 is the target column
        new_entry = test_matrix[start_point + config.LOOKBACK + i, :].copy()  # Copy the next row
        new_entry[0] = predicted.item()  # Replace the first column with the prediction

        
        # Update sequence: remove oldest, add new prediction
        current_lookback = np.vstack([
            current_lookback[1:],  # Remove oldest entry
            new_entry       # Add new prediction
        ])
        
    return predictions


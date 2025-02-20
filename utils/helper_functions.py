import numpy as np
import torch
import config

def create_sequences(data, lookback, predictforward, target_col=0):
    """
    Data should be of shape (# timestamps, # features)
    """
    sequences = []
    targets = []
    # for i in range(len(data) - lookback):
    #     sequences.append(data[i:i+lookback])
    #     targets.append(data[i+lookback, target_col]) # only add the internal temperature to the y vector
    
    for i in range(len(data) - lookback - predictforward):
        sequences.append(data[i:i+lookback])
        targets.append(data[i+lookback:i+lookback+predictforward, target_col])

    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32).squeeze(-1)



def autoregressive_predict(model, test_data, num_predictions, start_point=0):
    """Make autoregressive predictions using a trained model"""
    model.eval()
    predictions = []

    current_lookback = test_data[start_point:config.LOOKBACK+start_point]
    
    for i in range(num_predictions):
        # Convert to tensor and add batch dimension (so that the input fits the size requirement)
        input_tensor = torch.tensor(current_lookback, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predicted = model(input_tensor)
            
        # Store prediction
        predictions.append(predicted.item())
        
        # Update sequence: assumes that column 0 is the target column
        new_entry = test_data[start_point + config.LOOKBACK + i].copy()  # Copy the next row
        new_entry[0] = predicted.item()  # Replace the first column with the prediction

        
        # Update sequence: remove oldest, add new prediction
        current_lookback = np.vstack([
            current_lookback[1:],  # Remove oldest entry
            new_entry       # Add new prediction
        ])
        
    return predictions


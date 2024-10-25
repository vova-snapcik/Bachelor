import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
import xarray as xr
import torchmetrics
import math
from scipy.stats import pearsonr
import logging  # Added for logging


# Input directories and preferred save locations
data_input_dir = "/zhome/cf/0/188047/Bachelor/Data/Transformer 3.0/"  # Absolute path to the location of training datasets
model_output_dir = "/zhome/cf/0/188047/Bachelor/Model Evaluation/Transformer 3.0" #Absolute path to the preferred location to save the model
graphic_output_dir = model_output_dir  # Absolute path to visualisation directory, set to model location for now


# Set up logging to a file
log_file = os.path.join(model_output_dir, 'training_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')


# Define functions to load data efficiently and convert to tensor
def load_sensor_data(file_path, sensor_names):
    """
    Loads sensor data from the .nc file and rearranges it so that time steps are the first dimension.
    This makes it compatible with sequence models that expect input in the format [seq_length, num_sensors].

    Parameters:
        file_path (str): Path to the .nc file.
        sensor_names (list of str): List of sensor variable names to load from the file.

    Returns:
        torch.Tensor: Sensor data tensor in the shape [time_steps, num_sensors].
        torch.Tensor: Time steps tensor in the shape [time_steps].
    """

    # Use memory-efficient loading from .nc file
    ds_sensors = xr.open_dataset(file_path, chunks={'time': 1000})  # Use chunking for large datasets

    # Extract and stack sensor data into a single tensor, ensuring time steps come first
    sensor_data_list = []
    for sensor_name in sensor_names:
        if sensor_name in ds_sensors.variables:
            sensor_data = torch.as_tensor(ds_sensors[sensor_name].values, dtype=torch.float32)
            sensor_data_list.append(sensor_data)
        else:
            raise ValueError(f"Sensor {sensor_name} not found in dataset.")

    # Stack along the new dimension (which will be the second dimension after stacking) and then transpose
    # This ensures that the output tensor is [time_steps, num_sensors]
    sensor_data_tensor = torch.stack(sensor_data_list, dim=-1)  # Initially [time_steps, num_sensors]

    # Return the sensor data tensor and time steps tensor
    return sensor_data_tensor, torch.as_tensor(ds_sensors['t'].values, dtype=torch.float32)

def load_modes_data(file_path):
    """
    Loads modal (inflow) data from the .nc file and rearranges it so that time steps are the first dimension,
    ensuring consistency with the expected input format for sequence models.

    Parameters:
        file_path (str): Path to the .nc file.

    Returns:
        torch.Tensor: Modal data tensor in the shape [time_steps, num_modes].
    """

    # Load the dataset using chunking for efficient memory usage
    ds_modes = xr.open_dataset(file_path, chunks={'t': 1000})  # Chunking along the 't' dimension

    # Extract the modal data (assumed to be under the 'Cmt' variable) and convert to a PyTorch tensor
    modes_data = torch.as_tensor(ds_modes['Cmt'].values,
                                 dtype=torch.float32)  # Original shape might be [num_modes, time_steps]

    # Ensure that the output shape is [time_steps, num_modes] (transpose if necessary)
    if modes_data.shape[0] != ds_modes.dims['t']:
        modes_data = modes_data.T  # Transpose to ensure the shape is [time_steps, num_modes]

    return modes_data


# Define function for creating sequences for splitting time-series data into input-output pairs in a seq2seq fashion
def create_sequences_for_seq2seq(sensor_data: torch.Tensor, modes_data: torch.Tensor, seq_length: int):
    """
    Creates sequences for the Transformer model for sequence-to-sequence tasks like inflow translation (akin
    to sequence translation tasks).

    Parameters:
        sensor_data (torch.Tensor): The input sensor data with shape [time_steps, num_sensors].
        modes_data (torch.Tensor): The modal (inflow) data with shape [time_steps, num_modes].
        seq_length (int): Length of the input sequences (Tin), which also applies to the target sequences.

    Returns:
        Tuple of torch.Tensor: (encoder_input, target)
            - encoder_input: The sequences for the sensor data.
            - target: The sequences for the modal data (translation target).
    """

    # Calculate the total number of sequences possible given the seq_length
    total_sequences = sensor_data.shape[0] - seq_length + 1
    assert total_sequences > 0, "Sequence length is too large for the available data."

    # Initialize lists for encoder inputs and target outputs
    encoder_inputs = []
    targets = []

    # Create sequences by sliding over the data
    for i in range(total_sequences):
        # Encoder input: slice from sensor data (input sequence to translate)
        encoder_input = sensor_data[i:i + seq_length, :]  # Shape: [seq_length, num_sensors]

        # Target: slice from modal data (corresponding sequence in modal data)
        target = modes_data[i:i + seq_length, :]  # Shape: [seq_length, num_modes]

        # Append to lists
        encoder_inputs.append(encoder_input)
        targets.append(target)

        # Optional: Log progress for long-running jobs on remote servers
        if (i + 1) % 1000 == 0 or (i + 1) == total_sequences:
            print(f"Created sequence {i + 1}/{total_sequences}")

    # Convert lists of tensors to batched tensors
    encoder_inputs = torch.stack(encoder_inputs)  # Shape: [num_sequences, seq_length, num_sensors]
    targets = torch.stack(targets)  # Shape: [num_sequences, seq_length, num_modes]

    return encoder_inputs, targets


# Create custom dataset for multi-sequence input (encoder) and target sequence
class WindTurbineDataset(Dataset):
    def __init__(self, encoder_inputs, targets, transform=None, target_transform=None):
        """
        Initializes the dataset with encoder inputs and target sequences.

        Parameters:
            encoder_inputs (torch.Tensor): The input sequences for the encoder,
                                           shape [num_samples, sequence_length, num_sensors].
            targets (torch.Tensor): The target sequences, shape [num_samples, sequence_length, num_modes].
            transform (callable, optional): Optional transform to apply to the encoder inputs.
            target_transform (callable, optional): Optional transform to apply to the targets.
        """
        if not isinstance(encoder_inputs, torch.Tensor):
            raise TypeError("encoder_inputs must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("targets must be a torch.Tensor")

        self.encoder_inputs = encoder_inputs
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - Encoder input sequence for the given index
                - Target sequence for the given index
        """
        encoder_input = self.encoder_inputs[idx]
        target = self.targets[idx]

        # Apply transformations if available
        if self.transform:
            encoder_input = self.transform(encoder_input)
        if self.target_transform:
            target = self.target_transform(target)

        return encoder_input, target


# Create positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initializes the positional encoding module.

        Parameters:
            d_model (int): The embedding size (dimensionality) of the model.
            max_len (int): The maximum possible length for the positional encoding.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Store pe without making it a learnable parameter

    def forward(self, x):
        """
        Adds positional encoding to the input tensor x.

        Parameters:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


# Define Transformer model for inflow reconstruction
class TransformerInflowReconstruction(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerInflowReconstruction, self).__init__()
        self.model_dim = model_dim

        # Positional encoding for both encoder and decoder
        self.positional_encoding = PositionalEncoding(d_model=model_dim)

        # Input embedding layers
        self.encoder_input_proj = nn.Linear(input_dim, model_dim)
        self.decoder_input_proj = nn.Linear(target_dim, model_dim)

        # Transformer encoder and decoder
        self.transformer = nn.Transformer(d_model=model_dim, nhead=nhead,
                                          num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dropout=dropout, batch_first=True)

        # Final projection to predict the target sequence (modes data)
        self.output_proj = nn.Linear(model_dim, target_dim)

    def forward(self, encoder_input, decoder_target=None):
        """
        Forward pass for all phases: training, validation, and testing.

        Parameters:
            encoder_input (tensor): Input for the encoder, shape [batch_size, sequence_length, num_sensors].
            decoder_target (tensor or None): Target for the decoder, shape [batch_size, target_length, num_modes],
                                             or None for testing.

        Returns:
            Tensor of predicted target sequence, shape [batch_size, target_length, num_modes].
        """
        batch_size, seq_length = encoder_input.size(0), encoder_input.size(1)

        # Project encoder inputs (sensor data)
        encoder_input = self.encoder_input_proj(encoder_input)  # [batch_size, seq_length, model_dim]
        encoder_input = self.positional_encoding(encoder_input)

        # Encode the input sequence
        encoder_output = self.transformer.encoder(encoder_input)  # [batch_size, seq_length, model_dim]

        if decoder_target is not None:
            # Training or Validation: Use actual target with positional encoding
            decoder_input = self.decoder_input_proj(decoder_target)  # [batch_size, target_length, model_dim]
            decoder_input = self.positional_encoding(decoder_input)
        else:
            # Testing: Initialize decoder input with zeros for the entire sequence length
            decoder_input = torch.zeros((batch_size, seq_length, self.model_dim), device=encoder_input.device)
            decoder_input = self.positional_encoding(decoder_input)

        # Decode the sequence using the full decoder input
        decoder_output = self.transformer.decoder(decoder_input, encoder_output)
        output = self.output_proj(decoder_output)  # [batch_size, target_length, num_modes]

        return output


# Sensor names can be expanded in the future
sensor_names = ['Vhub', 'Power', 'Omega']  # Add more sensor names as needed


# Load sensor and modal data from their respective files for training
print("Loading data from .nc files...")
training_sensor_file = os.path.join(data_input_dir, 'WT01U8_sensor.nc')  # Training sensor data
if not os.path.exists(training_sensor_file):
    print(f"Data file not found at {training_sensor_file}")
    exit()
training_modes_file = os.path.join(data_input_dir, 'WT01U8_modes.nc')  # Training modes data
if not os.path.exists(training_modes_file):
    print(f"Data file not found at {training_modes_file}")
    exit()

# Load sensor and modal data from their respective files for validation
validation_sensor_file = os.path.join(data_input_dir, 'WT01U12_sensor.nc')  # Validation sensor data
if not os.path.exists(validation_sensor_file):
    print(f"Data file not found at {validation_sensor_file}")
    exit()
validation_modes_file = os.path.join(data_input_dir, 'WT01U12_modes.nc')  # Validation modes data
if not os.path.exists(validation_modes_file):
    print(f"Data file not found at {validation_modes_file}")
    exit()

# Load sensor (Vhub, Power, Omega, etc.) and modes (Cmt) for training (WT01U8)
training_sensor_data, training_time_steps = load_sensor_data(training_sensor_file, sensor_names)
training_modes_data = load_modes_data(training_modes_file)

# Load sensor (Vhub, Power, Omega, etc.) and modes (Cmt) for validation (WT01U12)
validation_sensor_data, validation_time_steps = load_sensor_data(validation_sensor_file, sensor_names)
validation_modes_data = load_modes_data(validation_modes_file)
print("Data files loaded successfully.")


# Logging tensor shapes for debugging
logging.info(f"Training sensor data tensor shape (stacked): {training_sensor_data.shape}")  # [time_steps, num_sensors]
logging.info(f"Training modes data tensor shape (Cmt): {training_modes_data.shape}")  # [time_steps, num_modes] = [1000, 5]
logging.info(f"Time steps tensor shape: {training_time_steps.shape}")

logging.info(f"Validation sensor data tensor shape (stacked): {validation_sensor_data.shape}")  # [time_steps, num_sensors]
logging.info(f"Validation modes data tensor shape (Cmt): {validation_modes_data.shape}")  # [time_steps, num_modes] = [1000, 5]
logging.info(f"Time steps tensor shape: {validation_time_steps.shape}")


# Create sequences for the model using training data
print("Creating sequences from data...")
encoder_X_train, y_train = create_sequences_for_seq2seq(training_sensor_data, training_modes_data, sequence_length)
# Create sequences for the model using validation data
encoder_X_val, y_val = create_sequences_for_seq2seq(validation_sensor_data, validation_modes_data, sequence_length)
print("Sequences created.")

# Output the shapes
print(f"Encoder input shape: training {encoder_X_train.shape}, validation {encoder_X_val.shape}")  # Tensor of shape [num_sequences, seq_length, num_sensors]
print(f"Target shape: training {y_train.shape}, validation {y_val.shape}")  # Tensor of shape [num_sequences, seq_length, num_modes]
logging.info(f"Encoder input shape: training {encoder_X_train.shape}, validation {encoder_X_val.shape}")  # Tensor of shape [num_sequences, seq_length, num_sensors]
logging.info(f"Target shape: training {y_train.shape}, validation {y_val.shape}")  # Tensor of shape [num_sequences, seq_length, num_modes]


# Create training dataset and DataLoader
train_dataset = WindTurbineDataset(encoder_X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Create validation dataset and DataLoader (no shuffling for validation)
val_dataset = WindTurbineDataset(encoder_X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)














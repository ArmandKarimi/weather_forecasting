import sys
import os
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader

# Get the absolute path of the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import SEQ_LENGTH, PRED_LENGTH, BATCH_SIZE

# --------------- creating new features ------------ #
def feature_eng(df):
    """
    Feature Engineering for Weather Forecasting Data:
    
    1. Subsamples the data to hourly intervals (original data is every 10 mins).
    2. Converts 'Date Time' from string type to datetime.
    3. Converts recurrent features (Wind and Date) to Cartesian coordinates.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing weather data.
    
    Returns:
        pd.DataFrame: Processed DataFrame with engineered features.
    """

    # Ensure we are working with a copy to avoid modifying original data
    df = df.copy()

    # 1. Sub-sampling the data from 10-minute intervals to one-hour intervals
    df = df.iloc[5::6].reset_index(drop=True)

    # 2. Convert 'Date Time' column from string to datetime
    if 'Date Time' in df.columns:
        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    else:
        raise ValueError("Column 'Date Time' not found in DataFrame")

    # 3.1 Convert Wind features to Cartesian coordinates
    required_columns = {'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract wind data
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert wind direction to radians
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Compute Cartesian wind components using `.loc[]` to avoid warnings
    df.loc[:, 'Wx'] = wv * np.cos(wd_rad)
    df.loc[:, 'Wy'] = wv * np.sin(wd_rad)
    df.loc[:, 'max Wx'] = max_wv * np.cos(wd_rad)
    df.loc[:, 'max Wy'] = max_wv * np.sin(wd_rad)

    # 3.2 Convert Time features to Cartesian coordinates
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = 365.2425 * day  # Accounting for leap years

    df.loc[:, 'Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df.loc[:, 'Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df.loc[:, 'Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df.loc[:, 'Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


#____SPLIT data chornolgically_______
def chronological_split(data, ratios=(0.7, 0.2, 0.1), buffer=7):
    # Split ratios
    
    """
    Splits the DataFrame into training, validation, and test sets in chronological order with a buffer.
    """
    n = len(data)
    train_end = int(n * ratios[0]) - buffer
    val_end = train_end + int(n * ratios[1]) - buffer

    df_train = data.iloc[:train_end]
    df_val = data.iloc[train_end+buffer:val_end]
    df_test = data.iloc[val_end+buffer:]

    return df_train, df_val, df_test

# ------------ Normalize data ---------------
def normalization(df_train, df_val, df_test):
    train_mean = df_train.mean()
    train_std = df_train.std()

    df_train = (df_train - train_mean) / train_std
    df_val = (df_val - train_mean) / train_std
    df_test = (df_test - train_mean) / train_std

    return df_train, df_val, df_test

# ---------- Create Sequences ---------------
def create_sequences(df, seq_length = SEQ_LENGTH, pred_length = PRED_LENGTH):
    """
    Creates sequences and corresponding targets from a DataFrame.
    
    Parameters:
        X (pd.DataFrame): The input data containing features. It must include a column 'T (degC)' 
                          for the target variable.
        seq_length (int): The length of the input sequence.
        pred_length (int): The number of future time steps to predict.
        
    Returns:
        tuple: (X_seq, y_seq) where:
            - X_seq is a torch.Tensor of shape (num_sequences, seq_length, num_features)
            - y_seq is a torch.Tensor of shape (num_sequences, pred_length)
    """
    sequences = []
    targets = []
    
    # Ensure there are enough rows to create at least one sequence
    for i in range(len(df) - seq_length - pred_length + 1):
        # Get the sequence of features; this takes all columns for rows i to i+seq_length
        sequences.append(df.iloc[i:i+seq_length].values)
        # Get the target sequence from the 'Close' column only
        targets.append(df['T (degC)'].iloc[i+seq_length:i+seq_length+pred_length].values)
    
    X_seq = np.array(sequences, dtype=np.float32)
    X_seq = torch.tensor(X_seq)

    y_seq = np.array(targets, dtype=np.float32)
    y_seq = torch.tensor(y_seq)

    return X_seq, y_seq

#----create data loaders------
def data_loader(X_seq, y_seq, batch_size=BATCH_SIZE):
    """
    Creates a DataLoader for the given sequences and targets.
    Parameters:
        X_seq (torch.Tensor): Tensor containing input sequences.
        y_seq (torch.Tensor): Tensor containing corresponding target sequences.
        batch_size (int): Batch size for the DataLoader.
        
    Returns:
        DataLoader: A DataLoader wrapping a TensorDataset of (X_seq, y_seq).
    """
    dataset = TensorDataset(X_seq, y_seq)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    # Get the absolute path of the project root directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import SEQ_LENGTH, PRED_LENGTH, BATCH_SIZE
    # --- Configure logging ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    df = pd.read_csv("../data/jena_climate_2009_2016.csv")
    logger.info(f"✅ data loaded => data-shape : {df.shape}")

    df = feature_eng(df)
    logger.info(f"✅ new features created => new data-shape {df.shape}")

    # --- Data Processing ---
    df_train, df_val, df_test = chronological_split(df)
    logger.info(f"✅ Data split into train, validation, and test sets created. shapes :{df_train.shape}, {df_val.shape}, {df_test.shape}")

    # --- Data Normalization ---
    df_train, df_val, df_test = normalization(df_train, df_val, df_test)
    logger.info(f"✅ Data normalized. df_train.mean() :{df_train.mean()}")

    # --- Sequence Creation ---
    X_train_seq, y_train_seq = create_sequences(df_train, SEQ_LENGTH, PRED_LENGTH)
    X_val_seq, y_val_seq     = create_sequences(df_val, SEQ_LENGTH, PRED_LENGTH)
    X_test_seq, y_test_seq   = create_sequences(df_test, SEQ_LENGTH, PRED_LENGTH)
    logger.info(f"✅ Sequences created: Training: {X_train_seq.shape}, Validation: {X_val_seq.shape}, Test: {X_test_seq.shape}")

    # Create DataLoaders
    train_loader = data_loader(X_train_seq, y_train_seq, BATCH_SIZE)
    val_loader   = data_loader(X_val_seq, y_val_seq, BATCH_SIZE)
    test_loader  = data_loader(X_test_seq, y_test_seq, BATCH_SIZE)
    logger.info(f"✅ DataLoaders created")






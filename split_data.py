import time
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

@time_tracker
def split_data(numerical_data_filtered, labels_encoded, split_sizes, random_state=42):
    splits = {}
    for size in split_sizes:
        if size <= len(numerical_data_filtered):
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=random_state)
            for train_idx, _ in splitter.split(numerical_data_filtered, labels_encoded):
                splits[size] = numerical_data_filtered.iloc[train_idx], [labels_encoded[i] for i in train_idx]
    return splits

@time_tracker
def train_val_test_split(data, labels, random_state=42):
    # First split: Train (70%) and Remaining (30%)
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        data, labels, test_size=0.3, random_state=random_state, stratify=labels
    )
    # Second split: Validation (60%) and Test (40%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.4, random_state=random_state, stratify=y_remaining
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

@time_tracker
def scale_and_save_data(train_val_test_splits, output_path, random_state=42):
    scaler = StandardScaler()
    for key, (X_train, X_val, X_test, y_train, y_val, y_test) in train_val_test_splits.items():
        # Fit the scaler only on the training data
        X_train_scaled = scaler.fit_transform(X_train)
        # Apply the same scaler to the validation and test sets
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create Datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create directory for saving datasets
        dataset_path = f'{output_path}/{key}/'
        directory_path = Path(dataset_path)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save datasets as tensors
        torch.save(train_dataset.tensors, f'{dataset_path}train_dataset_tensors.pt')  # Saves (X_train, y_train)
        torch.save(val_dataset.tensors, f'{dataset_path}val_dataset_tensors.pt')      # Saves (X_val, y_val)
        torch.save(test_dataset.tensors, f'{dataset_path}test_dataset_tensors.pt')    # Saves (X_test, y_test)

@time_tracker
def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16, random_state=42):
    scaler = StandardScaler()
    train_dataset = TensorDataset(
        torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(scaler.transform(X_val), dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(scaler.transform(X_test), dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(random_state))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(random_state))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(random_state))

    return train_loader, val_loader, test_loader




import pandas as pd
import torch
import random
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables fast auto-tuning

set_seed(42)

# Check if numerical data and metadata CSVs already exist
numerical_data_path = '/data/v2/numerical_data_filtered.csv'
metadata_path = '/data/v2/metadata_with_labels.csv'

if os.path.exists(numerical_data_path) and os.path.exists(metadata_path):
    # Load existing CSV files
    numerical_data_filtered = pd.read_csv(numerical_data_path)
    df_metadata = pd.read_csv(metadata_path)
else:
    # Load dataframes from pickle files and concatenate them
    data_files = [f'data/v2_HM450/methyl_scores_v2_HM450k_{i}.pkl' for i in range(1, 12)]
    dataframes = [pd.read_pickle(file, compression="bz2") for file in data_files]
    df = pd.concat(dataframes, axis=0)

    # Metadata and label columns
    metadata_columns = [
        'id', 'geo_accession', 'title', 'sex', 'age', 'race', 'tissue',
        'geo_platform', 'inferred_age_Hannum', 'inferred_age_SkinBlood',
        'inferred_age_Horvath353'
    ]
    label_column = 'disease'
    sex_condition_column = 'inferred_sex'
    age_condition_column = 'inferred_age_MepiClock'

    # Prepare numerical data by dropping metadata and condition columns
    numerical_data = df.drop(
        metadata_columns + [label_column, sex_condition_column, age_condition_column],
        axis=1
    )

    # Fill missing labels with default value
    df[label_column].fillna('no_label', inplace=True)

    # Extract labels and conditions
    labels = df[label_column]
    sex_conditions = df[sex_condition_column]
    age_conditions = df[age_condition_column]

    # Calculate the percentage of NaN values in each column
    nan_percentage = numerical_data.isna().sum() / numerical_data.shape[0] * 100

    # Filter columns with less than 10% NaN values
    selected_columns = nan_percentage[nan_percentage < 10].index.tolist()
    numerical_data_filtered = numerical_data[selected_columns]

    # One-hot encode categorical labels
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = onehot_encoder.fit_transform(labels.values.reshape(-1, 1))

    # Label encode categorical labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels.values)

    # Add encoded labels as a new column to the DataFrame
    df['labels_encoded'] = labels_encoded

    # Keep the index as a column and reindex the DataFrame
    df.reset_index(inplace=True)

    # Save numerical data filtered and metadata with labels, sex condition, and age condition to CSV
    numerical_data_filtered.to_csv(numerical_data_path, index=False)
    metadata_columns_with_labels = metadata_columns + [label_column, sex_condition_column, age_condition_column, 'labels_encoded']
    df_metadata = df[metadata_columns_with_labels]
    df_metadata.to_csv(metadata_path, index=False)




# Main execution
split_sizes = [len(numerical_data_filtered), 20000, 10000, 5000]
splits = split_data(numerical_data_filtered, labels_encoded, split_sizes)
train_val_test_splits = {
    key: train_val_test_split(data, labels) for key, (data, labels) in splits.items()
}

scale_and_save_data(train_val_test_splits, output_path='data/v2/m1')

# Create DataLoaders (for the largest split as an example)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_splits[len(numerical_data_filtered)]
train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test)

# Check the stratified split
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

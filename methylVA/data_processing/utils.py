import numpy as np
import pandas as pd
from methylVA.utils.decorators import time_tracker
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pickle




@time_tracker
def sample_data(numerical_data_filtered, labels_encoded, split_size, random_state=42):
    if split_size == 'shuffled_10000':
        return generate_shuffled_data(numerical_data_filtered, 10000, random_state)
    else:
        size = int(split_size)
        if size < len(numerical_data_filtered):
            print(f"Splitting data with size: {size}")
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=random_state)
            for train_idx, _ in splitter.split(numerical_data_filtered, labels_encoded):
                return numerical_data_filtered.iloc[train_idx], [labels_encoded[i] for i in train_idx]
        else:
            return numerical_data_filtered, labels_encoded

@time_tracker
def generate_shuffled_data(numerical_data_filtered, size, random_state=42):
    print("Generating shuffled data for null hypothesis...")
    flattened_values = numerical_data_filtered.values.flatten()
    np.random.seed(random_state)
    np.random.shuffle(flattened_values)
    shuffled_matrix = flattened_values.reshape(numerical_data_filtered.shape)
    shuffled_data = pd.DataFrame(shuffled_matrix, columns=numerical_data_filtered.columns).sample(n=size, random_state=random_state).reset_index(drop=True)
    shuffled_labels = [0] * size  # Use dummy labels to represent null hypothesis
    return shuffled_data, shuffled_labels


def load_data_tensor(train_config):
    # Load the training, validation, and test datasets from the specified paths
    input_dir = train_config['input_dir']
    batch_size = train_config['batch_size']
    input_dir = "../" + input_dir
    
    # Loading data splits
    split_dirs = ['train', 'val', 'test']
    loaded_data = {}

    # Iterate over the splits to read the data
    for split in split_dirs:
        split_path = Path(input_dir).joinpath(split)

        # Load features
        with open(split_path.joinpath('features.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        # Load labels
        with open(split_path.joinpath('labels.pkl'), 'rb') as f:
            label = pickle.load(f)

        # Store loaded data in a dictionary
        loaded_data[split] = (data, label)

    # Now you have loaded_data dictionary containing (features, labels) for 'train', 'val', 'test'
    X_train, y_train = loaded_data['train']
    X_val, y_val = loaded_data['val']
    X_test, y_test = loaded_data['test']

    del loaded_data  # Free up memory

    # Convert features and labels to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)  # Assuming data is in a DataFrame
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) if hasattr(y_train, 'values') else torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long) if hasattr(y_val, 'values') else torch.tensor(y_val, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) if hasattr(y_test, 'values') else torch.tensor(y_test, dtype=torch.long)

    # Create TensorDataset for each split
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Creating DataLoaders
    # todo: add seed for reproducibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))

    return train_loader, val_loader, test_loader


def load_train_test_data(train_config):
    # Load the training, validation, and test datasets from the specified paths
    input_dir = train_config['input_dir']
    batch_size = train_config['batch_size']
    input_dir = "../" + input_dir
    
    # Loading data splits
    split_dirs = ['train', 'val', 'test']
    loaded_data = {}

    # Iterate over the splits to read the data
    for split in split_dirs:
        split_path = Path(input_dir).joinpath(split)

        # Load features
        with open(split_path.joinpath('features.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        # Load labels
        with open(split_path.joinpath('labels.pkl'), 'rb') as f:
            label = pickle.load(f)

        # Store loaded data in a dictionary
        loaded_data[split] = (data, label)

    # Now you have loaded_data dictionary containing (features, labels) for 'train', 'val', 'test'
    X_train, y_train = loaded_data['train']
    X_val, y_val = loaded_data['val']
    X_test, y_test = loaded_data['test']

    return X_train, y_train, X_val, y_val, X_test, y_test
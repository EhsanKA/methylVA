import os
import time
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

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

# Time tracking decorator
def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

# Load or prepare data
numerical_data_path = 'data/v2/numerical_data_filtered.csv'
metadata_path = 'data/v2/metadata_with_labels.csv'

print("Checking if preprocessed data exists...")
if os.path.exists(numerical_data_path) and os.path.exists(metadata_path):
    print("Loading preprocessed data from CSV files...")
    numerical_data_filtered = pd.read_csv(numerical_data_path)
    df_metadata = pd.read_csv(metadata_path)
else:
    print("Preprocessed data not found. Loading raw data and processing...")
    data_files = [f'data/v2_HM450/methyl_scores_v2_HM450k_{i}.pkl' for i in range(1, 12)]
    dataframes = [pd.read_pickle(file, compression="bz2") for file in data_files]
    df = pd.concat(dataframes, axis=0)

    metadata_columns = [
        'id', 'geo_accession', 'title', 'sex', 'age', 'race', 'tissue',
        'geo_platform', 'inferred_age_Hannum', 'inferred_age_SkinBlood',
        'inferred_age_Horvath353'
    ]
    label_column = 'disease'
    sex_condition_column = 'inferred_sex'
    age_condition_column = 'inferred_age_MepiClock'

    numerical_data = df.drop(
        metadata_columns + [label_column, sex_condition_column, age_condition_column],
        axis=1
    )

    # Fix FutureWarning
    df[label_column] = df[label_column].fillna('no_label')

    # Fix PerformanceWarning
    labels_encoded = df[label_column].astype('category').cat.codes
    df = pd.concat([df, labels_encoded.rename('labels_encoded')], axis=1)
    df = df.reset_index()

    nan_percentage = numerical_data.isna().sum() / numerical_data.shape[0] * 100
    selected_columns = nan_percentage[nan_percentage < 10].index.tolist()
    numerical_data_filtered = numerical_data[selected_columns]

    numerical_data_filtered.to_csv(numerical_data_path, index=False)
    metadata_columns_with_labels = metadata_columns + [label_column, sex_condition_column, age_condition_column, 'labels_encoded']
    df_metadata = df[metadata_columns_with_labels]
    df_metadata.to_csv(metadata_path, index=False)
    print("Data saved successfully.")

# Split data into training, validation, and test sets
@time_tracker
def split_data(numerical_data_filtered, labels_encoded, split_sizes, random_state=42):
    splits = {}
    for size in split_sizes:
        if size <= len(numerical_data_filtered):
            print(f"Splitting data with size: {size}")
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=random_state)
            for train_idx, _ in splitter.split(numerical_data_filtered, labels_encoded):
                splits[size] = numerical_data_filtered.iloc[train_idx], [labels_encoded[i] for i in train_idx]
    return splits

@time_tracker
def train_val_test_split(data, labels, random_state=42):
    print("Splitting data into training, validation, and test sets...")
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        data, labels, test_size=0.3, random_state=random_state, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.4, random_state=random_state, stratify=y_remaining
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

@time_tracker
def scale_and_save_data(train_val_test_splits, output_path):
    scaler = StandardScaler()
    for key, (X_train, X_val, X_test, y_train, y_val, y_test) in train_val_test_splits.items():
        print(f"Scaling and saving data for split: {key}")
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        dataset_path = f'{output_path}/{key}/'
        directory_path = Path(dataset_path)
        directory_path.mkdir(parents=True, exist_ok=True)

        torch.save(train_dataset.tensors, f'{dataset_path}train_dataset_tensors.pt')
        torch.save(val_dataset.tensors, f'{dataset_path}val_dataset_tensors.pt')
        torch.save(test_dataset.tensors, f'{dataset_path}test_dataset_tensors.pt')
        print(f"Data for split {key} saved successfully.")

@time_tracker
def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16, random_state=42):
    print("Creating dataloaders...")
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

    print("Dataloaders created successfully.")
    return train_loader, val_loader, test_loader

# Main execution
print("Starting main execution...")
split_sizes = [5000, 10000, 20000, len(numerical_data_filtered)]
splits = split_data(numerical_data_filtered, labels_encoded, split_sizes)

# Generate shuffled data for null hypothesis
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

shuffled_data_10000, shuffled_labels_10000 = generate_shuffled_data(numerical_data_filtered, 10000)
splits['shuffled_10000'] = (shuffled_data_10000, shuffled_labels_10000)

train_val_test_splits = {
    key: train_val_test_split(data, labels) for key, (data, labels) in splits.items()
}

scale_and_save_data(train_val_test_splits, output_path='data/v2/m1')

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_splits[len(numerical_data_filtered)]
train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")
print("Main execution finished.")

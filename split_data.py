import os
import time
import random
import numpy as np
import pandas as pd
import torch
import argparse
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

# Split data into training, validation, and test sets
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
def train_val_test_split(data, labels, random_state=42):
    print("Splitting data into training, validation, and test sets...")
    
    # Convert labels list to pandas Series to use value_counts()
    import pandas as pd
    labels_series = pd.Series(labels, index=data.index)
    
    # Filter labels to exclude classes with fewer than 20 samples
    label_counts = labels_series.value_counts()
    valid_labels = label_counts[label_counts > 20].index

    # Select only data and labels corresponding to classes with more than 20 samples
    valid_indices = labels_series.isin(valid_labels)
    data_filtered = data.loc[valid_indices]
    labels_filtered = labels_series.loc[valid_indices]

    # Perform the split with stratification on filtered data
    from sklearn.model_selection import train_test_split

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        data_filtered, labels_filtered, test_size=0.3, random_state=random_state, stratify=labels_filtered
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.4, random_state=random_state, stratify=y_remaining
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test



@time_tracker
def scale_and_save_data(train_val_test_splits, output_path):
    scaler = StandardScaler()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_splits
    print(f"Scaling and saving data...")
    
    # Scaling the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Converting to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Convert labels to numpy arrays before converting to tensors
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # Creating TensorDataset objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create output directory if it does not exist
    dataset_path = f'{output_path}/{split_data_value}/'
    directory_path = Path(dataset_path)
    directory_path.mkdir(parents=True, exist_ok=True)

    # Save the datasets
    torch.save(train_dataset.tensors, f'{dataset_path}train_dataset_tensors.pt')
    torch.save(val_dataset.tensors, f'{dataset_path}val_dataset_tensors.pt')
    torch.save(test_dataset.tensors, f'{dataset_path}test_dataset_tensors.pt')
    print(f"Data saved successfully.")


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






set_seed(42)


# Load or prepare data
numerical_data_path = 'data/v2/numerical_data_filtered.csv'
metadata_path = 'data/v2/metadata_with_labels.csv'

print("Checking if preprocessed data exists...")
# if os.path.exists(numerical_data_path) and os.path.exists(metadata_path):
#     print("Loading preprocessed data from CSV files...")
#     numerical_data_filtered = pd.read_csv(numerical_data_path)
#     df_metadata = pd.read_csv(metadata_path)
# else:
if True:
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
    print("Data is processed successfully.")


    # # Save a subset of the numerical data filtered (first 20000 columns)
    numerical_data_filtered_subset = numerical_data_filtered.iloc[:, :20000]
    numerical_data_filtered_subset.to_csv('data/v2/numerical_data_filtered_subset.csv')
    # numerical_data_filtered.to_csv(numerical_data_path)
    metadata_columns_with_labels = metadata_columns + [label_column, sex_condition_column, age_condition_column, 'labels_encoded']
    df_metadata = df[metadata_columns_with_labels]
    df_metadata.to_csv(metadata_path)
    # print("Data saved successfully, including subset of first 20000 columns.")

    # numerical_data_filtered.to_csv(numerical_data_path, index=False)
    # metadata_columns_with_labels = metadata_columns + [label_column, sex_condition_column, age_condition_column, 'labels_encoded']
    # df_metadata = df[metadata_columns_with_labels]
    # df_metadata.to_csv(metadata_path, index=False)
    # print("Data saved successfully.")


# for split_data_value in ['shuffled_10000', '10000', '20000','37067']:
#     print(f"Splitting data with value: {split_data_value}")
#     # Main execution
#     print("Starting main execution...")
#     labels_encoded = df_metadata['labels_encoded']
#     data, labels = split_data(numerical_data_filtered, labels_encoded, split_data_value)

#     X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data, labels)
#     scale_and_save_data((X_train, X_val, X_test, y_train, y_val, y_test), output_path='data/v2/m1')

#     print(f"Training set size: {X_train.shape}")
#     print(f"Validation set size: {X_val.shape}")
#     print(f"Test set size: {X_test.shape}")
#     print("Main execution finished.")
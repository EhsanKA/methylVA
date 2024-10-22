from sklearn.preprocessing import QuantileTransformer, StandardScaler
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
from methylVA.utils.decorators import time_tracker

@time_tracker
def scale_and_save_data(train_val_test_splits, output_path, split_data_value=10000):
    scaler = StandardScaler()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_splits
    print(f"Scaling and saving data...")
    
    # Assuming X is your data matrix
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    X_transformed = quantile_transformer.fit_transform(X)

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

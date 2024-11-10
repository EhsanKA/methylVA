import torch
from torchvision import datasets
from torchvision.transforms import v2
import torch.utils.data
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def get_data_loaders():
    """
    Get the data loaders for the MNIST dataset.

    Returns:
        torch.utils.data.DataLoader: Training data loader
        torch.utils.data.DataLoader: Test data loader
    """

    batch_size = 128
    transform = v2.Compose([
        v2.ToImage(),                           # Convert tensor to image
        v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
        v2.Lambda(lambda x: x.view(-1) -0.5),   # Normalize to [-0.5, 0.5] and flatten to 1D tensor
    ])

    # Download and load the training data
    train_data = datasets.MNIST(
        '../data/MNIST_data/',
        download=True,
        train=True,
        transform=transform
    )

    # Download and load the test data
    test_data = datasets.MNIST(
        '../data/MNIST_data/',
        download=True,
        train=False,
        transform=transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


import torch
from torch.utils.data import DataLoader, Dataset

class UniformNoiseDataset(Dataset):
    def __init__(self, num_samples, noise_shape=(28, 28), num_classes=10):
        """
        Initialize the noise dataset.
        
        Args:
            num_samples (int): Number of samples in the dataset.
            noise_shape (tuple): Shape of the noise sample, e.g., (28, 28).
            num_classes (int): Number of target classes to simulate.
        """
        self.num_samples = num_samples
        self.noise_shape = noise_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random noise and scale it to the range [0, 1]
        noise = torch.rand(self.noise_shape, dtype=torch.float32) -0.5
        # Generate a random target label between 0 and num_classes-1
        target = torch.randint(0, self.num_classes, (1,)).item()
        return noise.view(-1), target  # Return flattened noise and target

def get_uniform_data_loaders():
    batch_size = 128
    num_train_samples = 60000  # Match MNIST training set size
    num_test_samples = 10000   # Match MNIST test set size

    # Create noise datasets
    train_data = UniformNoiseDataset(num_samples=num_train_samples)
    test_data = UniformNoiseDataset(num_samples=num_test_samples)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class GaussianNoiseDataset(Dataset):
    def __init__(self, num_samples, noise_shape=(28, 28), num_classes=10):
        """
        Initialize the Gaussian noise dataset with specified boundaries.
        
        Args:
            num_samples (int): Number of samples in the dataset.
            noise_shape (tuple): Shape of the noise sample, e.g., (28, 28).
            num_classes (int): Number of target classes to simulate.
        """
        self.num_samples = num_samples
        self.noise_shape = noise_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate Gaussian noise with mean 0 and standard deviation 1
        noise = torch.randn(self.noise_shape, dtype=torch.float32)
        
        # Clip the noise to be within [-0.5, 0.5]
        noise = torch.clamp(noise, min=-0.5, max=0.5)
        
        # Generate a random target label between 0 and num_classes-1
        target = torch.randint(0, self.num_classes, (1,)).item()
        
        return noise.view(-1), target  # Return flattened noise and target
    

def get_gaussian_data_loaders():
    batch_size = 128
    num_train_samples = 60000  # Match MNIST training set size
    num_test_samples = 10000   # Match MNIST test set size

    # Create noise datasets
    train_data = GaussianNoiseDataset(num_samples=num_train_samples)
    test_data = GaussianNoiseDataset(num_samples=num_test_samples)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



class MethylDataset(Dataset):
    def __init__(self, data=None, metadata=None,
                 pkl_file_data=None, pkl_file_metadata=None,
                 transform=None
                 ):
        
        if data is not None and metadata is not None:
            self.data = data
            self.metadata = metadata
            self.labels = self.metadata['labels_encoded'].values
        elif pkl_file_data is not None and pkl_file_metadata is not None:
            self.data = pd.read_pickle(pkl_file_data)
            # print(self.data.head())
            # .astype('float32')
            # self.data.set_index(self.data.columns[0], inplace=True)

            self.metadata = pd.read_pickle(pkl_file_metadata)
            self.labels = self.metadata['labels_encoded'].values
        else:
            raise ValueError("Either data and metadata or pkl files must be provided")
            
        self.transform = transform

        # Check for unique problematic values
        # Convert non-numeric values to NaN, keep them in self.data for inspection
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.data = np.round(self.data, decimals=4)

        if self.data.isna().any().any():
            print("Found NaN values in the data after conversion.")
            # print(self.data[self.data.isna().any(axis=1)])


    def __len__(self):
        # Returns the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get sample data
        sample = self.data.iloc[idx, :].values

        # Replace NaNs with 0.0 and convert to tensor
        sample = np.where(np.isnan(sample), 0.0, sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        sample = torch.where(torch.isinf(sample), torch.tensor(1.0, dtype=torch.float32), sample)
        sample -= 0.5

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply any transformations, if provided
        if self.transform:
            sample = self.transform(sample)

        return sample, label
    


def get_methyl_data_loaders(train_data_path,
                            train_metadata_path,
                            test_data_path,
                            test_metadata_path,
                            batch_size=128
                            ):  
    

    # Create noise datasets
    train_data = MethylDataset(pkl_file_data=train_data_path, pkl_file_metadata=train_metadata_path)
    test_data = MethylDataset(pkl_file_data=test_data_path, pkl_file_metadata=test_metadata_path)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader


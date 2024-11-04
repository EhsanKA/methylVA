from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class MethylDataset(Dataset):
    def __init__(self, csv_file_data, csv_file_metadata, transform=None):
        # Load data from the CSV file
        self.data = pd.read_csv(csv_file_data)
        self.metadata = pd.read_csv(csv_file_metadata)
        self.labels = self.metadata['labels_encoded'].values
        self.transform = transform

    def __len__(self):
        # Returns the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get sample data
        sample = self.data.iloc[idx, :].values
        # Get the corresponding label
        label = self.labels[idx]
        
        # Apply any transformations, if provided
        if self.transform:
            sample = self.transform(sample)

        # Convert to tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # Adjust dtype as needed

        return sample, label


def get_methyl_data_loaders(config):

    # Create noise datasets
    train_data = MethylDataset(config['train_data_path'], config['train_metadata_path'])
    test_data = MethylDataset(config['test_data_path'], config['test_metadata_path'])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader
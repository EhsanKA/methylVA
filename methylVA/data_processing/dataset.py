from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

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
            self.data.set_index(self.data.columns[0], inplace=True)

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

        # Replace NaNs with torch.nan and convert to tensor
        sample = np.where(np.isnan(sample), float('nan'), sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        sample = torch.where(torch.isinf(sample), torch.tensor(1.0, dtype=torch.float32), sample)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply any transformations, if provided
        if self.transform:
            sample = self.transform(sample)

        return sample, label



def get_methyl_data_loaders(config,):

    # Create noise datasets
    train_data = MethylDataset(config['train_data_path'], config['train_metadata_path'])
    test_data = MethylDataset(config['test_data_path'], config['test_metadata_path'])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader


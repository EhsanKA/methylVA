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
    def __init__(self, csv_file_data, csv_file_metadata, transform=None):
        # Load data from the CSV file and convert to numeric
        self.data = pd.read_csv(csv_file_data, index_col=0)
        self.metadata = pd.read_csv(csv_file_metadata)
        self.labels = self.metadata['labels_encoded'].values
        self.transform = transform

        # Check for unique problematic values
        # Convert non-numeric values to NaN, keep them in self.data for inspection
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.data = np.round(self.data, decimals=4)

        if self.data.isna().any().any():
            print("Found NaN values in the data after conversion.")
            # print(self.data[self.data.isna().any(axis=1)])


        # Flatten data to get all unique values
        # unique_values = pd.Series(self.data.values.ravel()).dropna().unique()

        # # Identify problematic values
        # problematic_values = []
        # for val in unique_values:
        #     if np.isinf(val):
        #         problematic_values.append(val)
        #     elif isinstance(val, str) or isinstance(val, object):
        #         problematic_values.append(val)

        # # Print out the unique problematic values
        # if problematic_values:
        #     print("Unique problematic values found in data:", set(problematic_values))
        # else:
        #     print("No problematic values (Inf, -Inf, non-numeric) detected in data.")

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



def get_methyl_data_loaders(config):

    # Create noise datasets
    train_data = MethylDataset(config['train_data_path'], config['train_metadata_path'])
    test_data = MethylDataset(config['test_data_path'], config['test_metadata_path'])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader
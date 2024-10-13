import pandas as pd
import torch

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables fast auto-tuning

# Set seed for reproducibility
set_seed(42)


df1 = pd.read_pickle('data/methyl_scores_v1_HM450k_1.pkl', compression="bz2")
df2 = pd.read_pickle('data/methyl_scores_v1_HM450k_2.pkl', compression="bz2")
df3 = pd.read_pickle('data/methyl_scores_v1_HM450k_3.pkl', compression="bz2")
df4 = pd.read_pickle('data/methyl_scores_v1_HM450k_4.pkl', compression="bz2")
df5 = pd.read_pickle('data/methyl_scores_v1_HM450k_5.pkl', compression="bz2")
df = pd.concat([df1, df2, df3, df4, df5], axis=0)

# Assuming `df` is your DataFrame
metadata_columns = ['id', 'geo_accession', 'title', 'sex', 'age', 'race',
                    'tissue', 'geo_platform', 'inferred_age_Hannum',
                    'inferred_age_SkinBlood', 'inferred_age_Horvath353']  # list of metadata columns

label_column = 'disease'  # column with target values for classification/regression
condition_column = 'inferred_sex'
numerical_data = df.drop(metadata_columns + [label_column] + [condition_column], axis=1)  # features for training

default_value = 'no_label'
df[label_column].fillna(default_value, inplace=True)

labels = df[label_column]  # target/label for model training
conditions = df[condition_column]  # target/label for model training



import matplotlib.pyplot as plt
import numpy as np

# # Calculate the percentage of NaN values in each column
nan_percentage = numerical_data.isna().sum(axis=0) / numerical_data.shape[0] * 100

# Plot the histogram of the percentage of NaN values per column
plt.figure(figsize=(10, 6))
plt.hist(nan_percentage, bins=50, edgecolor='k', alpha=0.7)
plt.title("Histogram of Percentage of NaN Values Per Column")
plt.xlabel("Percentage of NaN values")  
plt.ylabel("Number of Columns")
plt.grid(True)

plt.show()



# Select subset of columns where NaN percentage is less than 10%
selected_columns = nan_percentage[nan_percentage < 10].index.tolist()

# Create a new DataFrame with the selected columns
numerical_data_filtered = numerical_data[selected_columns]

from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# Convert categorical labels to one-hot vectors
labels_onehot = onehot_encoder.fit_transform(labels.values.reshape(-1, 1))

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.preprocessing import StandardScaler

# Start by assuming your data is unscaled
numerical_data_filtered = numerical_data_filtered.values  # Assuming this is your data
labels_onehot = labels_onehot  # Assuming these are your labels

# Convert one-hot encoded labels to class indices
labels_class = np.argmax(labels_onehot, axis=1)  # Convert one-hot to class labels

# Stratified shuffle split (30% test set, 70% train) based on class indices
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in splitter.split(numerical_data_filtered, labels_class):
    X_train, X_temp = numerical_data_filtered[train_index], numerical_data_filtered[test_index]
    y_train, y_temp = labels_onehot[train_index], labels_onehot[test_index]  # Use one-hot labels for actual training

# Split the temp set into validation and test sets (15% val, 15% test)
splitter_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for val_index, test_index in splitter_val_test.split(X_temp, np.argmax(y_temp, axis=1)):  # Use class labels here
    X_val, X_test = X_temp[val_index], X_temp[test_index]
    y_val, y_test = y_temp[val_index], y_temp[test_index]


# Now scale the data
SCALE = True
if SCALE:
    scaler = StandardScaler()

    # Fit the scaler only on the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaler to the validation and test sets
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
else:
    # If scaling is turned off, use the raw data
    X_train_scaled = X_train
    X_val_scaled = X_val
    X_test_scaled = X_test
    


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


# This saves both X (features) and y (labels) for train, val, and test datasets
from pathlib import Path
dataset_path = 'data/m1/'
directory_path = Path(dataset_path)
directory_path.mkdir(parents=True, exist_ok=True)

torch.save(train_dataset.tensors, f'{dataset_path}train_dataset_tensors.pt')  # Saves (X_train, y_train)
torch.save(val_dataset.tensors, f'{dataset_path}val_dataset_tensors.pt')      # Saves (X_val, y_val)
torch.save(test_dataset.tensors, f'{dataset_path}test_dataset_tensors.pt')    # Saves (X_test, y_test)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))

# Check the stratified split
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")


import os

# Set the environment variable inside the script
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[2048,1024,512], dropout_rate=0.2):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_layers = self.build_layers(input_dim, hidden_dims, dropout_rate)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)  # for mean
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)  # for log variance
        
        # Decoder
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder_layers =self.build_layers(latent_dim, decoder_hidden_dims, dropout_rate)
        self.fc_output = nn.Linear(hidden_dims[0], input_dim)
        # self.fc3 = nn.Linear(latent_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, input_dim)

    def build_layers(self, input_dim, hidden_dims, dropout_rate):
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = h_dim
        return nn.Sequential(*layers)
    
    def encode(self, x):
        h = self.encoder_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Check if logvar has NaN or Inf values
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print(f"NaN or Inf detected in logvar: logvar={logvar}")
        
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Calculate std from logvar
        std = torch.exp(0.5 * logvar)
        
        # Check if std has NaN or Inf values
        if torch.isnan(std).any() or torch.isinf(std).any():
            print(f"NaN or Inf detected in std computation: std={std}")
        
        # Sample from the latent space
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Check if z has NaN or Inf values
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"NaN or Inf detected in z computation: z={z}")
        
        return z

    def decode(self, z):
        h = self.decoder_layers(z)
        # h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc_output(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent_embedding(self, x):
        """
        Method to get the latent embedding (the `z` vector) for an input.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)  # this is the embedding
        return z


class VAE_Lightning(pl.LightningModule):
    def __init__(self, input_dim=485577, latent_dim=128, hidden_dims=[2048, 1024, 512], dropout_rate=0.2, lr=1e-6):
        super(VAE_Lightning, self).__init__()
        
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        self.model = VAE(input_dim, latent_dim, hidden_dims, dropout_rate)
        self.lr = lr
    
    def forward(self, x):
        mu, logvar = self.model.encode(x)
        z = self.model.reparameterize(mu, logvar)
        return z, mu, logvar

    def get_latent_embedding(self, x):
        return self.model.get_latent_embedding(x)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Step 1: Create mask before replacing NaN values
        mask = ~torch.isnan(x)  # mask where values are not NaN

        # Step 2: Replace NaNs with zero or another neutral value for forward pass
        x_filled = replace_nan_with_mean(x)
        # x_filled = torch.nan_to_num(x, nan=0.0)

        # Step 3: Pass through the model with filled values
        z, mu, logvar = self.forward(x_filled)
        x_hat, _, _ = self.model(x_filled)

        # Step 4: Use the original x (with NaNs) and mask to calculate the loss
        loss = self._vae_loss(x, x_hat, mu, logvar, mask)
        print(f"Training loss: {loss.item()}")

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Step 1: Create mask before replacing NaN values
        mask = ~torch.isnan(x)

        # Step 2: Replace NaNs with zero or another neutral value for forward pass
        x_filled = replace_nan_with_mean(x)
        # x_filled = torch.nan_to_num(x, nan=0.0)

        # Step 3: Pass through the model with filled values
        z, mu, logvar = self.forward(x_filled)
        x_hat, _, _ = self.model(x_filled)

        # Step 4: Use the original x (with NaNs) and mask to calculate the loss
        loss = self._vae_loss(x, x_hat, mu, logvar, mask)
        print(f"Validation loss: {loss.item()}")

        self.log('val_loss', loss, on_step=False, on_epoch=True)
  

    def _vae_loss(self, original_x, x_hat, mu, logvar, mask):
        # Apply mask to ignore NaN values in the loss calculation
        recon_loss = F.mse_loss(x_hat[mask], original_x[mask], reduction='mean')
    
        # Scale the KL divergence to balance the losses
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original_x.shape[0]  # Normalize by batch size or apply weighting
    
        return recon_loss + kl_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    

from pytorch_lightning.callbacks import Callback

class LossHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Access the loss for the last training epoch from the logs
        train_loss = trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the loss for the last validation epoch from the logs
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())



def replace_nan_with_mean(x):
    # Calculate the column-wise mean, ignoring NaNs
    col_mean = torch.nanmean(x, dim=0)
    
    # Find where NaN values are located
    nan_mask = torch.isnan(x)
    
    # Replace NaNs with the corresponding column means
    x[nan_mask] = torch.take(col_mean, nan_mask.nonzero()[:, 1])
    
    # Check if there are still NaN or Inf values
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN or Inf detected in the input data after imputation!")
    
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, loggers

# Create a logger
logger = loggers.CSVLogger('lightning_logs/', name='m1_vae')


pl.seed_everything(42)

# Initialize the VAE Lightning model
input_dim = X_train_tensor.shape[1]  # The number of input features
latent_dim = 256  # Latent dimension size, can be tuned
hidden_dims = [2048, 1024, 512]
dropout_rate = 0.2
lr = 1e-6

model = VAE_Lightning(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dims=hidden_dims,
    dropout_rate=dropout_rate,
    lr=lr)

# Training
loss_history_callback = LossHistoryCallback()
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min',
    dirpath=f'{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/',
    filename='m1-vae-{epoch:02d}-{val_loss:.2f}'
    )

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

trainer = pl.Trainer(
    max_epochs=200,
    gradient_clip_val=0.5,  # Clip gradients to avoid explosion
    callbacks=[checkpoint_callback, early_stopping_callback, loss_history_callback],
    precision=32,
    accelerator='gpu',          # Use 'gpu' or 'cpu'
    devices=1 if torch.cuda.is_available() else 'auto',  # Use 1 GPU or CPU ('auto' will pick the appropriate one)
    deterministic=True,  # Ensure reproducibility
    logger=logger
)
trainer.fit(model, train_loader, val_loader)


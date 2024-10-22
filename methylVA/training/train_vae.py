import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers
from methylVA.models.vae import VAE_Lightning
from methylVA.training.trainer_utils import LossHistoryCallback
import pandas as pd
import os
import pickle
from pathlib import Path
import numpy as np


def train_vae(train_config):
    # Extract training configuration details from the config
    # train_config = config['training']
    
    # Load the training, validation, and test datasets from the specified paths
    input_dir = train_config['input_dir']
    batch_size = train_config['batch_size']
    kl_weight = train_config.get('kl_weight', 1.0)
    
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

    pl.seed_everything(42)

    # Set up the VAE model
    input_dim = X_train.shape[1]  # The number of input features
    latent_dim = train_config['latent_dim']
    hidden_dims = train_config['hidden_dims']
    dropout_rate = train_config['dropout_rate']
    learning_rate = train_config['learning_rate']

    vae_model = VAE_Lightning(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        lr=learning_rate,
        kl_weight=kl_weight
    )

    # Set up callbacks and loggers for training
    output_dir = train_config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Logger
    logger = loggers.CSVLogger(output_dir, name=train_config['model_type'])

    # Checkpoint and early stopping
    checkpoint_monitor = train_config.get('checkpoint_monitor', 'val_loss')
    checkpoint_mode = train_config.get('checkpoint_mode', 'min')

    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_monitor,
        save_top_k=1,
        mode=checkpoint_mode,
        dirpath=f'{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/',
        filename='vae-{epoch:02d}-{val_loss:.2f}'
    )
    
    early_stopping_callback = EarlyStopping(
        monitor=checkpoint_monitor,
        patience=train_config.get('early_stopping_patience', 5)
    )

    loss_history_callback = LossHistoryCallback()

    # Setting up the trainer
    trainer = pl.Trainer(
        max_epochs=train_config['max_epochs'],
        gradient_clip_val=train_config.get('gradient_clip_val', 0.1),
        callbacks=[checkpoint_callback, early_stopping_callback, loss_history_callback],
        precision=32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        deterministic=True,
        logger=logger
    )

    # Train the model
    trainer.fit(vae_model, train_loader, val_loader)
    print(f"VAE model training completed.")

    # Optionally test the model after training
    # trainer.test(vae_model, test_loader)

    return vae_model, trainer


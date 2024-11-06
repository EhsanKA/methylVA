import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers
from methylVA.models.vae import VAE_Lightning
from methylVA.training.trainer_utils import LossHistoryCallback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import os
import pickle
from pathlib import Path
import numpy as np

from methylVA.data_processing.dataset import get_methyl_data_loaders




def train_vae(train_config, train_loader=None, val_loader=None):


    kl_weight = train_config.get('kl_weight', 1.0)
    patience = train_config.get('early_stopping_patience', None)

    pl.seed_everything(42)

    data_batch, _ = next(iter(train_loader))


    num_train_rows = len(train_loader.dataset)
    num_val_rows = len(val_loader.dataset)

    print("Number of features in each dataset:", data_batch.shape[1])
    print("Number of rows in the training dataset:", num_train_rows)
    print("Number of rows in the validation dataset:", num_val_rows)

    # Set up the VAE model
    input_dim = data_batch.shape[1]  # The number of input features
    latent_dim = train_config['latent_dim']
    hidden_dims = train_config['hidden_dims']
    dropout_rate = train_config['dropout_rate']
    learning_rate = train_config['learning_rate']
    activation = train_config.get('activation', 'Silu')
    batch_norm = train_config.get('batch_norm', True)

    vae_model = VAE_Lightning(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        lr=learning_rate,
        kl_weight=kl_weight,
        activation=activation,
        batch_norm=batch_norm
    )

    # Set up TensorBoard Logger
    output_dir = train_config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # TensorBoard Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_dir, name=train_config['model_type'])

    # Set up model checkpointing and early stopping
    checkpoint_monitor = train_config.get('checkpoint_monitor', 'Val/loss')
    checkpoint_mode = train_config.get('checkpoint_mode', 'min')

    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_monitor,
        save_top_k=1,
        mode=checkpoint_mode,
        dirpath=f'{tb_logger.log_dir}/checkpoints/',
        # dirpath=f'{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/',
        filename='vae-{epoch:02d}-{Val/loss:.2f}'

    )
    # loss_history_callback = LossHistoryCallback()
    # callbacks = [checkpoint_callback, loss_history_callback]

    callbacks = [checkpoint_callback]

    if patience:
        early_stopping_callback = EarlyStopping(
            monitor=checkpoint_monitor,
            patience=train_config.get('early_stopping_patience', patience)
        )
        callbacks.append(early_stopping_callback)

    # Set up the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=train_config['max_epochs'],
        gradient_clip_val=train_config.get('gradient_clip_val', 0.1),
        callbacks=callbacks,
        precision=32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        deterministic=True,
        logger=tb_logger  
    )

    # Train the model
    trainer.fit(vae_model, train_loader, val_loader)
    print(f"VAE model training completed.")

    # Optionally test the model after training
    # trainer.test(vae_model, test_loader)

    return vae_model, trainer


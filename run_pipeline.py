import yaml
from pathlib import Path
import numpy as np
from methylVA.utils.random_data import generate_random_noise_matrix, split_random_data_and_save
from methylVA.training.train_vae import train_vae
from methylVA.data_processing.split_data import train_val_test_split
from methylVA.utils.set_seed import set_seed
import pandas as pd 
from methylVA.utils.common import load_config
from methylVA.data_processing.preprocessing import select_HV_cpgs
from methylVA.data_processing.dataset import get_methyl_data_loaders


# Training Step
def train_vae_pipeline(config):
    train_config = config['training_vae']
    train_vae(train_config)
    print(f"VAE model training completed.")



# Run the full pipeline based on the config file provided
def run_pipeline(config_path):
    # Load the pipeline configuration
    config = load_config(config_path)
    set_seed(config['set_seed'])

    if 'hvcpg_selection' in config:
        select_HV_cpgs(config['hvcpg_selection'])


    if 'train_test_loader' in config:
        train_loader, val_loader  = get_methyl_data_loaders(config)
        
        if 'training_vae' in config:
            train_vae_pipeline(config, train_loader, val_loader)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the pipeline configuration file')
    args = parser.parse_args()

    run_pipeline(args.config)

import argparse
import yaml
from methylVA.data_processing.preprocessing import preprocess_data
from methylVA.training.train_vae import train_vae
# from methylVA.evaluation.evaluate import evaluate_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_pipeline(config):
    # Load and preprocess data
    data = preprocess_data(config['data'])
    
    # Train VAE model
    vae_model, train_loader, val_loader = train_vae(data, config['training'])
    
    # Evaluate the model
    # evaluate_model(vae_model, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_pipeline(config)

import torch
import yaml
from methylVA.models.vae import VAE_Lightning
from methylVA.utils.common import load_config

# Optional utility function to load the trained model checkpoint
def load_trained_vae(checkpoint_path, hparams_path, device='cuda'):
    
    hparams = load_config(hparams_path)
    vae_model = VAE_Lightning.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device if torch.cuda.is_available() else 'cpu'),
        **hparams
    )
    vae_model.eval()
    return vae_model

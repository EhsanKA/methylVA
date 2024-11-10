from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from methylVA.models.model import VAE

def training_args():
    """
    Get the training arguments for the VAE model.

    Returns:
        int: Batch size
        float: Learning rate
        float: Weight decay
        int: Number of epochs
        int: Latent dimension
        int: Hidden dimension
        torch.device: Device
        VAE: Model
        torch.optim.Optimizer: Optimizer
        SummaryWriter: TensorBoard writer
    """
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 50
    latent_dim = 2
    hidden_dim = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_dim=784, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.adamw(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(f'../../experiments/VAE_MNIST/{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    output_dict = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'device': device,
        'model': model,
        'optimizer': optimizer,
        'writer': writer
    }

    return output_dict
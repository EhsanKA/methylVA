from dataclasses import dataclass
import torch

@dataclass
class VAEOutput:
    """
    Dataclass to store the output of the VAE model.
    
    Attributes:
        z_dist (torch.distributions.MultivariateNormal): Multivariate normal distribution
        z_sample (torch.Tensor): Sampled data from the latent space
        x_recon (torch.Tensor): Reconstructed data
        loss (torch.Tensor): Total loss
        loss_recon (torch.Tensor): Reconstruction loss
        loss_kl (torch.Tensor): KL divergence loss
    """

    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor



@dataclass
class AEOutput:
    """
    Dataclass to store the output of the AE model.
    
    Attributes:
        z (torch.Tensor): Latent data
        x_recon (torch.Tensor): Reconstructed data
        loss (torch.Tensor): Total loss
    """

    z: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor
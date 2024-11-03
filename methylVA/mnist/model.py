import torch
import torch.nn as nn
import torch.nn.functional as F

from methylVA.mnist.vae_output import VAEOutput

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden dimensions
        latent_dim (int): Number of latent dimensions
    """

    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, latent_dim * 2), # 2 for mean and variance
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into latent space.
        
        Args:
            x (torch.Tensor): Input tensor
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Multivariate normal distribution
        """
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale) 
        
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=scale_tril)
    
    def reparameterize(self, dist):
        """
        Reparameterizes the distribution to sample from it.

        Args:
            dist (torch.distributions.MultivariateNormal): Multivariate normal distribution of the encoded input data

        Returns:
            torch.Tensor: Sampled data from the latent space

        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the latent data to the original input space.

        Args:
            z (torch.Tensor): Latent data

        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Forward pass of the VAE model.

        Args:
            x (torch.Tensor): Input tensor
            compute_loss (bool): Flag to compute the loss

        Returns:
            torch.Tensor: Reconstructed data
            torch.distributions.MultivariateNormal: Multivariate normal distribution
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        x_recon = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=x_recon,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        # compute loss terms
        loss_recon = F.binary_cross_entropy(x_recon, x + 0.5, reduction='none').sum(dim=-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.size(-1), device=z.device).unsqueeze(0).expand(z.size(0), -1, -1),
        )
        kl_loss_weight = 1.0

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        loss_kl = kl_loss_weight * loss_kl

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=x_recon,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
    


from methylVA.mnist.vae_output import AEOutput

class AE(nn.Module):
    """
    Autoencoder (AE) class.

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden dimensions
        latent_dim (int): Number of latent dimensions
    """

    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super(AE, self).__init__()

        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, latent_dim ), # 2 size of latent space
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, eps: float = 1e-8):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)

        # compute loss terms
        loss = F.binary_cross_entropy(x_recon, x + 0.5, reduction='none').sum(dim=-1).mean()

        return AEOutput(
            z=z,
            x_recon=x_recon,
            loss=loss,
        )
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from methylVA.training.trainer_utils import replace_nan_with_mean


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[2048,1024,512], dropout_rate=0.2,
                  activation='Silu', batch_norm=True):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_layers = self.build_layers(input_dim, hidden_dims, dropout_rate)
        self.activation = activation
        self.batch_norm = batch_norm

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)  # for mean
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)  # for log variance
        
        # Decoder
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder_layers =self.build_layers(latent_dim, decoder_hidden_dims, dropout_rate,
                                                self.activation, self.batch_norm) 
        self.fc_output = nn.Linear(hidden_dims[0], input_dim)


    def build_layers(self, input_dim, hidden_dims, dropout_rate, activation='Silu',
                      batch_norm=True):
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'Silu':
                layers.append(nn.SiLU())
            # layers.append(nn.Tanh())
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
        logvar = torch.clamp(logvar, min=-5, max=5)
        
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
    def __init__(self,
                 input_dim=5000,
                 latent_dim=128,
                 hidden_dims=[2048, 1024, 512],
                 dropout_rate=0.2,
                 lr=1e-6,
                 kl_weight=0.1,
                 activation='Silu',
                 batch_norm=True):
        super(VAE_Lightning, self).__init__()
        
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        self.model = VAE(input_dim, latent_dim, hidden_dims, dropout_rate,
                         activation=activation, batch_norm=batch_norm)
        self.lr = lr
        self.kl_weight = kl_weight
    
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
        loss, recon_loss, kl_loss = self._vae_loss(x, x_hat, mu, logvar, mask, self.kl_weight)

        print(f"Training loss: {loss.item()}")
    
        # self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step) 

        self.log('Train/loss', loss, on_step=False, on_epoch=True)
        self.log('Train/BCE_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('Train/KLD_loss', kl_loss, on_step=False, on_epoch=True)


        # Calculate and log gradient norm
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("Grad_norm/train", total_norm, on_step=True, on_epoch=True)

        # # Calculate Pearson correlation between input and output
        # corr = self.pearson_correlation(x, x_hat, mask)
        # # Log both step-wise and epoch-wise
        # self.log('train_corr_step', corr, on_step=True, on_epoch=False)
        # self.log('train_corr_epoch', corr, on_step=False, on_epoch=True)

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
        loss, recon_loss, kl_loss = self._vae_loss(x, x_hat, mu, logvar, mask, self.kl_weight)
        print(f"Validation loss: {loss.item()}")

        self.log('Val/loss', loss, on_step=False, on_epoch=True)
        self.log('Val/BCE_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('Val/KLD_loss', kl_loss, on_step=False, on_epoch=True)
  

    def _vae_loss(self, original_x, x_hat, mu, logvar, mask, kl_weight=1.0):
        # Apply mask to ignore NaN values in the loss calculation
        # recon_loss = F.mse_loss(x_hat[mask], original_x[mask], reduction='mean')
        ## todo: change the loss to BCE loss
        recon_loss = F.binary_cross_entropy(x_hat[mask], original_x[mask], reduction='mean')

        # Scale the KL divergence to balance the losses
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original_x.shape[0]  # Normalize by batch size or apply weighting
        
        kl_loss = kl_weight * kl_loss

        return recon_loss + kl_loss, recon_loss, kl_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def pearson_correlation(original_x, x_hat, mask=None):
        """
        Calculates the Pearson correlation between the input and the output vectors.

        Parameters:
        - original_x: torch.Tensor
            The original input tensor.
        - x_hat: torch.Tensor
            The reconstructed output tensor.
        - mask: torch.Tensor, optional
            A boolean mask to ignore NaN values or irrelevant elements.

        Returns:
        - correlation: float
            The Pearson correlation coefficient between original_x and x_hat.
        """
        if mask is not None:
            original_x = original_x[mask]
            x_hat = x_hat[mask]
        
        # Calculate mean and standard deviation
        mean_x = torch.mean(original_x)
        mean_x_hat = torch.mean(x_hat)
        
        std_x = torch.std(original_x)
        std_x_hat = torch.std(x_hat)
        
        # Calculate covariance
        covariance = torch.mean((original_x - mean_x) * (x_hat - mean_x_hat))
        
        # Calculate Pearson correlation
        correlation = covariance / (std_x * std_x_hat)
        
        return correlation.item()
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np


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

# Function to replace NaN values with the column-wise mean
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


def correlation_between_rows(matrix1, matrix2):
    """
    Calculates the correlation between corresponding rows of two NumPy matrices.
    Each row is treated as a separate sample.

    Parameters:
    - matrix1: numpy.ndarray
        The first matrix with shape (n_samples, n_features1).
    - matrix2: numpy.ndarray
        The second matrix with shape (n_samples, n_features2).

    Returns:
    - correlations: numpy.ndarray
        A 1D array containing the correlation between corresponding rows of matrix1 and matrix2.
    """
    # Ensure matrices have the same number of rows (samples)
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError("Both matrices must have the same number of rows (samples).")
    
    # Calculate correlation for each pair of rows
    correlations = np.array([np.corrcoef(matrix1[i], matrix2[i])[0, 1] for i in range(matrix1.shape[0])])
    
    return correlations


def reconstruct_data(model, data, num_rows=10):
    """
    Passes selected rows of data through the VAE model to get the reconstructed version.

    Parameters:
    - model: VAE_Lightning
        The loaded VAE model.
    - data: numpy.ndarray
        The input data matrix to be reconstructed.
    - num_rows: int
        The number of rows to select from the input data for reconstruction.

    Returns:
    - original_data: numpy.ndarray
        The selected original data as a matrix.
    - reconstructed_data: numpy.ndarray
        The reconstructed version of the selected data.
    """
    # Select specified number of rows
    selected_data = data[:num_rows]
    
    # Convert to PyTorch tensor and move to the correct device
    data_tensor = torch.tensor(selected_data, dtype=torch.float32).to(model.device)
    
    # Pass through VAE to get reconstructed version
    with torch.no_grad():
        mu, logvar = model.model.encode(data_tensor)
        z = model.model.reparameterize(mu, logvar)
        reconstructed_tensor = model.model.decode(z)
    
    # Convert reconstructed tensor to CPU and then to NumPy array
    reconstructed_data = reconstructed_tensor.cpu().numpy()
    
    return selected_data, reconstructed_data

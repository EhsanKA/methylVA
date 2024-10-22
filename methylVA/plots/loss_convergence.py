import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
import torch.nn.functional as F
import pandas as pd


def plot_training_and_validation_losses(file_path):
    """
    Plots training and validation losses, including reconstruction and KL divergence losses.

    Parameters:
    - file_path: str
        Path to the metrics CSV file for the model.
    """
    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.suptitle('Training and Validation Loss over Epochs', fontsize=24)

    # Read the metrics CSV file
    df_logs = pd.read_csv(file_path)

    # Extract relevant columns
    epochs = df_logs['epoch'].unique()  # Get unique epoch values
    train_loss = df_logs['train_loss'].dropna()  # Drop NaN values for train loss
    val_loss = df_logs['val_loss'].dropna()      # Drop NaN values for validation loss
    train_recon_loss = df_logs['train_recon_loss'].dropna()  # Drop NaN values for train reconstruction loss
    train_kl_loss = df_logs['train_kl_loss'].dropna()        # Drop NaN values for train KL divergence loss

    # Plot training and validation losses
    ax.plot(epochs[:len(train_loss)], train_loss, label='Training Loss', marker='o', markersize=8, linewidth=3)
    ax.plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', marker='o', markersize=8, linewidth=3)
    ax.plot(epochs[:len(train_recon_loss)], train_recon_loss, label='Train Recon Loss', linestyle='--', marker='x', markersize=8, linewidth=3)
    ax.plot(epochs[:len(train_kl_loss)], train_kl_loss, label='Train KL Loss', linestyle='-.', marker='s', markersize=8, linewidth=3)
    
    # Set plot labels and title
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    ax.legend(fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

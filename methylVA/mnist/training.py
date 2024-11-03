import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, dataloader, optimizer, prev_updates, batch_size=128, writer=None):
    """
    Trains the model on the given data.

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Data loader
        optimizer (torch.optim.Optimizer): Optimizer
        prev_updates (int): Number of updates made before this training
        writer (SummaryWriter, optional): TensorBoard writer
    """
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_updates = prev_updates + batch_idx
        
        data = data.to(model.device)

        optimizer.zero_grad()

        output = model(data) # Forward pass
        loss = output.loss

        loss.backward()  # Backward pass

        if n_updates % 100 == 0:
            # Calculate the log gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            print(f'Step {n_updates:,}, (N samples: {n_updates* batch_size:,}), Loss: {loss.item():.4f}, (Recon: {output.loss_recon.item():.4f}, KLD: {output.loss_kl.item():.4f}), Gradient norm: {total_norm:.4f}')
            
            if writer is not None:
                global_step = n_updates
                writer.add_scalar('Loss', loss.item(), global_step)
                writer.add_scalar('Loss/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradientNorm', total_norm, global_step)
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
    
    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, batch_size=128, writer=None):
    """
    Tests the model on the given data.

    Args:
        model (torch.nn.Module): Model to test
        dataloader (torch.utils.data.DataLoader): Data loader
        prev_updates (int): Number of updates made before this testing
        writer (SummaryWriter, optional): TensorBoard writer
    """

    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc='Testing')):
            data = data.to(model.device)
            data = data.view(data.size(0), -1) # Flatten the data

            output = model(data, compute_loss=True)

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
    
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f}, (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')

    if writer is not None:
        writer.add_scalar('Loss', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/KLD', output.loss_kl.item(), global_step=cur_step)

        # log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)

        # log random samples from the latent space
        z = torch.randn(16, model.latent_dim).to(model.device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)





import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_ae(model, dataloader, optimizer, prev_updates, batch_size=128, writer=None):
    """
    Trains the model on the given data.

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): Data loader
        optimizer (torch.optim.Optimizer): Optimizer
        prev_updates (int): Number of updates made before this training
        writer (SummaryWriter, optional): TensorBoard writer
    """
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_updates = prev_updates + batch_idx
        
        data = data.to(model.device)

        optimizer.zero_grad()

        output = model(data) # Forward pass
        loss = output.loss

        loss.backward()  # Backward pass

        if n_updates % 100 == 0:
            # Calculate the log gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            print(f'Step {n_updates:,}, (N samples: {n_updates* batch_size:,}), Loss: {loss.item():.4f}, Gradient norm: {total_norm:.4f}')
            
            if writer is not None:
                global_step = n_updates
                writer.add_scalar('Loss', loss.item(), global_step)
                writer.add_scalar('GradientNorm', total_norm, global_step)
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
    
    return prev_updates + len(dataloader)


def test_ae(model, dataloader, cur_step, batch_size=128, writer=None):
    """
    Tests the model on the given data.

    Args:
        model (torch.nn.Module): Model to test
        dataloader (torch.utils.data.DataLoader): Data loader
        prev_updates (int): Number of updates made before this testing
        writer (SummaryWriter, optional): TensorBoard writer
    """

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc='Testing')):
            data = data.to(model.device)
            data = data.view(data.size(0), -1) # Flatten the data

            output = model(data)

            test_loss += output.loss.item()
    
    test_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f}')

    if writer is not None:
        writer.add_scalar('Loss', test_loss, global_step=cur_step)

        # log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)

        # # log random samples from the latent space
        # z = torch.randn(16, model.latent_dim).to(model.device)
        # samples = model.decode(z)
        # writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)
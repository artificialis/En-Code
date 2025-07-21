"""
Training script for the Autoencoder model on the MNIST dataset.

This script handles the complete training pipeline for an autoencoder model:
- Loading configuration from a YAML file
- Setting up data loaders for the MNIST dataset
- Creating and training the autoencoder model
- Validating the model performance
- Visualizing reconstructed images
- Saving model checkpoints and results

The script can be run from the command line with optional arguments for
configuration file path and CUDA usage.

Example:
    python train.py --config config.yaml
"""

import os
import argparse
import yaml
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.pretty import Pretty
import wandb

from model import Autoencoder, ConvAutoEncoder

# Initialize Rich console
console = Console()

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary with model and training parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # end with
    return config

def get_data_loaders(batch_size, use_cuda=False):
    """
    Create data loaders for training and testing MNIST dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        use_cuda (bool): Whether to use CUDA acceleration and related optimizations
        
    Returns:
        tuple: (train_loader, test_loader)
            - train_loader (DataLoader): DataLoader for the training dataset
            - test_loader (DataLoader): DataLoader for the testing dataset
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load the test data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        **kwargs
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        **kwargs
    )
    
    return train_loader, test_loader

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, total_epochs, train_val_loss=None):
    """
    Train the model for one epoch.
    
    Args:
        model (Autoencoder): The autoencoder model to train
        device (torch.device): Device to use for training (CPU or CUDA)
        train_loader (DataLoader): DataLoader for the training dataset
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights
        criterion (torch.nn.Module): Loss function
        epoch (int): Current epoch number (for progress display)
        total_epochs (int): Total number of epochs for training
        train_val_loss (tuple, optional): Tuple containing (train_loss, val_loss) from previous epoch
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    # Create progress display for this epoch
    progress_desc = f"[cyan]Epoch {epoch}/{total_epochs}"
    if train_val_loss:
        progress_desc += f" [green]Train: {train_val_loss[0]:.6f}[/green] [yellow]Val: {train_val_loss[1]:.6f}[/yellow]"
    
    progress_obj = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]Batch Loss: {task.fields[loss]:.6f}"),
        TimeRemainingColumn(),
    )
    progress_obj.start()
    
    batch_task = progress_obj.add_task(progress_desc, total=total_batches, loss=0.0)
    
    try:
        for data, _ in train_loader:
            data = data.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, _ = model(data)
            
            # Flatten the input if it's not already flattened
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            # end if
            
            # Flatten the output if it's not already flattened
            if len(recon_batch.shape) > 2:
                recon_batch = recon_batch.reshape(recon_batch.size(0), -1)
            # end if
            
            # Calculate loss
            loss = criterion(recon_batch, data)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            current_loss = loss.item()
            progress_obj.update(batch_task, advance=1, loss=current_loss)
        # end for
    finally:
        progress_obj.stop()
    
    return train_loss / len(train_loader)

def validate(model, device, test_loader, criterion):
    """
    Validate the model on the test dataset.
    
    Args:
        model (Autoencoder): The autoencoder model to validate
        device (torch.device): Device to use for validation (CPU or CUDA)
        test_loader (DataLoader): DataLoader for the test dataset
        criterion (torch.nn.Module): Loss function
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Forward pass
            recon_batch, _ = model(data)
            
            # Flatten the input if it's not already flattened
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            # end if
            
            # Flatten the output if it's not already flattened
            if len(recon_batch.shape) > 2:
                recon_batch = recon_batch.reshape(recon_batch.size(0), -1)
            # end if
            
            # Calculate loss
            loss = criterion(recon_batch, data)
            val_loss += loss.item()
        # end for
    # end with
    
    return val_loss / len(test_loader)

def visualize_reconstruction(model, device, test_loader, epoch, save_dir='./results', args=None, config=None):
    """
    Visualize original and reconstructed images from the test dataset.
    
    Creates a figure with original images in the top row and their
    reconstructions in the bottom row, then saves it to disk.
    If wandb is enabled, also logs the images to wandb.
    
    Args:
        model (Autoencoder): The trained autoencoder model
        device (torch.device): Device to use for inference (CPU or CUDA)
        test_loader (DataLoader): DataLoader for the test dataset
        epoch (int): Current epoch number (used in the saved filename)
        save_dir (str): Directory to save the visualization images
        args (argparse.Namespace, optional): Command line arguments
        config (dict, optional): Configuration dictionary
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data, _ = next(iter(test_loader))
        data = data.to(device)
        
        # Reconstruct
        recon_batch, _ = model(data)
        
        # Plot
        n = min(8, data.size(0))
        plt.figure(figsize=(12, 6))
        
        for i in range(n):
            # Original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            # Reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_batch[i].cpu().numpy().reshape(28, 28), cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')
        # end for
        
        plt.tight_layout()
        fig_path = f"{save_dir}/reconstruction_epoch_{epoch}.png"
        plt.savefig(fig_path)
        
        # Log to wandb if enabled
        if args and config and not args.no_wandb and 'wandb' in config and wandb.run is not None:
            wandb.log({
                "reconstructions": wandb.Image(fig_path, caption=f"Epoch {epoch}")
            }, step=epoch)
            
            # Also log individual images for better visualization
            images = []
            for i in range(min(4, n)):  # Limit to 4 pairs to avoid clutter
                images.append(wandb.Image(
                    data[i].cpu().numpy().reshape(28, 28),
                    caption=f"Original {i+1}"
                ))
                images.append(wandb.Image(
                    recon_batch[i].cpu().numpy().reshape(28, 28),
                    caption=f"Reconstructed {i+1}"
                ))
            wandb.log({"samples": images}, step=epoch)
            
        plt.close()
    # end with

def main():
    """
    Main function to run the autoencoder training pipeline.
    
    This function:
    1. Parses command line arguments
    2. Loads configuration from the specified YAML file
    3. Sets up CUDA if available
    4. Creates data loaders for the MNIST dataset
    5. Initializes the autoencoder model
    6. Sets up loss function and optimizer
    7. Trains the model for the specified number of epochs
    8. Periodically visualizes reconstructions and saves model checkpoints
    9. Saves the final model and plots the loss curves
    """
    parser = argparse.ArgumentParser(description='Autoencoder for MNIST')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    
    # Weights & Biases arguments
    parser.add_argument('--wandb-project', type=str, help='Weights & Biases project name (overrides config)')
    parser.add_argument('--wandb-entity', type=str, help='Weights & Biases entity/username (overrides config)')
    parser.add_argument('--wandb-name', type=str, help='Weights & Biases run name (overrides config)')
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline', 'disabled'], 
                        help='Weights & Biases mode (overrides config)')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # CUDA setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    console.print(f"[bold green]Using device:[/bold green] [bold cyan]{device}[/bold cyan]")
    
    # Initialize Weights & Biases if not disabled
    if not args.no_wandb and 'wandb' in config:
        # Prepare wandb config, prioritizing command line arguments over config file
        wandb_config = {
            'project': args.wandb_project or config['wandb'].get('project', 'autoencoder-mnist'),
            'entity': args.wandb_entity or config['wandb'].get('entity'),
            'name': args.wandb_name or config['wandb'].get('name'),
            'tags': config['wandb'].get('tags', ['autoencoder', 'mnist']),
            'notes': config['wandb'].get('notes', 'Autoencoder training on MNIST dataset'),
            'mode': args.wandb_mode or config['wandb'].get('mode', 'online'),
            'config': {
                'model': config['model'],
                'training': config['training'],
                'device': str(device)
            }
        }
        
        # Initialize wandb
        console.print("[bold blue]Initializing Weights & Biases...[/bold blue]")
        wandb.init(**{k: v for k, v in wandb_config.items() if v is not None})
        console.print(f"[bold green]Weights & Biases initialized:[/bold green] [bold cyan]{wandb.run.name}[/bold cyan]")
    # end if
    
    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=config['training']['batch_size'],
        use_cuda=use_cuda
    )
    
    # Display experiment parameters
    console.print("\n[bold blue]Experiment Parameters:[/bold blue]")
    
    # Create a table for model parameters
    model_table = Table(title="Model Architecture")
    model_table.add_column("Parameter", style="cyan")
    model_table.add_column("Value", style="green")
    model_type = config['model'].get('model_type', 'model.Autoencoder')
    model_table.add_row("Model Type", str(model_type))
    if 'hidden_dims' in config['model']:
        model_table.add_row("Hidden Dimensions", str(config['model']['hidden_dims']))
    if 'hidden_channels' in config['model']:
        model_table.add_row("Hidden Channels", str(config['model']['hidden_channels']))
    model_table.add_row("Latent Dimension", str(config['model']['latent_dim']))
    model_table.add_row("Output activation", str(config['model']['output_activation']))
    console.print(model_table)
    
    # Create a table for training parameters
    training_table = Table(title="Training Parameters")
    training_table.add_column("Parameter", style="cyan")
    training_table.add_column("Value", style="green")
    training_table.add_row("Batch Size", str(config['training']['batch_size']))
    training_table.add_row("Epochs", str(config['training']['epochs']))
    training_table.add_row("Learning Rate", str(config['training']['learning_rate']))
    training_table.add_row("Weight Decay", str(config['training']['weight_decay']))
    training_table.add_row("Visualization Frequency", str(config['training']['vis_frequency']))
    training_table.add_row("Save Frequency", str(config['training']['save_frequency']))
    training_table.add_row("Criterion", str(config['training']['criterion']))
    console.print(training_table)

    def load_class(full_class_path):
        module_name, class_name = full_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    # end load_class
    
    # Create model
    model_class = load_class(model_type)
    
    # Initialize model based on its type
    if model_class == Autoencoder:
        model = model_class(
            input_dim=784,  # 28x28 MNIST images
            hidden_dims=config['model']['hidden_dims'],
            latent_dim=config['model']['latent_dim'],
            output_activation=config['model']['output_activation'],
        ).to(device)
    elif model_class == ConvAutoEncoder:
        model = model_class(
            input_channels=1,  # MNIST has 1 channel
            input_size=28,     # MNIST images are 28x28
            hidden_channels=config['model'].get('hidden_channels', [32, 64, 128]),
            latent_dim=config['model']['latent_dim'],
            output_activation=config['model']['output_activation'],
        ).to(device)
    else:
        # For other model types, pass all config parameters
        model = model_class(**config['model']).to(device)
    
    # Display model architecture
    console.print("\n[bold blue]Model Summary:[/bold blue]")
    console.print(Panel(Pretty(model), title=f"{model_class.__name__} Model"))
    
    # Define loss function and optimizer
    criterion_cls = load_class(config['training']['criterion'])
    criterion = criterion_cls()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create directories for results
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    total_epochs = config['training']['epochs']
    prev_losses = None
    
    for epoch in range(1, total_epochs + 1):
        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch, total_epochs, prev_losses)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, device, test_loader, criterion)
        val_losses.append(val_loss)
        
        # Store losses for next epoch's progress bar
        prev_losses = (train_loss, val_loss)
        
        # Print losses to console (without progress bar)
        console.print(f'[bold]Epoch: {epoch}[/bold], [green]Train Loss: {train_loss:.6f}[/green], [yellow]Val Loss: {val_loss:.6f}[/yellow]')
        
        # Log metrics to wandb if enabled
        if not args.no_wandb and 'wandb' in config and wandb.run is not None:
            log_freq = config['wandb'].get('log_freq', 1)
            if epoch % log_freq == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, step=epoch)
        
        # Visualize reconstructions every few epochs
        if epoch % config['training']['vis_frequency'] == 0:
            console.print(f"[italic]Visualizing reconstructions for epoch {epoch}...[/italic]")
            visualize_reconstruction(
                model,
                device,
                test_loader,
                epoch,
                save_dir=config['paths']['results_dir'],
                args=args, config=config
            )
        # end if
        
        # Save model checkpoint
        if epoch % config['training']['save_frequency'] == 0:
            console.print(f"[italic]Saving model checkpoint for epoch {epoch}...[/italic]")
            checkpoint_path = f"{config['paths']['model_dir']}/autoencoder_epoch_{epoch}.pt"
            torch.save(
                model.state_dict(),
                checkpoint_path
            )
            
            # Log model checkpoint to wandb if enabled
            if not args.no_wandb and 'wandb' in config and wandb.run is not None and config['wandb'].get('log_model', False):
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{epoch}",
                    type="model",
                    description=f"Model checkpoint at epoch {epoch}"
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
            # end if
        # end if
    # end for
    
    # Save the final model
    console.print("[bold green]Saving final model...[/bold green]")
    final_model_path = f"{config['paths']['model_dir']}/autoencoder_final.pt"
    torch.save(
        model.state_dict(),
        final_model_path
    )
    
    # Log final model to wandb if enabled
    if not args.no_wandb and 'wandb' in config and wandb.run is not None and config['wandb'].get('log_model', False):
        artifact = wandb.Artifact(
            name="model-final",
            type="model",
            description="Final trained model"
        )
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)
    # end if
    
    # Plot loss curves
    console.print("[bold green]Plotting loss curves...[/bold green]")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_curve_path = f"{config['paths']['results_dir']}/loss_curve.png"
    plt.savefig(loss_curve_path)
    
    # Log loss curve to wandb if enabled
    if not args.no_wandb and 'wandb' in config and wandb.run is not None:
        wandb.log({
            "loss_curve": wandb.Image(loss_curve_path, caption="Training and Validation Loss")
        })
        
        # Also log final metrics
        wandb.log({
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses)
        })
    # end if

    # Close
    plt.close()
    
    console.print("[bold green]Training completed![/bold green]")
# end main

if __name__ == "__main__":
    main()
# end if

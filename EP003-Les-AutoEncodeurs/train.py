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
import json
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
from safetensors.torch import save_file

from model import Autoencoder, ConvAutoEncoder
from visualization import (
    visualize_reconstruction,
    visualize_latent,
    visualize_latent_2d,
    visualize_digit_reconstruction,
    visualize_training_mse
)

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

def train_epoch(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        total_epochs,
        train_val_loss=None,
        iteration=0,
        config=None,
        args=None,
        test_loader=None,
        loss_tracker=None,
        val_loss_tracker=None
):
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
        iteration (int, optional): Current global iteration counter
        config (dict, optional): Configuration dictionary
        args (argparse.Namespace, optional): Command line arguments
        test_loader (DataLoader, optional): DataLoader for the test dataset (for visualization)
        loss_tracker (list, optional): List of dictionaries containing loss tracking data across all epochs
        val_loss_tracker (list, optional): List of dictionaries containing validation loss tracking data
        
    Returns:
        tuple: (average_training_loss, updated_iteration_counter)
    """
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    # Create progress display for this epoch
    progress_desc = f"[cyan]Epoch {epoch}/{total_epochs}"
    if train_val_loss:
        progress_desc += f" [green]Train: {train_val_loss[0]:.6f}[/green] [yellow]Val: {train_val_loss[1]:.6f}[/yellow]"
    # end if
    
    progress_obj = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]Batch Loss: {task.fields[loss]:.6f}"),
        TimeRemainingColumn(),
    )
    progress_obj.start()
    
    batch_task = progress_obj.add_task(progress_desc, total=total_batches, loss=0.0)
    
    # Get visualization parameters from config
    vis_threshold_iter = config['training'].get('vis_threshold_iter', 0) if config else 0
    vis_freq_before = config['training'].get('vis_freq_before', 0) if config else 0
    vis_freq_after = config['training'].get('vis_freq_after', 0) if config else 0

    # List of losses
    if loss_tracker is None:
        loss_tracker = list()
    
    try:
        # Iterate through training
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

            # Add to tracker
            loss_tracker.append({"epoch": epoch, "iteration": iteration, "loss": loss.item()})
            
            # Increment iteration counter
            iteration += 1
            
            # Check if visualization should be done based on iteration
            if config and test_loader and (
                (vis_freq_before > 0 and iteration < vis_threshold_iter and iteration % vis_freq_before == 0) or
                (vis_freq_after > 0 and iteration >= vis_threshold_iter and iteration % vis_freq_after == 0)
            ):
                console = Console()
                console.print(f"[italic]Visualizing reconstructions for iteration {iteration}...[/italic]")

                # Visualize reconstruction
                visualize_reconstruction(
                    model=model,
                    device=device,
                    test_loader=test_loader,
                    step=iteration,  # Use iteration instead of epoch
                    save_dir=config['paths']['results_dir']
                )

                # Save single digits reconstruction
                visualize_digit_reconstruction(
                    model=model,
                    device=device,
                    test_loader=test_loader,
                    n_samples=8,
                    step=iteration,
                    save_dir=config['paths']['results_dir']
                )
                
                # Visualize latent space using t-SNE
                console.print(f"[italic]Visualizing latent space for iteration {iteration}...[/italic]")
                
                # Set fixed random seed for reproducibility
                torch.manual_seed(42)
                np.random.seed(42)
                
                # Get a batch of test data
                all_latents = []
                all_digits = []
                
                # Number of samples to visualize
                n_samples = 500
                
                model.eval()
                with torch.no_grad():
                    # Collect latent representations and corresponding digits
                    for test_data, target in test_loader:
                        test_data = test_data.to(device)
                        
                        # Get latent representations
                        if hasattr(model, 'encode'):
                            latent = model.encode(test_data)
                        else:
                            _, latent = model(test_data)
                        # end if
                        
                        all_latents.append(latent.cpu())
                        all_digits.extend(target.tolist())
                        
                        if len(all_digits) >= n_samples:
                            break
                        # end if
                    # end
                    
                    # Concatenate all latents
                    all_latents = torch.cat(all_latents, dim=0)
                    
                    # Select a fixed random subset
                    if len(all_digits) > n_samples:
                        indices = np.random.choice(len(all_digits), n_samples, replace=False)
                        selected_latents = all_latents[indices]
                        selected_digits = [all_digits[i] for i in indices]
                    else:
                        selected_latents = all_latents
                        selected_digits = all_digits
                    # end if
                    
                    # Visualize latent space
                    visualize_latent(
                        latents=selected_latents,
                        digits=selected_digits,
                        step=iteration,  # Use iteration instead of epoch
                        save_dir=config['paths']['results_dir'],
                    )
                    
                    # Visualize latent space with specific dimensions
                    visualize_latent_2d(
                        selected_latents,
                        selected_digits,
                        iteration,  # Use iteration instead of epoch
                        dims=[0, 1],  # Use first two dimensions by default
                        save_dir=config['paths']['results_dir'],
                        args=args, config=config
                    )
                # end with
                
                # Visualize MSE (training and validation)
                console.print(f"[italic]Visualizing MSE for iteration {iteration}...[/italic]")
                visualize_training_mse(
                    loss_tracker=loss_tracker,
                    step=iteration,
                    save_dir=config['paths']['results_dir'],
                    max_iterations=config['training']['epochs'] * len(train_loader),  # Estimate total iterations
                    val_loss_tracker=val_loss_tracker
                )
                
                # Set model back to training mode
                model.train()
            # end if
        # end for
    finally:
        progress_obj.stop()
    # end try
    
    return (
        train_loss / len(train_loader),
        loss_tracker,  # Return the same loss_tracker that was passed in and updated
        iteration
    )
# end train_epoch

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
# end validate


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
    model_table.add_row("Output activation", str(config['model'].get('output_activation', 'torch.nn.Sigmoid')))
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
    training_table.add_row("Criterion", str(config['training'].get('criterion', 'torch.nn.MSELoss')))
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
            output_activation=config['model'].get('output_activation', 'torch.nn.Sigmoid'),
        ).to(device)
    elif model_class == ConvAutoEncoder:
        model = model_class(
            input_channels=1,  # MNIST has 1 channel
            input_size=28,     # MNIST images are 28x28
            hidden_channels=config['model'].get('hidden_channels', [32, 64, 128]),
            latent_dim=config['model']['latent_dim'],
            output_activation=config['model'].get('output_activation', 'torch.nn.Sigmoid'),
        ).to(device)
    else:
        # For other model types, pass all config parameters
        model = model_class(**config['model']).to(device)
    
    # Display model architecture
    console.print("\n[bold blue]Model Summary:[/bold blue]")
    console.print(Panel(Pretty(model), title=f"{model_class.__name__} Model"))
    
    # Define loss function and optimizer
    criterion_cls = load_class(config['training'].get('criterion', 'torch.nn.MSELoss'))
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
    
    # Initialize iteration counter
    iteration = 0
    
    # Get visualization parameters from config
    vis_threshold_iter = config['training'].get('vis_threshold_iter', 0)
    vis_freq_before = config['training'].get('vis_freq_before', 0)
    vis_freq_after = config['training'].get('vis_freq_after', 0)
    
    # Visualize at iteration 0 (before training starts)
    if vis_freq_before > 0 or vis_freq_after > 0:
        # Visualize reconstruction
        console.print("[italic]Visualizing reconstructions for iteration 0 (before training)...[/italic]")
        visualize_reconstruction(
            model=model,
            device=device,
            test_loader=test_loader,
            step=0,  # iteration 0
            save_dir=config['paths']['results_dir']
        )

        # Save single digits reconstruction
        visualize_digit_reconstruction(
            model=model,
            device=device,
            test_loader=test_loader,
            n_samples=8,
            step=0,
            save_dir=config['paths']['results_dir']
        )
        
        # Visualize latent space using t-SNE
        console.print("[italic]Visualizing latent space for iteration 0 (before training)...[/italic]")
        
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Get a batch of test data
        all_latents = []
        all_digits = []
        
        # Number of samples to visualize
        n_samples = 500
        
        model.eval()
        with torch.no_grad():
            # Collect latent representations and corresponding digits
            for data, target in test_loader:
                data = data.to(device)
                
                # Get latent representations
                if hasattr(model, 'encode'):
                    latent = model.encode(data)
                else:
                    _, latent = model(data)
                
                all_latents.append(latent.cpu())
                all_digits.extend(target.tolist())
                
                if len(all_digits) >= n_samples:
                    break
                # end if
            # end for
            
            # Concatenate all latents
            all_latents = torch.cat(all_latents, dim=0)
            
            # Select a fixed random subset
            if len(all_digits) > n_samples:
                indices = np.random.choice(len(all_digits), n_samples, replace=False)
                selected_latents = all_latents[indices]
                selected_digits = [all_digits[i] for i in indices]
            else:
                selected_latents = all_latents
                selected_digits = all_digits
            # end if
            
            # Visualize latent space
            visualize_latent(
                selected_latents,
                selected_digits,
                0,  # iteration 0
                save_dir=config['paths']['results_dir'],
            )
            
            # Visualize latent space with specific dimensions
            visualize_latent_2d(
                selected_latents,
                selected_digits,
                0,  # iteration 0
                dims=[0, 1],  # Use first two dimensions by default
                save_dir=config['paths']['results_dir'],
                args=args, config=config
            )
        # end with
        
        # Initialize loss_tracker with a placeholder entry for iteration 0
        loss_tracker = [{"epoch": 0, "iteration": 0, "loss": 0.0}]
        
        # Initialize val_loss_tracker (empty at this point)
        val_loss_tracker = list()
        
        # Visualize initial MSE plot (empty at this point)
        console.print("[italic]Visualizing initial MSE plot...[/italic]")
        visualize_training_mse(
            loss_tracker=loss_tracker,
            step=0,
            save_dir=config['paths']['results_dir'],
            max_iterations=config['training']['epochs'] * len(train_loader),  # Estimate total iterations
            val_loss_tracker=val_loss_tracker
        )
    # end if
    for epoch in range(1, total_epochs + 1):
        # Train (now returns updated iteration counter)
        train_loss, epoch_loss_tracker, iteration = train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            total_epochs=total_epochs,
            train_val_loss=prev_losses,
            iteration=iteration,
            config=config,
            args=args,
            test_loader=test_loader,
            loss_tracker=loss_tracker,
            val_loss_tracker=val_loss_tracker
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, device, test_loader, criterion)
        val_losses.append(val_loss)
        val_loss_tracker.append({"epoch": epoch, "val_loss": val_loss})
        
        # Save training and validation losses to text files
        train_loss_file_path = os.path.join(config['paths']['results_dir'], 'train_loss.json')
        val_loss_file_path = os.path.join(config['paths']['results_dir'], 'val_loss.json')
        
        # Append current losses
        with open(train_loss_file_path, 'a') as f:
            json.dump(
                obj={"train_losses": train_losses},
                fp=f
            )
        # end with
            
        with open(val_loss_file_path, 'a') as f:
            json.dump(
                obj={"val_losses": val_losses},
                fp=f
            )
        # end with
        
        # Store losses for next epoch's progress bar
        prev_losses = (train_loss, val_loss)
        
        # Print losses to console (without progress bar)
        console.print(f'[bold]Epoch: {epoch}[/bold], [green]Train Loss: {train_loss:.6f}[/green], [yellow]Val Loss: {val_loss:.6f}[/yellow], [blue]Iteration: {iteration}[/blue]')
        
        # Log metrics to wandb if enabled
        if not args.no_wandb and 'wandb' in config and wandb.run is not None:
            log_freq = config['wandb'].get('log_freq', 1)
            if epoch % log_freq == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'iteration': iteration,
                }, step=epoch)
        
        # Save model checkpoint
        if epoch % config['training']['save_frequency'] == 0:
            console.print(f"[italic]Saving model checkpoint for epoch {epoch}...[/italic]")
            
            # Save model in safetensors format
            checkpoint_path = f"{config['paths']['model_dir']}/autoencoder_epoch_{epoch}.safetensors"
            save_file(model.state_dict(), checkpoint_path)
            
            # Create YAML file with model parameters
            yaml_path = f"{config['paths']['model_dir']}/autoencoder_epoch_{epoch}.yaml"
            
            # Extract model parameters
            model_params = {}
            model_params['model_type'] = model_class.__module__ + "." + model_class.__name__
            
            # Add model-specific parameters
            if model_class == Autoencoder:
                model_params['input_dim'] = 784  # 28x28 MNIST images
                model_params['hidden_dims'] = config['model']['hidden_dims']
                model_params['latent_dim'] = config['model']['latent_dim']
                model_params['output_activation'] = config['model'].get('output_activation', 'torch.nn.Sigmoid')
            elif model_class == ConvAutoEncoder:
                model_params['input_channels'] = 1  # MNIST has 1 channel
                model_params['input_size'] = 28     # MNIST images are 28x28
                model_params['hidden_channels'] = config['model'].get('hidden_channels', [32, 64, 128])
                model_params['latent_dim'] = config['model']['latent_dim']
                model_params['output_activation'] = config['model'].get('output_activation', 'sigmoid')
            else:
                # For other model types, include all config parameters
                model_params.update(config['model'])
            
            # Save parameters to YAML file
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(model_params, yaml_file, default_flow_style=False)
            
            console.print(f"[italic]Saved model parameters to {yaml_path}[/italic]")
            
            # Log model checkpoint to wandb if enabled
            if not args.no_wandb and 'wandb' in config and wandb.run is not None and config['wandb'].get('log_model', False):
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{epoch}",
                    type="model",
                    description=f"Model checkpoint at epoch {epoch}"
                )
                artifact.add_file(checkpoint_path)
                artifact.add_file(yaml_path)
                wandb.log_artifact(artifact)
            # end if
        # end if
    # end for
    
    # Save the final model
    console.print("[bold green]Saving final model...[/bold green]")
    
    # Save model in safetensors format
    final_model_path = f"{config['paths']['model_dir']}/autoencoder_final.safetensors"
    save_file(model.state_dict(), final_model_path)
    
    # Create YAML file with model parameters
    yaml_path = f"{config['paths']['model_dir']}/autoencoder_final.yaml"
    
    # Extract model parameters
    model_params = {}
    model_params['model_type'] = model_class.__module__ + "." + model_class.__name__
    
    # Add model-specific parameters
    if model_class == Autoencoder:
        model_params['input_dim'] = 784  # 28x28 MNIST images
        model_params['hidden_dims'] = config['model']['hidden_dims']
        model_params['latent_dim'] = config['model']['latent_dim']
        model_params['output_activation'] = config['model'].get('output_activation', 'sigmoid')
    elif model_class == ConvAutoEncoder:
        model_params['input_channels'] = 1  # MNIST has 1 channel
        model_params['input_size'] = 28     # MNIST images are 28x28
        model_params['hidden_channels'] = config['model'].get('hidden_channels', [32, 64, 128])
        model_params['latent_dim'] = config['model']['latent_dim']
        model_params['output_activation'] = config['model'].get('output_activation', 'sigmoid')
    else:
        # For other model types, include all config parameters
        model_params.update(config['model'])
    
    # Save parameters to YAML file
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(model_params, yaml_file, default_flow_style=False)
    
    console.print(f"[italic]Saved model parameters to {yaml_path}[/italic]")
    
    # Log final model to wandb if enabled
    if not args.no_wandb and 'wandb' in config and wandb.run is not None and config['wandb'].get('log_model', False):
        artifact = wandb.Artifact(
            name="model-final",
            type="model",
            description="Final trained model"
        )
        artifact.add_file(final_model_path)
        artifact.add_file(yaml_path)
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

"""
Visualization utilities for the Autoencoder model.

This module contains functions for visualizing the results of the autoencoder model,
including reconstructed images and other visualizations.

Each type of visualization is saved in its own subdirectory:
- reconstructions: Contains original and reconstructed image comparisons
- latent_tsne: Contains t-SNE projections of the latent space
- latent_2d: Contains 2D projections of specific latent dimensions
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.manifold import TSNE


def visualize_digit_reconstruction(
        model,
        device,
        test_loader,
        n_samples: int,
        step=0,
        save_dir="./results"
):
    """
    Visualize the results of the autoencoder model on digit at a time.

    Args:
        model (Autoencoder): The trained autoencoder model
        device (torch.device): Device to use for inference (CPU or CUDA)
        test_loader (DataLoader): DataLoader for the test dataset
        n_samples (int): Number of samples to visualize
        step (int): Current step number (iteration or epoch, used in the saved filename)
        save_dir (str): Base directory where a 'reconstructions' subdirectory will be created to save the images
    """
    # Create main results directory
    os.makedirs(save_dir, exist_ok=True)

    # Create specific subdirectory for reconstruction plots
    reconstruction_dir = os.path.join(save_dir, "reconstructions")
    os.makedirs(reconstruction_dir, exist_ok=True)

    # Create specific subdirectory for this sample
    for sample_i in range(n_samples):
        sample_dir = os.path.join(reconstruction_dir, f"{sample_i}")
        os.makedirs(sample_dir, exist_ok=True)
    # end for

    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data, _ = next(iter(test_loader))
        data = data.to(device)

        # We have enough sampels
        assert n_samples <= data.size(0)

        # Reconstruct
        recon_batch, _ = model(data)

        for sample_i in range(n_samples):
            # Reshape
            digit_original = data[sample_i, ...].detach().cpu().numpy()
            digit_reconstruction = recon_batch[sample_i].detach().cpu().numpy()

            # Save original
            plt.imsave(
                f"{reconstruction_dir}/{sample_i}/original.png",
                digit_original.reshape(28, 28),
                cmap='gray'
            )

            # Save reconstruction
            plt.imsave(
                f"{reconstruction_dir}/{sample_i}/recon_{step:05d}.png",
                digit_reconstruction.reshape(28, 28),
                cmap='gray'
            )
        # end for
    # end with
# end visualize_digit_reconstruction


def visualize_reconstruction(
        model,
        device,
        test_loader,
        step,
        save_dir='./results'
):
    """
    Visualize original and reconstructed images from the test dataset.
    
    Creates a figure with original images in the top row and their
    reconstructions in the bottom row, then saves it to disk.
    If wandb is enabled, also logs the images to wandb.
    
    Args:
        model (Autoencoder): The trained autoencoder model
        device (torch.device): Device to use for inference (CPU or CUDA)
        test_loader (DataLoader): DataLoader for the test dataset
        step (int): Current step number (iteration or epoch, used in the saved filename)
        save_dir (str): Base directory where a 'reconstructions' subdirectory will be created to save the images
    """
    # Create main results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create specific subdirectory for reconstruction plots
    reconstruction_dir = os.path.join(save_dir, "reconstructions")
    os.makedirs(reconstruction_dir, exist_ok=True)
    
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
        fig_path = f"{reconstruction_dir}/reconstruction_iter_{step}.png"
        plt.savefig(fig_path)

        plt.close()
    # end with
# end visualize_reconstruction


def visualize_latent(
        latents,
        digits,
        step,
        save_dir='./results'
):
    """
    Visualize the latent space using t-SNE projection.
    
    Creates a 2D projection of the latent representations using t-SNE and colors
    each point according to its digit class.
    
    Args:
        latents (torch.Tensor): Tensor of latent representations with shape (n, latent_size)
        digits (list): List of n numbers representing the digit of each latent representation
        step (int): Current step number (iteration or epoch, used in the saved filename)
        save_dir (str): Base directory where a 'latent_tsne' subdirectory will be created to save the images
    """
    # Create main results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create specific subdirectory for latent TSNE plots
    latent_tsne_dir = os.path.join(save_dir, "latent_tsne")
    os.makedirs(latent_tsne_dir, exist_ok=True)
    
    # Convert latents to numpy for t-SNE
    latents_np = latents.cpu().numpy()
    
    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents_np)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=digits, cmap='tab10', 
                         alpha=0.8, s=50, edgecolors='w')
    
    # Add colorbar and labels
    plt.colorbar(scatter, label='Digit')
    plt.title(f'Latent Space Visualization (t-SNE) - Iteration {step}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.ylim(-35, 35)
    plt.xlim(-35, 35)
    
    # Save the figure
    fig_path = f"{latent_tsne_dir}/latent_tsne_iter_{step}.png"
    plt.savefig(fig_path)
    
    # Log to wandb if enabled
    # if args and config and not args.no_wandb and 'wandb' in config and wandb.run is not None:
    #     wandb.log({
    #         "latent_space": wandb.Image(fig_path, caption=f"Latent Space (t-SNE) - Iteration {step}")
    #     }, step=step)
    # # end if
    
    plt.close()
# end visualize_latent


def visualize_latent_2d(latents, digits, step, dims=[0, 1], save_dir='./results', args=None, config=None):
    """
    Visualize the latent space by projecting to 2D using two specific dimensions.
    
    Creates a 2D projection of the latent representations by taking two specific dimensions
    and colors each point according to its digit class.
    
    Args:
        latents (torch.Tensor): Tensor of latent representations with shape (n, latent_size)
        digits (list): List of n numbers representing the digit of each latent representation
        step (int): Current step number (iteration or epoch, used in the saved filename)
        dims (list): List of two integers specifying which dimensions to use for projection
        save_dir (str): Base directory where a 'latent_2d' subdirectory will be created to save the images
        args (argparse.Namespace, optional): Command line arguments
        config (dict, optional): Configuration dictionary
    """
    # Create main results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create specific subdirectory for latent 2D plots
    latent_2d_dir = os.path.join(save_dir, "latent_2d")
    os.makedirs(latent_2d_dir, exist_ok=True)
    
    # Convert latents to numpy
    latents_np = latents.cpu().numpy()
    
    # Extract the specified dimensions
    dim1, dim2 = dims
    latents_2d = latents_np[:, [dim1, dim2]]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=digits, cmap='tab10', 
                         alpha=0.8, s=50, edgecolors='w')
    
    # Add colorbar and labels
    plt.colorbar(scatter, label='Digit')
    plt.title(f'Latent Space Visualization (Dims {dim1},{dim2}) - Iteration {step}')
    plt.xlabel(f'Latent Dimension {dim1}')
    plt.ylabel(f'Latent Dimension {dim2}')
    plt.ylim(-2.0, 2.0)
    plt.xlim(-1.0, 3.5)
    
    # Save the figure
    fig_path = f"{latent_2d_dir}/latent_dims_{dim1}_{dim2}_iter_{step}.png"
    plt.savefig(fig_path)
    
    # Log to wandb if enabled
    # if args and config and not args.no_wandb and 'wandb' in config and wandb.run is not None:
    #     wandb.log({
    #         f"latent_space_dims_{dim1}_{dim2}": wandb.Image(fig_path, caption=f"Latent Space (Dims {dim1},{dim2}) - Iteration {step}")
    #     }, step=step)
    # # end if
    
    plt.close()
# end visualize_latent_2d


def visualize_training_mse(loss_tracker, step, save_dir='./results', max_iterations=None, val_loss_tracker=None):
    """
    Visualize the training MSE over iterations.
    
    Creates a plot of the training MSE over iterations with a transparent background,
    white grid, axes, title, and ticks. The plot is saved with a consistent scale
    to create a stable animation.
    
    Args:
        loss_tracker (list): List of dictionaries containing 'iteration' and 'loss' values
        step (int): Current iteration number (used in the saved filename)
        save_dir (str): Base directory where a 'training_mse' subdirectory will be created to save the images
        max_iterations (int, optional): Maximum number of iterations to show on x-axis (for stable animation)
        val_loss_tracker (list, optional): List of dictionaries containing 'epoch' and 'val_loss' values
    """
    # Create main results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create specific subdirectory for training MSE plots
    mse_dir = os.path.join(save_dir, "training_mse")
    os.makedirs(mse_dir, exist_ok=True)
    
    # Extract iterations and losses
    iterations = [entry['iteration'] for entry in loss_tracker]
    losses = [entry['loss'] for entry in loss_tracker]
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(10, 6), facecolor='none')
    ax = fig.add_subplot(111)
    
    # Set background to transparent
    ax.patch.set_alpha(0.0)

    # Plot the training MSE curve
    ax.plot(iterations[1:], losses[1:], '-', linewidth=2.0, color="yellow", label="Training")
    
    # Plot validation MSE if provided
    if val_loss_tracker is not None and len(val_loss_tracker) > 0:
        # Extract validation epochs and losses
        val_epochs = [entry['epoch'] for entry in val_loss_tracker]
        val_losses = [entry['val_loss'] for entry in val_loss_tracker]
        
        # Map epochs to iterations for plotting on the same scale
        # We'll use the last iteration of each epoch as the x-coordinate
        # since validation is performed at the end of each epoch
        epoch_to_iteration = {}
        for entry in loss_tracker:
            # Always update with the latest iteration for each epoch
            epoch_to_iteration[entry['epoch']] = entry['iteration']
        
        # Filter out any validation epochs that don't have a corresponding training iteration
        valid_indices = []
        valid_val_iterations = []
        valid_val_losses = []
        
        for i, epoch in enumerate(val_epochs):
            iteration = epoch_to_iteration.get(epoch)
            if iteration is not None and iteration > 0:
                valid_indices.append(i)
                valid_val_iterations.append(iteration)
                valid_val_losses.append(val_losses[i])
        
        # Plot validation curve in bright purple if we have valid data
        if valid_val_iterations and valid_val_losses:
            ax.plot(valid_val_iterations, valid_val_losses, '-', linewidth=2.0, color="#FF00FF", label="Validation")
    
    # Set white grid, axes, title, and ticks
    ax.grid(True, color='white', linestyle='-', linewidth=1.2, alpha=0.7)
    ax.set_title(f'MSE Loss - Iteration {step}', color='white')
    ax.set_xlabel('Iteration', color='white')
    ax.set_ylabel('MSE Loss', color='white')
    
    # Always add legend with both training and validation entries
    #legend = ax.legend(loc='upper right')
    #for text in legend.get_texts():
    #    text.set_color('white')
    
    # Set tick colors to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Set spine colors to white
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Set fixed limits for stable animation
    if max_iterations is not None:
        # ax.set_xlim(0, max_iterations)
        ax.set_xlim(-100, 10000)
    else:
        # If max_iterations not provided, use a reasonable buffer
        ax.set_xlim(-100, max(iterations[-1] * 1.1, 100))
    # end if
    
    # Set y-axis limits with a reasonable buffer
    max_loss = max(losses) if losses else 1.0
    min_loss = min(losses) if losses else 0.0
    buffer = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
    # ax.set_ylim(max(0, min_loss - buffer), max_loss + buffer)
    ax.set_ylim(0, 0.3)
    
    # Save the figure with transparent background
    fig_path = f"{mse_dir}/training_mse_iter_{step:06d}.png"
    plt.savefig(fig_path, transparent=True, bbox_inches='tight')
    
    plt.close()
# end visualize_training_mse
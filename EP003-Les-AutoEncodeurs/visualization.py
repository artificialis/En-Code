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


def visualize_reconstruction(model, device, test_loader, step, save_dir='./results', args=None, config=None):
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
        args (argparse.Namespace, optional): Command line arguments
        config (dict, optional): Configuration dictionary
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
        
        # Log to wandb if enabled
        if args and config and not args.no_wandb and 'wandb' in config and wandb.run is not None:
            wandb.log({
                "reconstructions": wandb.Image(fig_path, caption=f"Iteration {step}")
            }, step=step)
            
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
            # end for
            wandb.log({"samples": images}, step=step)
        # end if
        plt.close()
    # end with
# end visualize_reconstruction


def visualize_latent(latents, digits, step, save_dir='./results', args=None, config=None):
    """
    Visualize the latent space using t-SNE projection.
    
    Creates a 2D projection of the latent representations using t-SNE and colors
    each point according to its digit class.
    
    Args:
        latents (torch.Tensor): Tensor of latent representations with shape (n, latent_size)
        digits (list): List of n numbers representing the digit of each latent representation
        step (int): Current step number (iteration or epoch, used in the saved filename)
        save_dir (str): Base directory where a 'latent_tsne' subdirectory will be created to save the images
        args (argparse.Namespace, optional): Command line arguments
        config (dict, optional): Configuration dictionary
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
    
    # Save the figure
    fig_path = f"{latent_tsne_dir}/latent_tsne_iter_{step}.png"
    plt.savefig(fig_path)
    
    # Log to wandb if enabled
    if args and config and not args.no_wandb and 'wandb' in config and wandb.run is not None:
        wandb.log({
            "latent_space": wandb.Image(fig_path, caption=f"Latent Space (t-SNE) - Iteration {step}")
        }, step=step)
    # end if
    
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
    
    # Save the figure
    fig_path = f"{latent_2d_dir}/latent_dims_{dim1}_{dim2}_iter_{step}.png"
    plt.savefig(fig_path)
    
    # Log to wandb if enabled
    if args and config and not args.no_wandb and 'wandb' in config and wandb.run is not None:
        wandb.log({
            f"latent_space_dims_{dim1}_{dim2}": wandb.Image(fig_path, caption=f"Latent Space (Dims {dim1},{dim2}) - Iteration {step}")
        }, step=step)
    # end if
    
    plt.close()
# end visualize_latent_2d
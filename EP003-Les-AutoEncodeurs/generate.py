#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate random images from a trained autoencoder model.

This script loads a trained autoencoder model (safetensors + yaml config) and generates
random images by sampling random latent vectors from a normal distribution with
configurable mean and standard deviation, then passing them through the decoder.

Usage:
    python generate.py --model_path <path_to_safetensors_file> --config_path <path_to_yaml_config> 
                      --output_dir <output_directory> [options]

Required Arguments:
    --model_path      Path to the model safetensors file
    --config_path     Path to the model YAML config file
    --output_dir      Output directory for generated images

Optional Arguments:
    --n_samples       Number of samples to generate (default: 10)
    --mu              Mean of the normal distribution (default: 0.0)
    --sigma           Standard deviation of the normal distribution (default: 1.0)
    --grid            Create a grid image of all generated samples
    --grid_rows       Number of rows in the grid (default: 5)
    --grid_cols       Number of columns in the grid (default: 5)
    --no-cuda         Disables CUDA even if available

Examples:
    # Generate 10 samples with default parameters
    python generate.py --model_path models/autoencoder.safetensors --config_path models/autoencoder.yaml 
                      --output_dir generated_images

    # Generate 20 samples with custom distribution parameters and create a grid
    python generate.py --model_path models/autoencoder.safetensors --config_path models/autoencoder.yaml 
                      --output_dir generated_images --n_samples 20 --mu 0.5 --sigma 0.8 --grid

    # Generate 15 samples and create a custom-sized grid
    python generate.py --model_path models/autoencoder.safetensors --config_path models/autoencoder.yaml 
                      --output_dir generated_images --n_samples 15 --grid --grid_rows 3 --grid_cols 5
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import importlib
from datetime import datetime
from tqdm import tqdm


def load_class(full_class_path):
    """
    Load a class from a string path.
    
    Args:
        full_class_path (str): Full path to the class (e.g., 'model.ConvAutoEncoder')
        
    Returns:
        class: The loaded class
    """
    module_name, class_name = full_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_model(model_path, config_path, device):
    """
    Load a model from safetensors file using its configuration.
    
    Args:
        model_path (str): Path to the safetensors file
        config_path (str): Path to the YAML configuration file
        device (torch.device): Device to load the model on
        
    Returns:
        nn.Module: The loaded model
        dict: The model configuration
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model configuration
    # Check if config has a 'model' section
    if 'model' in config and isinstance(config['model'], dict):
        model_config = config['model']
        model_type = model_config.get('model_type')
    else:
        # If no 'model' section, use the top-level config
        model_config = config
        model_type = model_config.get('model_type')
    
    if not model_type:
        # Default model type if not specified
        model_type = 'model.Autoencoder'
    
    # Get model class
    model_class = load_class(model_type)
    
    # Initialize model based on its type
    if 'Autoencoder' in model_type:
        if 'ConvAutoEncoder' in model_type:
            model = model_class(
                input_channels=model_config.get('input_channels', 1),
                input_size=model_config.get('input_size', 28),
                hidden_channels=model_config.get('hidden_channels', [32, 64, 128]),
                latent_dim=model_config.get('latent_dim', 10),
                output_activation=model_config.get('output_activation', 'torch.nn.Sigmoid'),
            ).to(device)
        else:
            model = model_class(
                input_dim=model_config.get('input_dim', 784),
                hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
                latent_dim=model_config.get('latent_dim', 10),
                output_activation=model_config.get('output_activation', 'torch.nn.Sigmoid'),
            ).to(device)
    else:
        # For other model types, pass all config parameters except 'model_type'
        model_params = {k: v for k, v in model_config.items() if k != 'model_type'}
        model = model_class(**model_params).to(device)
    
    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, model_config


def generate_random_latents(n_samples, latent_dim, mu=0.0, sigma=1.0, device='cpu'):
    """
    Generate random latent vectors from a normal distribution.
    
    Args:
        n_samples (int): Number of latent vectors to generate
        latent_dim (int): Dimension of the latent space
        mu (float): Mean of the normal distribution
        sigma (float): Standard deviation of the normal distribution
        device (str): Device to generate the latent vectors on
        
    Returns:
        torch.Tensor: Tensor of random latent vectors with shape (n_samples, latent_dim)
    """
    return torch.normal(mean=mu, std=sigma, size=(n_samples, latent_dim)).to(device)


def save_generated_images(images, output_dir, prefix='generated', image_size=(28, 28)):
    """
    Save generated images to disk.
    
    Args:
        images (torch.Tensor): Tensor of images with shape (n_samples, channels*height*width)
        output_dir (str): Directory to save the images
        prefix (str): Prefix for the image filenames
        image_size (tuple): Size of the images (height, width)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, img in enumerate(images):
        # Reshape and convert to numpy
        img_np = img.cpu().numpy().reshape(*image_size)
        
        # Create figure
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np, cmap='gray')
        plt.axis('off')
        
        # Save image
        filename = f"{prefix}_{timestamp}_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def create_grid_image(images, grid_size, output_dir, filename='grid.png', image_size=(28, 28)):
    """
    Create a grid of images and save it to disk.
    
    Args:
        images (torch.Tensor): Tensor of images with shape (n_samples, channels*height*width)
        grid_size (tuple): Size of the grid (rows, cols)
        output_dir (str): Directory to save the grid image
        filename (str): Filename for the grid image
        image_size (tuple): Size of each image (height, width)
    """
    rows, cols = grid_size
    n_images = min(rows * cols, images.size(0))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n_images:
                # Get image
                img = images[idx].cpu().numpy().reshape(*image_size)
                
                # Plot image
                if rows == 1 and cols == 1:
                    ax = axes
                elif rows == 1:
                    ax = axes[j]
                elif cols == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')
            else:
                # Hide empty subplots
                if rows == 1 and cols == 1:
                    ax = axes
                elif rows == 1:
                    ax = axes[j]
                elif cols == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                ax.axis('off')
    
    # Save grid image
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def main():
    """
    Main function to run the generation script.
    """
    parser = argparse.ArgumentParser(description='Generate random images from a trained autoencoder model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model safetensors file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model YAML config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for generated images')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--mu', type=float, default=0.0, help='Mean of the normal distribution')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation of the normal distribution')
    parser.add_argument('--grid', action='store_true', help='Create a grid image of all generated samples')
    parser.add_argument('--grid_rows', type=int, default=5, help='Number of rows in the grid')
    parser.add_argument('--grid_cols', type=int, default=5, help='Number of columns in the grid')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA')
    
    args = parser.parse_args()
    
    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path} with config {args.config_path}")
    model, model_config = load_model(args.model_path, args.config_path, device)
    
    # Get latent dimension
    latent_dim = model_config.get('latent_dim', 10)
    print(f"Model latent dimension: {latent_dim}")
    
    # Generate random latent vectors
    print(f"Generating {args.n_samples} random latent vectors with mu={args.mu}, sigma={args.sigma}")
    latents = generate_random_latents(args.n_samples, latent_dim, args.mu, args.sigma, device)
    
    # Generate images
    print("Generating images...")
    with torch.no_grad():
        # Check if the model is a ConvAutoEncoder
        if hasattr(model, 'decode'):
            # Use the decode method directly if available
            generated_images = model.decode(latents)
        else:
            # Otherwise, assume it's a standard autoencoder and use the decoder part
            generated_images = model.decoder(latents)
    
    # Print the shape of the generated images for debugging
    print(f"Generated images shape: {generated_images.shape}")
    
    # Determine image size and reshape if necessary
    if len(generated_images.shape) == 4:
        # Convolutional output: [batch_size, channels, height, width]
        batch_size, channels, height, width = generated_images.shape
        # Set image size for later use
        image_size = (height, width)
        print(f"Detected image size: {image_size} (from shape {generated_images.shape})")
        
        # Reshape to [batch_size, flattened_image] for consistent handling
        generated_images = generated_images.view(batch_size, channels * height * width)
        print(f"Reshaped to: {generated_images.shape}")
    else:
        # Fully connected output: [batch_size, flattened_image]
        output_size = generated_images.size(1)  # Get the number of elements in each output
        
        # Try to determine the image size
        if output_size == 784:
            # Standard MNIST size (28x28)
            image_size = (28, 28)
        else:
            # Calculate the side length assuming square images
            side_length = int(np.sqrt(output_size))
            if side_length * side_length == output_size:
                image_size = (side_length, side_length)
            else:
                # If not a perfect square, try to find appropriate dimensions
                # For now, reshape to the closest square
                side_length = int(np.sqrt(output_size))
                image_size = (side_length, output_size // side_length)
                print(f"Warning: Output size {output_size} is not a perfect square. Reshaping to {image_size}")
        
        print(f"Detected image size: {image_size} (from output size {output_size})")
    
    # Save individual images
    print(f"Saving generated images to {args.output_dir}")
    save_generated_images(generated_images, args.output_dir, image_size=image_size)
    
    # Create grid image if requested
    if args.grid:
        grid_size = (args.grid_rows, args.grid_cols)
        grid_filename = f"grid_mu{args.mu}_sigma{args.sigma}.png"
        print(f"Creating grid image with size {grid_size}")
        create_grid_image(generated_images, grid_size, args.output_dir, grid_filename, image_size)
    
    print("Done!")


if __name__ == "__main__":
    main()
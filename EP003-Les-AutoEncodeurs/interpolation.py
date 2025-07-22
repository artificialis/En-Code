"""
Interpolation script for autoencoder latent space.

This script takes two random MNIST samples from specified digit classes,
encodes them to get their latent representations, creates interpolations
between these representations, and then decodes them to generate images.

Usage:
    python interpolation.py --model_path <path_to_safetensors> --config_path <path_to_yaml> 
                           --digit1 <first_digit> --digit2 <second_digit> 
                           --steps <num_steps> --output_dir <output_directory>

Example:
    python interpolation.py --model_path ./models/conv_latent_dim04/autoencoder_final.safetensors 
                           --config_path ./models/conv_latent_dim04/autoencoder_final.yaml 
                           --digit1 3 --digit2 8 --steps 10 --output_dir ./interpolation_results
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from safetensors.torch import load_file
import importlib


def load_class(full_class_path):
    """
    Load a class from its full path.
    
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
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model class
    model_class = load_class(config['model_type'])
    
    # Initialize model based on its type
    if 'Autoencoder' in config['model_type']:
        if 'ConvAutoEncoder' in config['model_type']:
            model = model_class(
                input_channels=config.get('input_channels', 1),
                input_size=config.get('input_size', 28),
                hidden_channels=config.get('hidden_channels', [32, 64, 128]),
                latent_dim=config['latent_dim'],
                output_activation=config.get('output_activation', 'torch.nn.Sigmoid'),
            ).to(device)
        else:
            model = model_class(
                input_dim=config.get('input_dim', 784),
                hidden_dims=config.get('hidden_dims', [512, 256, 128]),
                latent_dim=config['latent_dim'],
                output_activation=config.get('output_activation', 'torch.nn.Sigmoid'),
            ).to(device)
    else:
        # For other model types, pass all config parameters except 'model_type'
        model_config = {k: v for k, v in config.items() if k != 'model_type'}
        model = model_class(**model_config).to(device)
    
    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def get_samples_by_digit(digit, dataset, num_samples=1):
    """
    Get random samples of a specific digit from the MNIST dataset.
    
    Args:
        digit (int): The digit class (0-9)
        dataset (Dataset): The MNIST dataset
        num_samples (int): Number of samples to retrieve
        
    Returns:
        list: List of (image, target) tuples
    """
    # Find all indices of the specified digit
    indices = [i for i, (_, target) in enumerate(dataset) if target == digit]
    
    # Randomly select indices
    selected_indices = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)
    
    # Get the corresponding samples
    samples = [dataset[i] for i in selected_indices]
    
    return samples


def interpolate_latent(latent1, latent2, steps):
    """
    Create interpolations between two latent vectors.
    
    Args:
        latent1 (torch.Tensor): First latent vector
        latent2 (torch.Tensor): Second latent vector
        steps (int): Number of interpolation steps
        
    Returns:
        torch.Tensor: Tensor of interpolated latent vectors with shape (steps, latent_dim)
    """
    # Create interpolation weights
    weights = torch.linspace(0, 1, steps).unsqueeze(1).to(latent1.device)
    
    # Interpolate
    interpolated = latent1 * (1 - weights) + latent2 * weights
    
    return interpolated


def main():
    """
    Main function to run the interpolation script.
    """
    parser = argparse.ArgumentParser(description='Autoencoder Latent Space Interpolation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model safetensors file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model YAML config file')
    parser.add_argument('--digit1', type=int, required=True, choices=range(10), help='First digit class (0-9)')
    parser.add_argument('--digit2', type=int, required=True, choices=range(10), help='Second digit class (0-9)')
    parser.add_argument('--steps', type=int, required=True, help='Number of interpolation steps')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for interpolated images')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA')
    
    args = parser.parse_args()
    
    # CUDA setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path} with config {args.config_path}")
    model = load_model(args.model_path, args.config_path, device)
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Get random samples for each digit
    print(f"Getting random samples for digits {args.digit1} and {args.digit2}")
    sample1 = get_samples_by_digit(args.digit1, test_dataset)[0][0].to(device)  # Get the image tensor
    sample2 = get_samples_by_digit(args.digit2, test_dataset)[0][0].to(device)  # Get the image tensor
    
    # Add batch dimension
    sample1 = sample1.unsqueeze(0)
    sample2 = sample2.unsqueeze(0)
    
    # Encode samples to get latent representations
    print("Encoding samples to latent space")
    with torch.no_grad():
        latent1 = model.encode(sample1)
        latent2 = model.encode(sample2)
    
    # Create interpolations
    print(f"Creating {args.steps} interpolations")
    interpolated_latents = interpolate_latent(latent1, latent2, args.steps)
    
    # Decode interpolated latents
    print("Decoding interpolated latent vectors")
    with torch.no_grad():
        interpolated_images = []
        for latent in interpolated_latents:
            # Add batch dimension
            latent = latent.unsqueeze(0)
            # Decode
            decoded = model.decode(latent)
            interpolated_images.append(decoded.squeeze(0).cpu())
    
    # Save original and interpolated images
    print(f"Saving images to {args.output_dir}")
    
    # Create a figure with all images
    plt.figure(figsize=(15, 3))
    
    # Plot first original image
    plt.subplot(1, args.steps + 2, 1)
    plt.imshow(sample1.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Original {args.digit1}")
    plt.axis('off')
    
    # Plot interpolated images
    for i, img in enumerate(interpolated_images):
        plt.subplot(1, args.steps + 2, i + 2)
        
        # Reshape if needed (for convolutional models)
        if len(img.shape) > 2:
            img = img.squeeze()
        
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f"Step {i+1}")
        plt.axis('off')
        
        # Also save individual images
        plt.figure(figsize=(5, 5))
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f"Interpolation Step {i+1}")
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, f"interp_{args.digit1}_to_{args.digit2}_step_{i+1}.png"))
        plt.close()
    
    # Plot second original image
    plt.subplot(1, args.steps + 2, args.steps + 2)
    plt.imshow(sample2.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Original {args.digit2}")
    plt.axis('off')
    
    # Save the combined figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"interpolation_{args.digit1}_to_{args.digit2}.png"))
    plt.close()
    
    print("Interpolation completed successfully!")


if __name__ == "__main__":
    main()
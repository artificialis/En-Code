#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract latent vectors from MNIST samples using a trained autoencoder model.

This script:
1. Loads a trained autoencoder model from safetensors and yaml files
2. Gets n samples from the MNIST dataset
3. Extracts the latent vector for each sample
4. Outputs the latent vectors

Usage:
    python extract.py --model /path/to/model --n 100
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import importlib
from safetensors.torch import load_file

# Import model classes
from model import Autoencoder, ConvAutoEncoder


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary with model parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_class(full_class_path):
    """
    Dynamically load a class from its full path.
    
    Args:
        full_class_path (str): Full path to the class (e.g., 'model.Autoencoder')
        
    Returns:
        class: The loaded class
    """
    module_name, class_name = full_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_model(model_path):
    """
    Load a model from safetensors and yaml files.
    
    Args:
        model_path (str): Path to the model files (without extension)
        
    Returns:
        torch.nn.Module: The loaded model
    """
    # Determine file paths
    safetensors_path = f"{model_path}.safetensors"
    yaml_path = f"{model_path}.yaml"
    
    # Check if files exist
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Model weights file not found: {safetensors_path}")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Model config file not found: {yaml_path}")
    
    # Load model configuration
    model_config = load_config(yaml_path)
    
    # Load model class
    model_class = load_class(model_config['model_type'])
    
    # Initialize model based on its type
    if model_class == Autoencoder:
        model = model_class(
            input_dim=model_config.get('input_dim', 784),  # 28x28 MNIST images
            hidden_dims=model_config['hidden_dims'],
            latent_dim=model_config['latent_dim'],
            output_activation=model_config.get('output_activation', 'torch.nn.Sigmoid')
        )
    elif model_class == ConvAutoEncoder:
        model = model_class(
            input_channels=model_config.get('input_channels', 1),  # MNIST has 1 channel
            input_size=model_config.get('input_size', 28),  # MNIST images are 28x28
            hidden_channels=model_config.get('hidden_channels', [32, 64, 128]),
            latent_dim=model_config['latent_dim'],
            output_activation=model_config.get('output_activation', 'torch.nn.Sigmoid')
        )
    else:
        raise ValueError(f"Unsupported model type: {model_config['model_type']}")
    
    # Load model weights
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def get_mnist_samples(n_samples, batch_size=128):
    """
    Get n samples from the MNIST test dataset.
    
    Args:
        n_samples (int): Number of samples to get
        batch_size (int): Batch size for the data loader
        
    Returns:
        tuple: (samples, labels) where samples is a tensor of shape [n_samples, 1, 28, 28]
        and labels is a tensor of shape [n_samples]
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the test dataset
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create a data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Get n samples
    samples = []
    labels = []
    sample_count = 0
    
    for data, target in test_loader:
        batch_size = data.size(0)
        if sample_count + batch_size <= n_samples:
            samples.append(data)
            labels.append(target)
            sample_count += batch_size
        else:
            # Get only the remaining samples needed
            remaining = n_samples - sample_count
            samples.append(data[:remaining])
            labels.append(target[:remaining])
            sample_count += remaining
        
        if sample_count >= n_samples:
            break
    
    # Concatenate all samples and labels
    samples = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return samples, labels


def extract_latent_vectors(model, samples):
    """
    Extract latent vectors from samples using the model's encoder.
    
    Args:
        model (torch.nn.Module): The autoencoder model
        samples (torch.Tensor): Input samples
        
    Returns:
        numpy.ndarray: Latent vectors
    """
    with torch.no_grad():
        # Use the model's encode method to get latent vectors
        latent_vectors = model.encode(samples)
    
    # Convert to numpy array
    return latent_vectors.cpu().numpy()


def main():
    """
    Main function to extract latent vectors from MNIST samples.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract latent vectors from MNIST samples')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the model files (without extension)')
    parser.add_argument('--n', type=int, default=100, 
                        help='Number of samples to process')
    parser.add_argument('--with-labels', action='store_true', default=False,
                        help='Include labels in the output (deprecated, labels are always included)')
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Get MNIST samples
    print(f"Getting {args.n} samples from MNIST...")
    samples, labels = get_mnist_samples(args.n)
    
    # Extract latent vectors
    print("Extracting latent vectors...")
    latent_vectors = extract_latent_vectors(model, samples)
    
    # Print some information about the latent vectors
    print(f"Extracted {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}")
    print(f"Latent vector statistics:")
    print(f"  Mean: {np.mean(latent_vectors, axis=0)}")
    print(f"  Std: {np.std(latent_vectors, axis=0)}")
    print(f"  Min: {np.min(latent_vectors, axis=0)}")
    print(f"  Max: {np.max(latent_vectors, axis=0)}")
    
    # Print all latent vectors with their digit class
    print("\nAll latent vectors with their digit class:")
    for i in range(latent_vectors.shape[0]):
        print(f"  Sample {i} (digit class {labels[i].item()}): {latent_vectors[i]}")


if __name__ == "__main__":
    main()
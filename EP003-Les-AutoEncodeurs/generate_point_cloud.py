#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a 3D point cloud visualization of the latent space of a trained autoencoder.

This script loads a trained autoencoder model (safetensors + yaml config), passes all test
samples through the encoder to extract latent vectors, applies t-SNE to reduce the latent
vectors to 3D, colors each point based on its class label, and exports the result as a
colored point cloud in PLY format.

Usage:
    python generate_point_cloud.py --model <path_to_safetensors_file> --config <path_to_yaml_config> [options]

Required Arguments:
    --model         Path to the .safetensors file (required)
    --config        Path to the YAML configuration file (required)

Optional Arguments:
    --perplexity    t-SNE perplexity parameter (default: 30)
    --output        Output .ply filename (default: latent_space.ply)
    --split         Dataset split to use (train or test, default: test)
    --device        PyTorch device to use (cpu or cuda, default: cuda if available)

Requirements:
    - PyTorch: For model loading and tensor operations
    - safetensors: For loading model weights
    - PyYAML: For parsing model configuration
    - torchvision: For loading the MNIST dataset
    - scikit-learn: For t-SNE dimensionality reduction
    - open3d (optional): For creating and exporting point clouds
      If open3d is not available, a fallback method will be used to create PLY files.

Example:
    python generate_point_cloud.py --model models/conv_latent_dim10/autoencoder_final.safetensors \\
                                  --config config/config_conv_latent_dim10.yaml \\
                                  --output mnist_latent_space.ply
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import importlib
from safetensors.torch import load_file
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from rich.console import Console
from rich.logging import RichHandler

# Initialize rich console
console = Console()

# Try to import open3d, provide installation instructions if not available
try:
    import open3d as o3d
except ImportError:
    console.print("[bold red]Error:[/bold red] open3d package not found. Please install it using:")
    console.print("pip install open3d")
    console.print("For more information, visit: http://www.open3d.org/docs/release/getting_started.html")
    o3d = None
# end try


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
# end def load_class


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
    # end with
    
    # Extract model configuration
    # Check if config has a 'model' section
    if 'model' in config and isinstance(config['model'], dict):
        model_config = config['model']
        model_type = model_config.get('model_type')
    else:
        # If no 'model' section, use the top-level config
        model_config = config
        model_type = model_config.get('model_type')
    # end if
    
    if not model_type:
        # Default model type if not specified
        model_type = 'model.Autoencoder'
    # end if
    
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
        # end if
    else:
        # For other model types, pass all config parameters except 'model_type'
        model_params = {k: v for k, v in model_config.items() if k != 'model_type'}
        model = model_class(**model_params).to(device)
    # end if
    
    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, model_config
# end def load_model


def load_dataset(split='test', batch_size=64, use_cuda=False):
    """
    Load the MNIST dataset.
    
    Args:
        split (str): Dataset split to use ('train' or 'test')
        batch_size (int): Number of samples per batch
        use_cuda (bool): Whether to use CUDA acceleration
        
    Returns:
        DataLoader: DataLoader for the specified dataset split
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # end transforms.Compose
    
    # Download and load the data
    dataset = datasets.MNIST(
        root='./data', 
        train=(split == 'train'), 
        download=True, 
        transform=transform
    )
    
    # Create data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle for latent space visualization
        **kwargs
    )
    
    return data_loader
# end def load_dataset


def extract_latent_vectors(model, data_loader, device):
    """
    Extract latent vectors and labels from the dataset using the encoder.
    
    Args:
        model (nn.Module): The autoencoder model
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to use for computation
        
    Returns:
        tuple: (latent_vectors, labels)
            - latent_vectors (numpy.ndarray): Extracted latent vectors
            - labels (numpy.ndarray): Corresponding class labels
    """
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Extracting latent vectors"):
            data = data.to(device)
            
            # Extract latent vectors using the encoder
            if hasattr(model, 'encode'):
                # Use the encode method if available
                latent = model.encode(data)
            else:
                # Otherwise, assume it's a standard autoencoder and use the encoder part
                latent = model.encoder(data)
            # end if
            
            # Move to CPU and convert to numpy
            latent_cpu = latent.cpu().numpy()
            target_cpu = target.cpu().numpy()
            
            latent_vectors.append(latent_cpu)
            labels.append(target_cpu)
        # end for
    # end with
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return latent_vectors, labels
# end def extract_latent_vectors


def apply_tsne(latent_vectors, perplexity=30):
    """
    Apply t-SNE dimensionality reduction to the latent vectors.
    
    Args:
        latent_vectors (numpy.ndarray): Latent vectors to reduce
        perplexity (float): t-SNE perplexity parameter
        
    Returns:
        numpy.ndarray: Reduced vectors in 3D space
    """
    console.print(f"Applying t-SNE with [bold cyan]perplexity={perplexity}[/bold cyan]...")
    try:
        # Try with max_iter (newer scikit-learn versions)
        tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=1000, random_state=42)
    except TypeError:
        # Fall back to n_iter (older scikit-learn versions)
        tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, random_state=42)
    # end try
    
    reduced_vectors = tsne.fit_transform(latent_vectors)
    
    return reduced_vectors
# end def apply_tsne


def get_colors_for_labels(labels):
    """
    Generate distinct RGB colors for each class label.
    
    Args:
        labels (numpy.ndarray): Class labels
        
    Returns:
        numpy.ndarray: RGB colors for each point
    """
    # Define a color map for MNIST digits (0-9)
    # Using a colorful palette for better visualization
    color_map = np.array([
        [1.0, 0.0, 0.0],  # Red (0)
        [0.0, 1.0, 0.0],  # Green (1)
        [0.0, 0.0, 1.0],  # Blue (2)
        [1.0, 1.0, 0.0],  # Yellow (3)
        [1.0, 0.0, 1.0],  # Magenta (4)
        [0.0, 1.0, 1.0],  # Cyan (5)
        [1.0, 0.5, 0.0],  # Orange (6)
        [0.5, 0.0, 1.0],  # Purple (7)
        [0.0, 0.5, 0.0],  # Dark Green (8)
        [0.5, 0.5, 0.5],  # Gray (9)
    ])
    
    # Map each label to its corresponding color
    colors = np.array([color_map[label] for label in labels])
    
    return colors
# end def get_colors_for_labels


def create_point_cloud(points, colors, output_file):
    """
    Create a colored point cloud and export it to a PLY file.
    
    Args:
        points (numpy.ndarray): 3D points
        colors (numpy.ndarray): RGB colors for each point
        output_file (str): Output PLY filename
    """
    if o3d is not None:
        # Use Open3D if available
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Export to PLY file
        o3d.io.write_point_cloud(output_file, pcd)
        console.print(f"Point cloud saved to [bold green]{output_file}[/bold green]")
    else:
        # Fallback method using NumPy to write PLY file
        console.print("[yellow]Using fallback method to save PLY file (open3d not available)[/yellow]")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
        
        # Combine points and colors
        num_points = points.shape[0]
        
        # Write PLY header
        with open(output_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float red\n")
            f.write("property float green\n")
            f.write("property float blue\n")
            f.write("end_header\n")
            
            # Write vertices with colors
            for i in range(num_points):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            # end for
        # end with
        
        console.print(f"Point cloud saved to [bold green]{output_file}[/bold green] using fallback method")
    # end if
# end def create_point_cloud


def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        bool: True if all required dependencies are available, False otherwise
    """
    missing_deps = []
    
    # Check required dependencies
    try:
        import torch
    except ImportError:
        missing_deps.append("PyTorch (pip install torch)")
    # end try
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("PyYAML (pip install pyyaml)")
    # end try
    
    try:
        import safetensors
    except ImportError:
        missing_deps.append("safetensors (pip install safetensors)")
    # end try
    
    try:
        import torchvision
    except ImportError:
        missing_deps.append("torchvision (pip install torchvision)")
    # end try
    
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        missing_deps.append("scikit-learn (pip install scikit-learn)")
    # end try
    
    # Report missing dependencies
    if missing_deps:
        console.print("[bold red]Error:[/bold red] Missing required dependencies:")
        for dep in missing_deps:
            console.print(f"  - [yellow]{dep}[/yellow]")
        console.print("\nPlease install these dependencies and try again.")
        return False
    # end if
    
    # Check optional dependencies
    if o3d is None:
        console.print("[bold yellow]Warning:[/bold yellow] open3d is not installed. A fallback method will be used to create PLY files.")
        console.print("For better visualization, install open3d: [cyan]pip install open3d[/cyan]")
        console.print("")
    # end if
    
    return True
# end def check_dependencies


def main():
    """
    Main function to run the point cloud generation script.
    """
    # Check dependencies
    if not check_dependencies():
        return
    # end if
    
    parser = argparse.ArgumentParser(description='Generate a 3D point cloud from autoencoder latent space')
    parser.add_argument('--model', type=str, required=True, help='Path to the .safetensors file')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--perplexity', type=float, default=30, help='t-SNE perplexity parameter')
    parser.add_argument('--output', type=str, default='latent_space.ply', help='Output .ply filename')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to use')
    parser.add_argument('--device', type=str, default=None, help='PyTorch device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # end if
    console.print(f"Using device: [bold cyan]{device}[/bold cyan]")
    
    # Load model
    console.print(f"Loading model from [bold]{args.model}[/bold] with config [bold]{args.config}[/bold]")
    model, model_config = load_model(args.model, args.config, device)
    
    # Load dataset
    console.print(f"Loading [bold]{args.split}[/bold] dataset")
    data_loader = load_dataset(split=args.split, use_cuda=(device.type == 'cuda'))
    
    # Extract latent vectors
    console.print("Extracting latent vectors...")
    latent_vectors, labels = extract_latent_vectors(model, data_loader, device)
    console.print(f"Extracted [bold green]{len(latent_vectors)}[/bold green] latent vectors with shape {latent_vectors.shape}")
    
    # Apply t-SNE
    reduced_vectors = apply_tsne(latent_vectors, perplexity=args.perplexity)
    console.print(f"Reduced vectors shape: {reduced_vectors.shape}")
    
    # Generate colors based on labels
    colors = get_colors_for_labels(labels)
    
    # Create and export point cloud
    create_point_cloud(reduced_vectors, colors, args.output)
# end def main


if __name__ == "__main__":
    main()
# end if
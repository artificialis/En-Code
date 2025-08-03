#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate 3D point cloud visualizations of the latent space of a trained autoencoder.

This script loads a trained autoencoder model (safetensors + yaml config), passes all test
samples through the encoder to extract latent vectors, applies t-SNE to reduce the latent
vectors to 3D, and generates separate point cloud files for each digit (0-9) while keeping
the coordinates related. The t-SNE is performed on the entire set of latent vectors, then
the points are split by digit class and exported to separate files.

The script supports two output formats:
1. PLY format: Points are represented as individual vertices with colors
2. OBJ format: Each point is represented as a small cube with the corresponding color

Usage:
    python generate_point_cloud.py --model <path_to_safetensors_file> --config <path_to_yaml_config> [options]

Required Arguments:
    --model         Path to the .safetensors file (required)
    --config        Path to the YAML configuration file (required)

Optional Arguments:
    --perplexity    t-SNE perplexity parameter (default: 30)
    --output-dir    Output directory for point cloud files (default: 'point_clouds')
    --combined      Optional filename for combined point cloud with all digits
    --split         Dataset split to use (train or test, default: test)
    --device        PyTorch device to use (cpu or cuda, default: cuda if available)
    --format        Output format (ply or obj, default: ply)
    --cube-size     Size of cubes for OBJ format (default: 0.05)

Requirements:
    - PyTorch: For model loading and tensor operations
    - safetensors: For loading model weights
    - PyYAML: For parsing model configuration
    - torchvision: For loading the MNIST dataset
    - scikit-learn: For t-SNE dimensionality reduction
    - open3d (optional): For creating and exporting PLY point clouds
      If open3d is not available, a fallback method will be used to create PLY files.

Examples:
    # Generate PLY point clouds (default)
    python generate_point_cloud.py --model models/conv_latent_dim10/autoencoder_final.safetensors \\
                                  --config config/config_conv_latent_dim10.yaml \\
                                  --output-dir mnist_point_clouds \\
                                  --combined all_digits.ply

    # Generate OBJ files with cubes
    python generate_point_cloud.py --model models/conv_latent_dim10/autoencoder_final.safetensors \\
                                  --config config/config_conv_latent_dim10.yaml \\
                                  --output-dir mnist_obj_clouds \\
                                  --format obj \\
                                  --cube-size 0.1 \\
                                  --combined all_digits.obj
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


def create_point_cloud(points, colors, output_file, format='ply', cube_size=0.05):
    """
    Create a colored point cloud and export it to a PLY or OBJ file.
    
    Args:
        points (numpy.ndarray): 3D points
        colors (numpy.ndarray): RGB colors for each point
        output_file (str): Output filename
        format (str): Output format ('ply' or 'obj')
        cube_size (float): Size of the cube for OBJ format
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    
    if format.lower() == 'obj':
        # Export as OBJ with cubes
        create_obj_with_cubes(points, colors, output_file, cube_size)
    else:
        # Export as PLY with points
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
    # end if
# end def create_point_cloud


def create_mtl_file(colors, mtl_file):
    """
    Create a MTL file with material definitions for each cube.
    
    Args:
        colors (numpy.ndarray): RGB colors for each point
        mtl_file (str): Output MTL filename
    """
    with open(mtl_file, 'w') as f:
        f.write("# MTL file with material definitions\n")
        f.write(f"# Generated by generate_point_cloud.py\n\n")
        
        # Create a material for each cube
        for i in range(len(colors)):
            color = colors[i]
            f.write(f"newmtl color_{i}\n")
            f.write(f"Ka {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")  # Ambient color
            f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")  # Diffuse color
            f.write(f"Ks 0.000000 0.000000 0.000000\n")  # Specular color
            f.write("d 1.000000\n")  # Opacity
            f.write("illum 1\n\n")  # Illumination model
    
    console.print(f"MTL file with {len(colors)} materials saved to [bold green]{mtl_file}[/bold green]")


def create_obj_with_cubes(points, colors, output_file, cube_size=0.05):
    """
    Create an OBJ file with colored cubes at each point.
    
    Args:
        points (numpy.ndarray): 3D points
        colors (numpy.ndarray): RGB colors for each point
        output_file (str): Output OBJ filename
        cube_size (float): Size of each cube
    """
    console.print(f"Creating OBJ file with cubes (size={cube_size})...")
    
    # Calculate half size for cube vertices
    half_size = cube_size / 2.0
    
    # Define the 8 vertices of a unit cube centered at origin
    cube_vertices = np.array([
        [-half_size, -half_size, -half_size],  # 0
        [half_size, -half_size, -half_size],   # 1
        [half_size, half_size, -half_size],    # 2
        [-half_size, half_size, -half_size],   # 3
        [-half_size, -half_size, half_size],   # 4
        [half_size, -half_size, half_size],    # 5
        [half_size, half_size, half_size],     # 6
        [-half_size, half_size, half_size]     # 7
    ])
    
    # Define the 6 faces of the cube (using 0-indexed vertices)
    # Each face is defined by 4 vertices in counter-clockwise order
    cube_faces = [
        [0, 1, 2, 3],  # Bottom face (-z)
        [4, 7, 6, 5],  # Top face (+z)
        [0, 4, 5, 1],  # Front face (-y)
        [1, 5, 6, 2],  # Right face (+x)
        [2, 6, 7, 3],  # Back face (+y)
        [3, 7, 4, 0]   # Left face (-x)
    ]
    
    # Create MTL file with the same base name as the OBJ file
    mtl_file = os.path.splitext(output_file)[0] + '.mtl'
    create_mtl_file(colors, mtl_file)
    
    with open(output_file, 'w') as f:
        f.write("# OBJ file with colored cubes\n")
        f.write(f"# Generated by generate_point_cloud.py\n\n")
        
        # Reference the MTL file
        mtl_filename = os.path.basename(mtl_file)
        f.write(f"mtllib {mtl_filename}\n\n")
        
        vertex_count = 0
        
        # For each point, create a cube
        for i in range(len(points)):
            point = points[i]
            color = colors[i]
            
            # Write material for this cube
            f.write(f"# Cube {i} with color {color}\n")
            f.write(f"usemtl color_{i}\n")
            f.write(f"# RGB: {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
            
            # Write the 8 vertices of this cube
            for v in cube_vertices:
                # Translate the vertex to the point position
                vertex = point + v
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
            
            # Write the 6 faces of this cube
            for face in cube_faces:
                # Adjust vertex indices for this cube
                # OBJ format uses 1-indexed vertices
                indices = [vertex_count + idx + 1 for idx in face]
                f.write(f"f {indices[0]} {indices[1]} {indices[2]} {indices[3]}\n")
            
            # Update vertex count for next cube
            vertex_count += 8
            
            # Add a blank line between cubes for readability
            f.write("\n")
    
    console.print(f"OBJ file with {len(points)} cubes saved to [bold green]{output_file}[/bold green]")


def split_by_digit(points, colors, labels):
    """
    Split points and colors by digit class.
    
    Args:
        points (numpy.ndarray): 3D points from t-SNE
        colors (numpy.ndarray): RGB colors for each point
        labels (numpy.ndarray): Digit labels (0-9)
        
    Returns:
        dict: Dictionary with digit labels as keys and tuples of (points, colors) as values
    """
    digit_data = {}
    
    # Process each digit (0-9)
    for digit in range(10):
        # Get indices for this digit
        indices = np.where(labels == digit)[0]
        
        # Extract points and colors for this digit
        digit_points = points[indices]
        digit_colors = colors[indices]
        
        # Store in dictionary
        digit_data[digit] = (digit_points, digit_colors)
        
        console.print(f"Digit [bold cyan]{digit}[/bold cyan]: {len(indices)} points")
    # end for
    
    return digit_data
# end def split_by_digit


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
    
    parser = argparse.ArgumentParser(description='Generate 3D point clouds from autoencoder latent space')
    parser.add_argument('--model', type=str, required=True, help='Path to the .safetensors file')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--perplexity', type=float, default=30, help='t-SNE perplexity parameter')
    parser.add_argument('--output-dir', type=str, default='point_clouds', help='Output directory for point cloud files')
    parser.add_argument('--combined', type=str, default=None, help='Optional filename for combined point cloud (all digits)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to use')
    parser.add_argument('--device', type=str, default=None, help='PyTorch device to use (cpu or cuda)')
    parser.add_argument('--format', type=str, default='ply', choices=['ply', 'obj'], help='Output format (ply or obj)')
    parser.add_argument('--cube-size', type=float, default=0.05, help='Size of cubes for OBJ format (default: 0.05)')
    
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
    
    # Apply t-SNE to all latent vectors together
    reduced_vectors = apply_tsne(latent_vectors, perplexity=args.perplexity)
    console.print(f"Reduced vectors shape: {reduced_vectors.shape}")
    
    # Generate colors based on labels
    colors = get_colors_for_labels(labels)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    console.print(f"Output directory: [bold cyan]{args.output_dir}[/bold cyan]")
    
    # Determine file extension based on format
    file_ext = '.obj' if args.format.lower() == 'obj' else '.ply'
    
    # Save combined point cloud if requested
    if args.combined:
        # Ensure the combined filename has the correct extension
        combined_filename = args.combined
        if not combined_filename.lower().endswith(file_ext):
            # Replace extension or add it if not present
            combined_filename = os.path.splitext(combined_filename)[0] + file_ext
        
        combined_output = os.path.join(args.output_dir, combined_filename)
        console.print(f"Creating combined point cloud: [bold]{combined_output}[/bold]")
        create_point_cloud(reduced_vectors, colors, combined_output, format=args.format, cube_size=args.cube_size)
    # end if
    
    # Split points by digit and create separate point clouds
    console.print("Splitting points by digit...")
    digit_data = split_by_digit(reduced_vectors, colors, labels)
    
    # Create a point cloud for each digit
    for digit, (digit_points, digit_colors) in digit_data.items():
        digit_output = os.path.join(args.output_dir, f"digit_{digit}{file_ext}")
        console.print(f"Creating point cloud for digit [bold cyan]{digit}[/bold cyan]: [bold]{digit_output}[/bold]")
        create_point_cloud(digit_points, digit_colors, digit_output, format=args.format, cube_size=args.cube_size)
    # end for
    
    console.print("[bold green]Done![/bold green] Created point clouds for all digits.")
# end def main


if __name__ == "__main__":
    main()
# end if
"""
Visualization script for the Autoencoder latent space.

This script provides various visualization tools for exploring the latent space
of a trained autoencoder model:
- 2D visualization of the latent space (if latent_dim=2)
- 3D visualization of the latent space (if latent_dim=3)
- Interactive plots using Plotly
- Rotating animations of 3D latent space
- Interpolation between different digits in the latent space

The script can be run from the command line with arguments for the model path,
configuration file, and visualization options.

Example:
    python visualize.py --model models/autoencoder_final.pt --config config.yaml --interpolate
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.express as px
import plotly.graph_objects as go
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Autoencoder

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary with model and visualization parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # end with
    return config

def get_test_loader(batch_size, use_cuda=False):
    """
    Create data loader for testing with the MNIST dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        use_cuda (bool): Whether to use CUDA acceleration and related optimizations
        
    Returns:
        DataLoader: DataLoader for the MNIST test dataset
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download and load the test data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        **kwargs
    )
    
    return test_loader

def encode_dataset(model, device, test_loader, n_samples=None):
    """
    Encode the test dataset into the latent space using the trained autoencoder.
    
    Args:
        model (Autoencoder): The trained autoencoder model
        device (torch.device): Device to use for encoding (CPU or CUDA)
        test_loader (DataLoader): DataLoader for the test dataset
        n_samples (int, optional): Maximum number of samples to encode. If None, encode all samples.
        
    Returns:
        tuple: (latent_vectors, labels)
            - latent_vectors (numpy.ndarray): Array of shape [n_samples, latent_dim] containing encoded vectors
            - labels (numpy.ndarray): Array of shape [n_samples] containing corresponding digit labels
    """
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Encoding dataset"):
            # Limit the number of samples if specified
            if n_samples is not None and len(latent_vectors) * test_loader.batch_size >= n_samples:
                break
            # end if
                
            data = data.to(device)
            
            # Encode data
            latent = model.encode(data)
            
            # Store latent vectors and labels
            latent_vectors.append(latent.cpu().numpy())
            labels.append(target.numpy())
        # end for
    # end with
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Limit to n_samples if specified
    if n_samples is not None:
        latent_vectors = latent_vectors[:n_samples]
        labels = labels[:n_samples]
    # end if
    
    return latent_vectors, labels

def plot_latent_space_2d(latent_vectors, labels, save_path, config):
    """
    Plot the latent space in 2D using both static (matplotlib) and interactive (plotly) visualizations.
    
    Args:
        latent_vectors (numpy.ndarray): Array of shape [n_samples, 2] containing encoded vectors
        labels (numpy.ndarray): Array of shape [n_samples] containing corresponding digit labels
        save_path (str): Path to save the static plot image
        config (dict): Configuration dictionary with visualization parameters
        
    Note:
        This function creates two files:
        - A static PNG image using matplotlib
        - An interactive HTML file using plotly
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_vectors[:, 0], 
        latent_vectors[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=config['visualization']['alpha'],
        s=config['visualization']['point_size']
    )
    plt.colorbar(scatter, label='Digit')
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig(save_path)
    plt.close()
    
    # Create interactive plot with Plotly
    fig = px.scatter(
        x=latent_vectors[:, 0], 
        y=latent_vectors[:, 1], 
        color=labels.astype(str),
        title='Latent Space Visualization (Interactive)',
        labels={'color': 'Digit', 'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2'},
        opacity=config['visualization']['alpha']
    )
    
    # Save as HTML for interactivity
    html_path = save_path.replace('.png', '.html')
    fig.write_html(html_path)
    
    print(f"2D plots saved to {save_path} and {html_path}")

def plot_latent_space_3d(latent_vectors, labels, save_path, config):
    """
    Plot the latent space in 3D using both static (matplotlib) and interactive (plotly) visualizations.
    
    Args:
        latent_vectors (numpy.ndarray): Array of shape [n_samples, 3] containing encoded vectors
        labels (numpy.ndarray): Array of shape [n_samples] containing corresponding digit labels
        save_path (str): Path to save the static plot image
        config (dict): Configuration dictionary with visualization parameters
        
    Note:
        This function creates two or three files depending on configuration:
        - A static PNG image using matplotlib
        - An interactive HTML file using plotly
        - Optionally, a GIF animation showing rotation of the 3D space
    """
    # Static 3D plot with matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        latent_vectors[:, 0], 
        latent_vectors[:, 1], 
        latent_vectors[:, 2], 
        c=labels, 
        cmap='tab10', 
        alpha=config['visualization']['alpha'],
        s=config['visualization']['point_size']
    )
    
    plt.colorbar(scatter, label='Digit')
    ax.set_title('Latent Space Visualization (3D)')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    plt.savefig(save_path)
    
    # Create interactive 3D plot with Plotly
    fig = px.scatter_3d(
        x=latent_vectors[:, 0], 
        y=latent_vectors[:, 1], 
        z=latent_vectors[:, 2], 
        color=labels.astype(str),
        title='Latent Space Visualization (Interactive 3D)',
        labels={'color': 'Digit', 'x': 'Dim 1', 'y': 'Dim 2', 'z': 'Dim 3'},
        opacity=config['visualization']['alpha']
    )
    
    # Save as HTML for interactivity
    html_path = save_path.replace('.png', '.html')
    fig.write_html(html_path)
    
    print(f"3D plots saved to {save_path} and {html_path}")
    
    # Create animation if requested
    if config['visualization']['create_animation']:
        create_rotating_animation(
            latent_vectors, 
            labels, 
            save_path.replace('.png', '.gif'), 
            config
        )
    # end if

def create_rotating_animation(latent_vectors, labels, save_path, config):
    """
    Create a rotating animation of the 3D latent space.
    
    This function creates a GIF animation that rotates the 3D view of the latent space,
    allowing for better visualization of the spatial relationships between different digits.
    
    Args:
        latent_vectors (numpy.ndarray): Array of shape [n_samples, 3] containing encoded vectors
        labels (numpy.ndarray): Array of shape [n_samples] containing corresponding digit labels
        save_path (str): Path to save the animation GIF file
        config (dict): Configuration dictionary with visualization parameters including:
            - n_frames: Number of frames in the animation
            - fps: Frames per second
            - dpi: Resolution of the saved animation
            - alpha: Transparency of points
            - point_size: Size of points in the scatter plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        latent_vectors[:, 0], 
        latent_vectors[:, 1], 
        latent_vectors[:, 2], 
        c=labels, 
        cmap='tab10', 
        alpha=config['visualization']['alpha'],
        s=config['visualization']['point_size']
    )
    
    plt.colorbar(scatter, label='Digit')
    ax.set_title('Latent Space Visualization (3D)')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    
    # Function to update the plot for animation
    def update(frame):
        ax.view_init(elev=30, azim=frame * (360 / config['visualization']['n_frames']))
        return [scatter]
    # end update
    
    # Create animation
    n_frames = config['visualization']['n_frames']
    fps = config['visualization']['fps']
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=n_frames, 
        interval=1000/fps, 
        blit=True
    )
    
    # Save animation
    anim.save(
        save_path, 
        writer='pillow', 
        fps=fps, 
        dpi=config['visualization']['dpi']
    )
    
    plt.close()
    print(f"Animation saved to {save_path}")

def interpolate_digits(model, device, test_loader, save_dir, config):
    """
    Interpolate between pairs of digits in the latent space.
    
    This function:
    1. Collects examples of each digit from the test dataset
    2. Encodes pairs of digits into the latent space
    3. Creates a linear interpolation between the latent representations
    4. Decodes the interpolated points back to images
    5. Visualizes the smooth transition between digits
    
    Args:
        model (Autoencoder): The trained autoencoder model
        device (torch.device): Device to use for encoding/decoding (CPU or CUDA)
        test_loader (DataLoader): DataLoader for the test dataset
        save_dir (str): Directory to save the interpolation images
        config (dict): Configuration dictionary with visualization parameters
    """
    model.eval()
    
    # Get some test samples
    all_data = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            all_data.append(data)
            all_labels.append(labels)
            if len(all_data) * test_loader.batch_size >= 1000:  # Collect enough samples
                break
            # end if
        # end for
    # end with
    
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Find examples of different digits
    digit_examples = {}
    for i in range(10):
        indices = (all_labels == i).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            digit_examples[i] = all_data[indices[0]]
        # end if
    # end for
    
    # Choose pairs to interpolate between
    pairs = [(1, 7), (3, 8), (4, 9), (0, 6)]
    n_steps = 10
    
    for start_digit, end_digit in pairs:
        if start_digit in digit_examples and end_digit in digit_examples:
            # Get latent representations
            start_img = digit_examples[start_digit].unsqueeze(0).to(device)
            end_img = digit_examples[end_digit].unsqueeze(0).to(device)
            
            start_latent = model.encode(start_img)
            end_latent = model.encode(end_img)
            
            # Create interpolations
            alphas = np.linspace(0, 1, n_steps)
            interpolated_images = []
            
            for alpha in alphas:
                # Interpolate in latent space
                interp_latent = (1 - alpha) * start_latent + alpha * end_latent
                # Decode
                interp_img = model.decode(interp_latent)
                interpolated_images.append(interp_img.cpu().numpy().reshape(28, 28))
            # end for
            
            # Plot
            plt.figure(figsize=(15, 3))
            for i, img in enumerate(interpolated_images):
                plt.subplot(1, n_steps, i + 1)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title(f"Digit {start_digit}")
                elif i == n_steps - 1:
                    plt.title(f"Digit {end_digit}")
                else:
                    plt.title(f"α={alphas[i]:.1f}")
                # end if
            # end for
            
            plt.suptitle(f"Interpolation between digits {start_digit} and {end_digit}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/interpolation_{start_digit}_to_{end_digit}.png")
            plt.close()
        # end if
    # end for
    
    print(f"Interpolation images saved to {save_dir}")

def main():
    """
    Main function to run the autoencoder latent space visualization.
    
    This function:
    1. Parses command line arguments
    2. Loads configuration from the specified YAML file
    3. Sets up CUDA if available
    4. Creates a data loader for the MNIST test dataset
    5. Loads the trained autoencoder model
    6. Encodes the test dataset into the latent space
    7. Visualizes the latent space in 2D or 3D based on latent dimension
    8. Optionally creates interpolation visualizations between digits
    
    The visualization type depends on the latent space dimension:
    - 2D: Creates a 2D scatter plot
    - 3D: Creates a 3D scatter plot and optionally a rotating animation
    """
    parser = argparse.ArgumentParser(description='Visualize Autoencoder Latent Space')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--interpolate', action='store_true', default=False, help='create interpolation visualizations')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # CUDA setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader
    test_loader = get_test_loader(
        batch_size=config['training']['batch_size'],
        use_cuda=use_cuda
    )
    
    # Create model and load weights
    latent_dim = config['model']['latent_dim']
    model = Autoencoder(
        input_dim=784,  # 28x28 MNIST images
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=latent_dim
    ).to(device)
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Model loaded from {args.model}")
    
    # Create results directory
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Encode dataset
    n_samples = config['visualization']['n_samples']
    latent_vectors, labels = encode_dataset(model, device, test_loader, n_samples)
    
    # Visualize latent space
    if latent_dim == 2:
        plot_latent_space_2d(
            latent_vectors, 
            labels, 
            f"{results_dir}/latent_space_2d.png", 
            config
        )
    elif latent_dim == 3:
        plot_latent_space_3d(
            latent_vectors, 
            labels, 
            f"{results_dir}/latent_space_3d.png", 
            config
        )
    else:
        print(f"Latent dimension is {latent_dim}, which is not suitable for direct visualization.")
        print("Consider using dimensionality reduction techniques like t-SNE or UMAP.")
    # end if
    
    # Create interpolation visualizations if requested
    if args.interpolate:
        interpolate_digits(model, device, test_loader, results_dir, config)
    # end if
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()
# end if
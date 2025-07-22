"""
Test script to verify that the visualization functions create the correct directories.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Autoencoder
from visualization import visualize_reconstruction, visualize_latent, visualize_latent_2d

def main():
    # Create a test directory
    test_dir = "./test_results"
    
    # Set up a simple model and data for testing
    device = torch.device("cpu")
    model = Autoencoder(input_dim=784, hidden_dims=[512, 256, 128], latent_dim=10, output_activation="torch.nn.Sigmoid")  # Simple autoencoder with 10-dim latent space
    model.to(device)
    
    # Create a simple test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test visualize_reconstruction
    print("Testing visualize_reconstruction...")
    visualize_reconstruction(model, device, test_loader, step=1, save_dir=test_dir)
    
    # Create some fake latent representations and labels for testing
    latents = torch.randn(100, 10)  # 100 samples, 10-dim latent space
    digits = np.random.randint(0, 10, size=100)  # Random digit labels
    
    # Test visualize_latent
    print("Testing visualize_latent...")
    visualize_latent(latents, digits, step=1, save_dir=test_dir)
    
    # Test visualize_latent_2d
    print("Testing visualize_latent_2d...")
    visualize_latent_2d(latents, digits, step=1, dims=[0, 1], save_dir=test_dir)
    
    # Check if directories were created
    expected_dirs = [
        os.path.join(test_dir, "reconstructions"),
        os.path.join(test_dir, "latent_tsne"),
        os.path.join(test_dir, "latent_2d")
    ]
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory created: {dir_path}")
            # Check if files were created
            files = os.listdir(dir_path)
            if files:
                print(f"  - Files in directory: {files}")
            else:
                print(f"  - No files found in directory")
        else:
            print(f"✗ Directory not created: {dir_path}")

if __name__ == "__main__":
    main()
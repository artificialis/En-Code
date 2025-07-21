"""
Test script for the ConvAutoEncoder model.

This script creates an instance of the ConvAutoEncoder model and tests it with a sample input
to verify that it can process 2D images as input and output 2D images.
"""

import torch
from model import ConvAutoEncoder

def test_conv_autoencoder():
    """
    Test the ConvAutoEncoder model with a sample input.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define model parameters
    input_channels = 1  # 1 for grayscale (MNIST)
    input_size = 28     # 28x28 for MNIST
    hidden_channels = [32, 64, 128]
    latent_dim = 10
    
    # Create model
    model = ConvAutoEncoder(
        input_channels=input_channels,
        input_size=input_size,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim
    )
    
    # Create a sample input (batch_size=4, channels=1, height=28, width=28)
    x = torch.randn(4, input_channels, input_size, input_size)
    
    # Forward pass
    reconstruction, latent = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Check that shapes match
    assert x.shape == reconstruction.shape, f"Input shape {x.shape} does not match reconstruction shape {reconstruction.shape}"
    assert latent.shape == (x.shape[0], latent_dim), f"Latent shape {latent.shape} does not match expected shape {(x.shape[0], latent_dim)}"
    
    print("All tests passed!")
    
    return model, x, reconstruction, latent

if __name__ == "__main__":
    model, x, reconstruction, latent = test_conv_autoencoder()
    
    # Additional test: check that the model can encode and decode separately
    encoded = model.encode(x)
    decoded = model.decode(encoded)
    
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")
    
    # Check that shapes match
    assert encoded.shape == latent.shape, f"Encoded shape {encoded.shape} does not match latent shape {latent.shape}"
    assert decoded.shape == reconstruction.shape, f"Decoded shape {decoded.shape} does not match reconstruction shape {reconstruction.shape}"
    
    print("Additional tests passed!")
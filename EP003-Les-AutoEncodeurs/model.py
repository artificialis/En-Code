"""
Autoencoder model implementation for dimensionality reduction and data reconstruction.

This module defines the architecture of an autoencoder neural network with configurable
dimensions for the hidden layers and latent space. The autoencoder consists of an encoder
that compresses input data into a lower-dimensional latent space, and a decoder that
reconstructs the original data from this latent representation.

The implementation is designed for PyTorch and includes three main classes:
- Encoder: Compresses input data into latent space
- Decoder: Reconstructs data from latent space
- Autoencoder: Combines encoder and decoder into a complete model
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder part of the autoencoder.
    Compresses the input data into a lower-dimensional latent space.
    
    The encoder consists of a series of fully connected layers with ReLU activations,
    progressively reducing the dimensionality until reaching the latent space size.
    
    Attributes:
        encoder (nn.Sequential): Sequential container of encoder layers
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=2):
        """
        Initialize the encoder network.
        
        Args:
            input_dim (int): Dimension of the input data (default: 784 for MNIST)
            hidden_dims (list): List of hidden layer dimensions (default: [512, 256, 128])
            latent_dim (int): Dimension of the latent space (default: 2)
        """
        super(Encoder, self).__init__()
        
        # Build encoder layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        # end for
        
        # Output layer (to latent space)
        layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Encoded representation in latent space of shape [batch_size, latent_dim]
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder part of the autoencoder.
    Reconstructs the input data from the latent space representation.
    
    The decoder consists of a series of fully connected layers with ReLU activations,
    progressively increasing the dimensionality from the latent space to the original
    input dimension. The final layer uses a Sigmoid activation to constrain output values
    between 0 and 1, suitable for image data like MNIST.
    
    Attributes:
        decoder (nn.Sequential): Sequential container of decoder layers
    """
    def __init__(self, latent_dim=2, hidden_dims=[128, 256, 512], output_dim=784):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim (int): Dimension of the latent space (default: 2)
            hidden_dims (list): List of hidden layer dimensions (default: [128, 256, 512])
            output_dim (int): Dimension of the output data (default: 784 for MNIST)
        """
        super(Decoder, self).__init__()
        
        # Build decoder layers
        layers = []
        
        # Input layer (from latent space)
        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        # end for
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid to get values between 0 and 1 for MNIST
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Latent space representation of shape [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed data of shape [batch_size, output_dim]
        """
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    Complete autoencoder model combining encoder and decoder.
    
    This class integrates the Encoder and Decoder components into a single model
    that can be trained end-to-end. It handles the data flow from input through
    the encoder to obtain a latent representation, and then through the decoder
    to reconstruct the original input.
    
    Attributes:
        encoder (Encoder): The encoder component
        decoder (Decoder): The decoder component
        latent_dim (int): Dimension of the latent space
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=2):
        """
        Initialize the autoencoder with encoder and decoder components.
        
        Args:
            input_dim (int): Dimension of the input data (default: 784 for MNIST)
            hidden_dims (list): List of hidden layer dimensions (default: [512, 256, 128])
            latent_dim (int): Dimension of the latent space (default: 2)
        """
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        # Reverse hidden_dims for decoder to create symmetric architecture
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, channels, height, width]
                              or [batch_size, input_dim] if already flattened
                              
        Returns:
            tuple: (reconstruction, latent)
                - reconstruction (torch.Tensor): Reconstructed data of shape [batch_size, input_dim]
                - latent (torch.Tensor): Latent space representation of shape [batch_size, latent_dim]
        """
        # Flatten the input if it's not already flattened
        batch_size = x.size(0)
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        # end if
            
        # Encode then decode
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        return reconstruction, latent
    
    def encode(self, x):
        """
        Encode input to latent space.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, channels, height, width]
                              or [batch_size, input_dim] if already flattened
                              
        Returns:
            torch.Tensor: Latent space representation of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        # end if
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode from latent space.
        
        Args:
            z (torch.Tensor): Latent space representation of shape [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed data of shape [batch_size, input_dim]
        """
        return self.decoder(z)
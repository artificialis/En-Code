"""
Autoencoder model implementation for dimensionality reduction and data reconstruction.

This module defines the architecture of an autoencoder neural network with configurable
dimensions for the hidden layers and latent space. The autoencoder consists of an encoder
that compresses input data into a lower-dimensional latent space, and a decoder that
reconstructs the original data from this latent representation.

The implementation is designed for PyTorch and includes the following main classes:
- Encoder: Compresses input data into latent space
- Decoder: Reconstructs data from latent space
- Autoencoder: Combines encoder and decoder into a complete model
- ConvAutoEncoder: Convolutional autoencoder that preserves spatial structure
"""

import torch
import torch.nn as nn
import importlib
import math


def load_class(full_class_path):
    module_name, class_name = full_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
# end load_class


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
    def __init__(
            self,
            latent_dim=2,
            hidden_dims=[128, 256, 512],
            output_dim=784,
            activation: str = "torch.nn.Sigmoid",
    ):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim (int): Dimension of the latent space (default: 2)
            hidden_dims (list): List of hidden layer dimensions (default: [128, 256, 512])
            output_dim (int): Dimension of the output space (default: 784)
            activation (str): Activation function (default: Sigmoid)
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

        # Output activation
        activation_cls = load_class(activation)
        layers.append(activation_cls())
        
        self.decoder = nn.Sequential(*layers)
    # end __init__
    
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
    def __init__(
            self,
            input_dim=784,
            hidden_dims=[512, 256, 128],
            latent_dim=2,
            output_activation: str = "torch.nn.Sigmoid",
    ):
        """
        Initialize the autoencoder with encoder and decoder components.
        
        Args:
            input_dim (int): Dimension of the input data (default: 784 for MNIST)
            hidden_dims (list): List of hidden layer dimensions (default: [512, 256, 128])
            latent_dim (int): Dimension of the latent space (default: 2)
            output_activation (nn.Module): Activation function
        """
        super(Autoencoder, self).__init__()

        # Encodeur
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        )

        # Decodeur
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=input_dim,
            activation=output_activation,
        )

        self.latent_dim = latent_dim
    # end __init__
    
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


class ConvEncoder(nn.Module):
    """
    Convolutional encoder model for image data.
    
    This class implements an encoder using convolutional layers (Conv2d) to compress
    input images into a lower-dimensional latent space while preserving spatial structure.
    
    Attributes:
        encoder_conv (nn.Sequential): Sequential container of encoder convolutional layers
        encoder_fc (nn.Sequential): Sequential container of encoder fully connected layers
        latent_dim (int): Dimension of the latent space
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        feature_channels (int): Number of channels in the feature map before flattening
        feature_height (int): Height of the feature map before flattening
        feature_width (int): Width of the feature map before flattening
    """
    def __init__(
            self,
            input_channels=1,  # 1 for grayscale (MNIST), 3 for RGB
            input_size=28,     # Input image size (28 for MNIST)
            hidden_channels=[32, 64, 128],  # Number of channels in hidden layers
            latent_dim=10,
    ):
        """
        Initialize the convolutional encoder.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            input_size (int): Size of the input images (assuming square images)
            hidden_channels (list): List of channel dimensions for hidden layers
            latent_dim (int): Dimension of the latent space
        """
        super(ConvEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        # Build encoder convolutional layers
        encoder_layers = []
        in_channels = input_channels
        
        for out_channels in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Create a dummy input to calculate the flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            dummy_output = self.encoder_conv(dummy_input)
            _, C, H, W = dummy_output.shape
            flattened_dim = C * H * W
        
        # Fully connected layers for latent space
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, latent_dim)
        )
        
        # Store dimensions for reshaping in decoder
        self.feature_channels = C
        self.feature_height = H
        self.feature_width = W
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width]
                              
        Returns:
            torch.Tensor: Latent space representation of shape [batch_size, latent_dim]
        """
        # Apply convolutional layers
        x = self.encoder_conv(x)
        
        # Flatten and project to latent space
        latent = self.encoder_fc(x)
        
        return latent


class ConvDecoder(nn.Module):
    """
    Convolutional decoder model for image data.
    
    This class implements a decoder using transposed convolutional layers (ConvTranspose2d)
    to reconstruct images from a latent space representation.
    
    Attributes:
        decoder_fc (nn.Sequential): Sequential container of decoder fully connected layers
        decoder_conv (nn.Sequential): Sequential container of decoder convolutional layers
        latent_dim (int): Dimension of the latent space
        input_channels (int): Number of input channels for the output image
    """
    def __init__(
            self,
            input_channels=1,  # 1 for grayscale (MNIST), 3 for RGB
            hidden_channels=[32, 64, 128],  # Number of channels in hidden layers
            latent_dim=10,
            feature_channels=128,
            feature_height=4,
            feature_width=4,
            output_activation: str = "torch.nn.Sigmoid",
    ):
        """
        Initialize the convolutional decoder.
        
        Args:
            input_channels (int): Number of input channels for the output image
            hidden_channels (list): List of channel dimensions for hidden layers
            latent_dim (int): Dimension of the latent space
            feature_channels (int): Number of channels in the feature map after projection
            feature_height (int): Height of the feature map after projection
            feature_width (int): Width of the feature map after projection
            output_activation (str): Activation function for the output layer
        """
        super(ConvDecoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.feature_channels = feature_channels
        self.feature_height = feature_height
        self.feature_width = feature_width
        
        flattened_dim = feature_channels * feature_height * feature_width
        
        # Fully connected layers from latent space
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flattened_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Build decoder convolutional layers
        decoder_layers = []
        in_channels = hidden_channels[-1]
        
        # Reverse the hidden channels for the decoder
        for out_channels in reversed(hidden_channels[:-1]):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        # Final layer to output the reconstructed image
        decoder_layers.extend([
            nn.ConvTranspose2d(in_channels, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        
        # Add output activation
        activation_cls = load_class(output_activation)
        decoder_layers.append(activation_cls())
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def forward(self, z, target_height=None, target_width=None):
        """
        Forward pass through the decoder.
        
        Args:
            z (torch.Tensor): Latent space representation of shape [batch_size, latent_dim]
            target_height (int, optional): Target height for the output image.
            target_width (int, optional): Target width for the output image.
            
        Returns:
            torch.Tensor: Reconstructed image of shape [batch_size, channels, height, width]
        """
        # Project from latent space and reshape
        batch_size = z.size(0)
        x = self.decoder_fc(z)
        x = x.view(batch_size, self.feature_channels, self.feature_height, self.feature_width)
        
        # Apply transposed convolutional layers
        reconstruction = self.decoder_conv(x)
        
        # Crop to match target dimensions if provided
        if target_height is not None and target_width is not None:
            if reconstruction.size(2) != target_height or reconstruction.size(3) != target_width:
                # Calculate cropping boundaries
                h_diff = reconstruction.size(2) - target_height
                w_diff = reconstruction.size(3) - target_width
                
                # Ensure we can crop (output must be larger than target)
                if h_diff >= 0 and w_diff >= 0:
                    h_start = h_diff // 2
                    w_start = w_diff // 2
                    
                    # Crop the reconstruction to match target dimensions
                    reconstruction = reconstruction[:, :, 
                                                h_start:h_start + target_height, 
                                                w_start:w_start + target_width]
        
        return reconstruction


class ConvAutoEncoder(nn.Module):
    """
    Convolutional autoencoder model for image data.
    
    This class implements an autoencoder using separate ConvEncoder and ConvDecoder modules.
    It preserves the spatial structure of the input images throughout the network.
    
    The encoder compresses the input images into a lower-dimensional latent space, and the
    decoder reconstructs the original images from this latent representation.
    
    Attributes:
        encoder (ConvEncoder): The encoder component
        decoder (ConvDecoder): The decoder component
        latent_dim (int): Dimension of the latent space
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
    """
    def __init__(
            self,
            input_channels=1,  # 1 for grayscale (MNIST), 3 for RGB
            input_size=28,     # Input image size (28 for MNIST)
            hidden_channels=[32, 64, 128],  # Number of channels in hidden layers
            latent_dim=10,
            output_activation: str = "torch.nn.Sigmoid",
    ):
        """
        Initialize the convolutional autoencoder with encoder and decoder components.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            input_size (int): Size of the input images (assuming square images)
            hidden_channels (list): List of channel dimensions for hidden layers
            latent_dim (int): Dimension of the latent space
            output_activation (str): Activation function for the output layer
        """
        super(ConvAutoEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        # Create the encoder
        self.encoder = ConvEncoder(
            input_channels=input_channels,
            input_size=input_size,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim
        )
        
        # Create the decoder
        self.decoder = ConvDecoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            feature_channels=self.encoder.feature_channels,
            feature_height=self.encoder.feature_height,
            feature_width=self.encoder.feature_width,
            output_activation=output_activation
        )
    
    def forward(self, x):
        """
        Forward pass through the convolutional autoencoder.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width]
                              
        Returns:
            tuple: (reconstruction, latent)
                - reconstruction (torch.Tensor): Reconstructed image of shape [batch_size, channels, height, width]
                - latent (torch.Tensor): Latent space representation of shape [batch_size, latent_dim]
        """
        # Store original input shape for later cropping
        self.original_height = x.size(2)
        self.original_width = x.size(3)
        
        # Encode
        latent = self.encode(x)
        
        # Decode with target dimensions matching the input
        reconstruction = self.decode(latent, target_height=self.original_height, target_width=self.original_width)
        
        return reconstruction, latent
    
    def encode(self, x):
        """
        Encode input images to latent space.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width]
                              
        Returns:
            torch.Tensor: Latent space representation of shape [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z, target_height=None, target_width=None):
        """
        Decode from latent space to reconstructed images.
        
        Args:
            z (torch.Tensor): Latent space representation of shape [batch_size, latent_dim]
            target_height (int, optional): Target height for the output image. If None, uses self.original_height if available.
            target_width (int, optional): Target width for the output image. If None, uses self.original_width if available.
            
        Returns:
            torch.Tensor: Reconstructed image of shape [batch_size, channels, height, width]
        """
        # Use stored original dimensions if available and no target dimensions provided
        if target_height is None and hasattr(self, 'original_height'):
            target_height = self.original_height
        if target_width is None and hasattr(self, 'original_width'):
            target_width = self.original_width
        
        return self.decoder(z, target_height=target_height, target_width=target_width)
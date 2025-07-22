# Visualization Script Documentation

The `visualization.py` script provides functions for visualizing autoencoder outputs and latent space representations.

## Overview

This script contains three main visualization functions:
1. `visualize_reconstruction`: Visualizes original and reconstructed images
2. `visualize_latent`: Visualizes the latent space using t-SNE dimensionality reduction
3. `visualize_latent_2d`: Visualizes the latent space by projecting to 2D using two specific dimensions

These functions are primarily used by the training script but can also be imported and used independently.

> **Note:** The example code in this documentation uses PyTorch's native model loading format, but if you're using models saved by the current version of the train.py script, they would be in SafeTensors format and would need to be loaded differently.

## Functions

### visualize_reconstruction

```python
visualize_reconstruction(model, device, test_loader, step, save_dir='./results', args=None, config=None)
```

Visualizes original and reconstructed images from the test dataset.

#### Parameters:
- `model` (Autoencoder): The trained autoencoder model
- `device` (torch.device): Device to use for inference (CPU or CUDA)
- `test_loader` (DataLoader): DataLoader for the test dataset
- `step` (int): Current step number (iteration or epoch, used in the saved filename)
- `save_dir` (str): Base directory where a 'reconstructions' subdirectory will be created to save the images
- `args` (argparse.Namespace, optional): Command line arguments
- `config` (dict, optional): Configuration dictionary

#### Output:
- Creates a figure with original images in the top row and their reconstructions in the bottom row
- Saves the figure to `{save_dir}/reconstructions/reconstruction_iter_{step}.png`
- If Weights & Biases is enabled, logs the images to wandb

### visualize_latent

```python
visualize_latent(latents, digits, step, save_dir='./results', args=None, config=None)
```

Visualizes the latent space using t-SNE projection.

#### Parameters:
- `latents` (torch.Tensor): Tensor of latent representations with shape (n, latent_size)
- `digits` (list): List of n numbers representing the digit of each latent representation
- `step` (int): Current step number (iteration or epoch, used in the saved filename)
- `save_dir` (str): Base directory where a 'latent_tsne' subdirectory will be created to save the images
- `args` (argparse.Namespace, optional): Command line arguments
- `config` (dict, optional): Configuration dictionary

#### Output:
- Creates a 2D projection of the latent representations using t-SNE
- Colors each point according to its digit class
- Saves the figure to `{save_dir}/latent_tsne/latent_tsne_iter_{step}.png`
- If Weights & Biases is enabled, logs the image to wandb

### visualize_latent_2d

```python
visualize_latent_2d(latents, digits, step, dims=[0, 1], save_dir='./results', args=None, config=None)
```

Visualizes the latent space by projecting to 2D using two specific dimensions.

#### Parameters:
- `latents` (torch.Tensor): Tensor of latent representations with shape (n, latent_size)
- `digits` (list): List of n numbers representing the digit of each latent representation
- `step` (int): Current step number (iteration or epoch, used in the saved filename)
- `dims` (list): List of two integers specifying which dimensions to use for projection
- `save_dir` (str): Base directory where a 'latent_2d' subdirectory will be created to save the images
- `args` (argparse.Namespace, optional): Command line arguments
- `config` (dict, optional): Configuration dictionary

#### Output:
- Creates a 2D projection of the latent representations by taking two specific dimensions
- Colors each point according to its digit class
- Saves the figure to `{save_dir}/latent_2d/latent_dims_{dim1}_{dim2}_iter_{step}.png`
- If Weights & Biases is enabled, logs the image to wandb

## Example Usage

Since this script provides functions rather than a standalone command-line interface, it's typically used by importing the functions into another script. Here's an example of how to use these functions:

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visualization import visualize_reconstruction, visualize_latent, visualize_latent_2d
from model import Autoencoder

# Load a trained model
model = Autoencoder(input_dim=784, hidden_dims=[512, 256], latent_dim=10, output_activation='sigmoid')
model.load_state_dict(torch.load('model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create a test data loader
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Visualize reconstructions
visualize_reconstruction(model, device, test_loader, step=0, save_dir='./results')

# Get latent representations for visualization
latents = []
digits = []
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        _, latent = model(data)
        latents.append(latent.cpu())
        digits.extend(target.tolist())
        if len(digits) >= 500:  # Limit to 500 samples
            break
    latents = torch.cat(latents, dim=0)[:500]
    digits = digits[:500]

# Visualize latent space using t-SNE
visualize_latent(latents, digits, step=0, save_dir='./results')

# Visualize latent space using specific dimensions
visualize_latent_2d(latents, digits, step=0, dims=[0, 1], save_dir='./results')
```

## Notes

- The visualization functions are designed to work with the MNIST dataset (28x28 grayscale images)
- For t-SNE visualization, a random seed is set for reproducibility
- The functions create subdirectories for each type of visualization
- Weights & Biases integration is optional and controlled by the `args` and `config` parameters
- The functions can be used during training to monitor progress or after training to analyze the model's performance
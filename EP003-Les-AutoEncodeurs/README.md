# MNIST Autoencoder Visualization

This project implements a simple autoencoder for the MNIST dataset, with visualization of the latent space. It's part of the "En-Code" series.

## Project Overview

The project builds a basic autoencoder to compress and reconstruct MNIST digits, and provides tools to visualize the latent space. The main components are:

- **Autoencoder Model**: A simple MLP-based autoencoder with configurable latent dimensions
- **Training Pipeline**: Complete training loop with validation and visualization
- **Visualization Tools**: 2D/3D visualization of the latent space with color-coded digits

## Features

- Train an autoencoder on MNIST with configurable architecture
- Visualize the latent space in 2D or 3D
- Generate interactive plots with Plotly
- Create rotating 3D animations of the latent space
- Interpolate between digits in the latent space
- Track experiments with Weights & Biases (wandb) integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/En-Code.git
cd En-Code
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Move to the directory
```bash
cd EP003-Les-AutoEncodeurs
```

## Usage

### Configuration

The project uses a configuration file (`config.yaml`) to set hyperparameters. You can modify this file to change:
- Model architecture (hidden dimensions, latent dimension)
- Training parameters (batch size, learning rate, etc.)
- Visualization settings
- Weights & Biases (wandb) settings (project name, entity, tags, etc.)

### Training

To train the autoencoder:

```bash
python train.py --config config.yaml
```

With Weights & Biases parameters:

```bash
python train.py --config config.yaml --wandb-project "my-project" --wandb-entity "my-username"
```

To disable Weights & Biases logging:

```bash
python train.py --config config.yaml --no-wandb
```

This will:
1. Load the MNIST dataset
2. Initialize Weights & Biases tracking (if enabled)
3. Train the autoencoder with the specified parameters
4. Save model checkpoints to the `models` directory
5. Generate reconstruction visualizations in the `results` directory
6. Log metrics, images, and model artifacts to Weights & Biases (if enabled)

### Visualization

To visualize the latent space:

```bash
python visualize.py --model models/autoencoder_final.pt --config config.yaml
```

For interpolation between digits:

```bash
python visualize.py --model models/autoencoder_final.pt --config config.yaml --interpolate
```

## Expected Outputs

The project generates several types of visualizations:

1. **Reconstruction Comparisons**: Original vs. reconstructed images during training
2. **Loss Curves**: Training and validation loss over epochs
3. **Latent Space Visualization**: 2D or 3D plots showing how digits cluster in the latent space
4. **Interactive Plots**: HTML files with interactive visualizations
5. **Animations**: Rotating 3D visualizations of the latent space (if enabled)
6. **Interpolations**: Smooth transitions between different digits (if requested)

## Project Structure

- `model.py`: Defines the autoencoder architecture
- `train.py`: Handles training and evaluation
- `visualize.py`: Generates latent space visualizations
- `config.yaml`: Configuration file for hyperparameters
- `requirements.txt`: Required dependencies

## License

See the LICENSE file for details.

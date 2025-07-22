# Train Script Documentation

The `train.py` script allows you to train autoencoder models on the MNIST dataset with various configurations.

## Overview

This script provides a complete training pipeline for autoencoders, including:
1. Loading configuration from a YAML file
2. Setting up data loaders for the MNIST dataset
3. Initializing the autoencoder model (standard or convolutional)
4. Training the model for a specified number of epochs
5. Visualizing reconstructions and the latent space during training
6. Saving model checkpoints and the final model
7. Plotting and saving loss curves
8. Optional integration with Weights & Biases for experiment tracking

> **Note:** This script is designed to save models in the SafeTensors format (`.safetensors`) along with corresponding YAML configuration files. However, the existing models in the repository are in PyTorch's native format (`.pt`). This suggests that either the script has been updated recently or there's a discrepancy between what the script is supposed to do and what it actually does.

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--config` | string | No | config.yaml | Path to config file |
| `--no-cuda` | flag | No | False | Disables CUDA training |
| `--wandb-project` | string | No | - | Weights & Biases project name (overrides config) |
| `--wandb-entity` | string | No | - | Weights & Biases entity/username (overrides config) |
| `--wandb-name` | string | No | - | Weights & Biases run name (overrides config) |
| `--wandb-mode` | string | No | - | Weights & Biases mode (online, offline, disabled) |
| `--no-wandb` | flag | No | False | Disable Weights & Biases logging |

## Configuration File

The training script relies on a YAML configuration file that specifies model architecture, training parameters, and paths. Here's an example structure:

```yaml
model:
  model_type: model.Autoencoder  # or model.ConvAutoEncoder
  latent_dim: 10
  hidden_dims: [512, 256, 128]  # For standard autoencoder
  hidden_channels: [32, 64, 128]  # For convolutional autoencoder
  output_activation: sigmoid

training:
  batch_size: 128
  epochs: 30
  learning_rate: 0.001
  weight_decay: 1e-5
  criterion: torch.nn.MSELoss
  vis_frequency: 5
  save_frequency: 10
  vis_threshold_iter: 1000
  vis_freq_before: 0
  vis_freq_after: 5

paths:
  model_dir: ./models/latent_dim10
  results_dir: ./results/latent_dim10

wandb:
  project: autoencoder-mnist
  entity: your_username
  name: autoencoder_latent10
  tags: [autoencoder, mnist]
  notes: Training autoencoder with latent dimension 10
  mode: online
  log_model: true
  log_freq: 1
```

## Examples

### Basic Usage

Train a model using the default configuration file:

```bash
python train.py
```

### Using a Custom Configuration File

Train a model using a custom configuration file:

```bash
python train.py --config config/config_conv_latent_dim10.yaml
```

### Disabling CUDA

Force the script to use CPU even if CUDA is available:

```bash
python train.py --config config/config.yaml --no-cuda
```

### Customizing Weights & Biases Integration

Train a model with custom Weights & Biases settings:

```bash
python train.py --config config/config.yaml \
                --wandb-project my-project \
                --wandb-entity my-username \
                --wandb-name my-experiment-name
```

### Disabling Weights & Biases

Train a model without logging to Weights & Biases:

```bash
python train.py --config config/config.yaml --no-wandb
```

## Output

The script generates:
1. Model checkpoints saved at the frequency specified in the config file
2. A final model saved at the end of training
3. Reconstruction visualizations showing original and reconstructed images
4. Latent space visualizations using t-SNE and direct 2D projections
5. A loss curve plot showing training and validation loss over epochs

All outputs are saved to the directories specified in the configuration file.

## Notes

- The script supports both standard fully-connected autoencoders and convolutional autoencoders
- Visualization frequency can be controlled through the configuration file
- The script uses the safetensors format for saving models, which is more secure than pickle-based formats
- For each saved model, a corresponding YAML file with model parameters is also created
- The script includes rich console output with progress bars and tables using the rich library
- Weights & Biases integration provides additional experiment tracking capabilities
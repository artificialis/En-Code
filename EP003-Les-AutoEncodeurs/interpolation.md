# Interpolation Script Documentation

The `interpolation.py` script allows you to perform latent space interpolation between two digits using a trained autoencoder model.

## Overview

This script loads a trained autoencoder model and performs interpolation by:
1. Loading random samples of two specified digits from the MNIST dataset
2. Encoding these samples to get their latent representations
3. Creating a series of interpolated points between these latent vectors
4. Decoding the interpolated latent vectors to generate images
5. Saving both individual interpolated images and a combined visualization

> **Note:** This script is designed to work with models saved in the SafeTensors format (`.safetensors`), but the models in the repository are currently in PyTorch's native format (`.pt`). You may need to convert the models to SafeTensors format or modify the script to load PyTorch models.

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model_path` | string | Yes | - | Path to the model safetensors file |
| `--config_path` | string | Yes | - | Path to the model YAML config file |
| `--digit1` | integer (0-9) | Yes | - | First digit class (0-9) |
| `--digit2` | integer (0-9) | Yes | - | Second digit class (0-9) |
| `--steps` | integer | Yes | - | Number of interpolation steps |
| `--output_dir` | string | Yes | - | Output directory for interpolated images |
| `--no-cuda` | flag | No | False | Disables CUDA (uses CPU instead) |

## Examples

### Basic Usage

Interpolate between digits 3 and 8 with 10 steps:

```bash
python interpolation.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                       --config_path models/latent_dim10/autoencoder_final.yaml \
                       --digit1 3 \
                       --digit2 8 \
                       --steps 10 \
                       --output_dir interpolation_test
```

### More Interpolation Steps

Interpolate between digits 0 and 9 with 20 steps for smoother transition:

```bash
python interpolation.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                       --config_path models/latent_dim10/autoencoder_final.yaml \
                       --digit1 0 \
                       --digit2 9 \
                       --steps 20 \
                       --output_dir interpolation_test
```

### Different Digit Pairs

Interpolate between digits 1 and 7:

```bash
python interpolation.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                       --config_path models/latent_dim10/autoencoder_final.yaml \
                       --digit1 1 \
                       --digit2 7 \
                       --steps 10 \
                       --output_dir interpolation_test
```

### Using CPU Instead of GPU

Force the script to use CPU even if CUDA is available:

```bash
python interpolation.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                       --config_path models/latent_dim10/autoencoder_final.yaml \
                       --digit1 4 \
                       --digit2 9 \
                       --steps 10 \
                       --output_dir interpolation_test \
                       --no-cuda
```

## Output

The script generates:
1. Individual images for each interpolation step, saved as PNG files in the specified output directory
2. A combined visualization showing the original digits and all interpolation steps in a single image

## Notes

- The script randomly selects samples of the specified digits from the MNIST test dataset
- Linear interpolation is performed in the latent space
- The quality of interpolation depends on how well the autoencoder has learned to structure its latent space
- For best results, use a model with a well-structured latent space (typically achieved with higher latent dimensions)
- Different digit pairs may produce more or less smooth interpolations depending on their similarity in the latent space
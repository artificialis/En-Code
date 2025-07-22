# Generate Script Documentation

The `generate.py` script allows you to generate random images from a trained autoencoder model by sampling from the latent space.

## Overview

This script loads a trained autoencoder model and generates random images by:
1. Sampling random vectors from the latent space (using a normal distribution)
2. Decoding these latent vectors to generate new images
3. Saving the generated images individually and optionally as a grid

> **Note:** This script is designed to work with models saved in the SafeTensors format (`.safetensors`), but the models in the repository are currently in PyTorch's native format (`.pt`). You may need to convert the models to SafeTensors format or modify the script to load PyTorch models.

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model_path` | string | Yes | - | Path to the model safetensors file |
| `--config_path` | string | Yes | - | Path to the model YAML config file |
| `--output_dir` | string | Yes | - | Output directory for generated images |
| `--n_samples` | integer | No | 10 | Number of samples to generate |
| `--mu` | float | No | 0.0 | Mean of the normal distribution |
| `--sigma` | float | No | 1.0 | Standard deviation of the normal distribution |
| `--grid` | flag | No | False | Create a grid image of all generated samples |
| `--grid_rows` | integer | No | 5 | Number of rows in the grid |
| `--grid_cols` | integer | No | 5 | Number of columns in the grid |
| `--no-cuda` | flag | No | False | Disables CUDA (uses CPU instead) |

## Examples

### Basic Usage

Generate 10 images using the default normal distribution parameters:

```bash
python generate.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                  --config_path models/latent_dim10/autoencoder_final.yaml \
                  --output_dir test_generated_images
```

### Generating More Samples

Generate 20 images:

```bash
python generate.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                  --config_path models/latent_dim10/autoencoder_final.yaml \
                  --output_dir test_generated_images \
                  --n_samples 20
```

### Adjusting the Distribution

Generate images with a different mean and standard deviation:

```bash
python generate.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                  --config_path models/latent_dim10/autoencoder_final.yaml \
                  --output_dir test_generated_images \
                  --mu 0.5 \
                  --sigma 0.8
```

### Creating a Grid

Generate 25 images and create a 5x5 grid:

```bash
python generate.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                  --config_path models/latent_dim10/autoencoder_final.yaml \
                  --output_dir test_generated_images \
                  --n_samples 25 \
                  --grid \
                  --grid_rows 5 \
                  --grid_cols 5
```

### Using CPU Instead of GPU

Force the script to use CPU even if CUDA is available:

```bash
python generate.py --model_path models/latent_dim10/autoencoder_final.safetensors \
                  --config_path models/latent_dim10/autoencoder_final.yaml \
                  --output_dir test_generated_images \
                  --no-cuda
```

## Output

The script generates:
1. Individual images saved as PNG files in the specified output directory
2. If the `--grid` flag is used, a grid image combining multiple generated samples

## Notes

- The script automatically detects the image size from the model output
- For convolutional autoencoders, the script handles the reshaping of the output
- The random seed is not fixed, so each run will generate different images
- The quality of generated images depends on how well the autoencoder was trained
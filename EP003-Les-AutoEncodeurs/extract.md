# Extract Latent Vectors from MNIST

This document describes how to use the `extract.py` script to extract latent vectors from MNIST samples using a trained autoencoder model.

## Overview

The `extract.py` script:
1. Loads a trained autoencoder model from safetensors and yaml files
2. Gets n samples from the MNIST dataset
3. Extracts the latent vector for each sample
4. Outputs the latent vectors to a file

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- safetensors
- numpy
- pyyaml

## Usage

Basic usage:

```bash
python extract.py --model /path/to/model --n 100
```

### Arguments

- `--model`: Path to the model files (without extension). The script will look for both `.safetensors` and `.yaml` files with this base name.
- `--n`: Number of samples to process (default: 100)
- `--output`: Output file path for latent vectors (default: latent_vectors.npy)
- `--with-labels`: Include labels in the output (optional flag)

### Examples

1. Extract latent vectors from 100 MNIST samples using a specific model:

```bash
python extract.py --model models/conv_latent_dim02/autoencoder_final --n 100
```

2. Extract latent vectors and save them with a custom filename:

```bash
python extract.py --model models/conv_latent_dim02/autoencoder_final --n 50 --output my_latent_vectors.npy
```

3. Extract latent vectors and include the corresponding labels:

```bash
python extract.py --model models/conv_latent_dim02/autoencoder_final --n 200 --with-labels --output mnist_vectors_with_labels.npz
```

## Output

By default, the script saves the latent vectors as a NumPy array in a `.npy` file. If the `--with-labels` flag is used, it saves both the latent vectors and their corresponding labels in a `.npz` file.

The script also prints:
- The number of extracted vectors and their dimension
- Basic statistics about the latent vectors (mean, standard deviation, min, max)
- The first 5 latent vectors with their corresponding labels

## Loading the Output

You can load the saved latent vectors in Python using NumPy:

```python
# For .npy files (without labels)
import numpy as np
latent_vectors = np.load('latent_vectors.npy')

# For .npz files (with labels)
data = np.load('latent_vectors.npz')
latent_vectors = data['latent_vectors']
labels = data['labels']
```
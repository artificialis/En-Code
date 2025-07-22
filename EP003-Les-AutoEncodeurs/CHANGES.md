# Iteration-Based Visualization Implementation

## Overview

This document describes the changes made to implement iteration-based visualization in the autoencoder training process. The visualization process now happens according to the iteration count rather than the epoch count, allowing visualizations to be generated in the middle of an epoch.

## Key Changes

### 1. Configuration Parameters

Added new configuration parameters in `config.yaml`:

```yaml
# Iteration-based visualization parameters
vis_threshold_iter: 1000      # Iteration threshold after which visualization frequency changes
vis_freq_before: 100          # Visualize every N iterations before threshold
vis_freq_after: 500           # Visualize every N iterations after threshold
```

These parameters control:
- `vis_threshold_iter`: The iteration threshold after which the visualization frequency changes
- `vis_freq_before`: How often to visualize before reaching the threshold (every N iterations)
- `vis_freq_after`: How often to visualize after reaching the threshold (every N iterations)

### 2. Iteration Counter

Added a global iteration counter in the main training loop that persists across epochs and increments with each batch processed.

### 3. Visualization at Iteration 0

Added automatic visualization at iteration 0 (before training starts) for comparison purposes.

### 4. Modified `train_epoch` Function

Updated the `train_epoch` function to:
- Accept and return the iteration counter
- Check if visualization should be done based on the iteration count
- Perform visualization inside the training loop if the conditions are met

### 5. Updated Visualization Functions

Modified the visualization functions in `visualization.py` to:
- Use "step" instead of "epoch" in function signatures and docstrings
- Update file naming to use iterations instead of epochs
- Update plot titles and wandb logging to reference iterations

## Usage

The visualization process now happens automatically at iteration 0 and then according to the specified frequencies:
- Before iteration `vis_threshold_iter`, visualization happens every `vis_freq_before` iterations
- After iteration `vis_threshold_iter`, visualization happens every `vis_freq_after` iterations

This allows for more frequent visualizations early in training when the model is changing rapidly, and less frequent visualizations later when changes are more gradual.

## Example

With the default configuration:
- Visualization at iteration 0 (before training)
- Visualization every 100 iterations until iteration 1000
- Visualization every 500 iterations after iteration 1000

This provides a more detailed view of the model's progress during training, especially in the early stages.
# Saved Models Directory

This directory contains trained transformer models and their associated metadata.

## Contents

- `shakespeare_transformer_model.pt` - Trained Shakespeare language model with architecture and training metadata

## File Format

Models are saved in PyTorch format (`.pt`) and include:
- Model weights and architecture parameters
- Training metadata (iterations, losses, accuracies, hyperparameters)
- Version information for compatibility

## Usage

Models can be loaded using the `Runner.load()` method:

```python
from runners import Runner

# Load a saved model
runner = Runner.load("saved/shakespeare_transformer_model.pt")

# Generate text
generated = runner.generate("To be, or not to be", encode, decode, max_new_tokens=50)
```

## Notes

- Models are automatically saved here during training
- This directory is gitignored to prevent large model files from being tracked
- Each model file contains complete information needed for reconstruction

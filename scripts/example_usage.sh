#!/bin/bash

# Example usage script for Shakespeare transformer training
# This script demonstrates different ways to run training with various hyperparameters

echo "=== Shakespeare Transformer Training Examples ==="
echo ""

# Example 1: Default parameters
echo "1. Running with default parameters:"
echo "   python shakespeare_word.py"
echo ""

# Example 2: Custom number of iterations
echo "2. Running with custom iterations:"
echo "   python shakespeare_word.py --iters 10000"
echo ""

# Example 3: Smaller model for faster training
echo "3. Running with smaller model (faster training):"
echo "   python shakespeare_word.py --n_heads 4 --n_layers 4 --embedding_size 64 --iters 2000"
echo ""

# Example 4: Larger model for better quality
echo "4. Running with larger model (better quality, slower training):"
echo "   python shakespeare_word.py --n_heads 16 --n_layers 12 --embedding_size 256 --iters 15000"
echo ""

# Example 5: Custom learning rate and batch size
echo "5. Running with custom learning rate and batch size:"
echo "   python shakespeare_word.py --lr 5e-5 --batch_size 32 --iters 8000"
echo ""

# Example 6: All custom parameters
echo "6. Running with all custom parameters:"
echo "   python shakespeare_word.py --iters 12000 --n_heads 8 --n_layers 8 --embedding_size 192 --block_size 100 --dropout 0.2 --batch_size 40 --lr 2e-4"
echo ""

echo "=== Available Command Line Arguments ==="
echo "  --iters, -i          Number of training iterations (default: 5000)"
echo "  --n_heads            Number of attention heads (default: 10)"
echo "  --n_layers           Number of transformer layers (default: 10)"
echo "  --embedding_size     Embedding dimension (default: 128)"
echo "  --block_size         Context block size (default: 80)"
echo "  --dropout            Dropout rate (default: 0.1)"
echo "  --batch_size         Training batch size (default: 50)"
echo "  --lr                 Learning rate (default: 1e-4)"
echo ""

echo "=== Tips ==="
echo "- Use smaller models (fewer heads/layers) for faster experimentation"
echo "- Increase embedding_size and block_size for better text understanding"
echo "- Adjust batch_size based on your GPU memory"
echo "- Lower learning rate for more stable training, higher for faster convergence"
echo ""

echo "=== From scripts/ directory ==="
echo "If running from the scripts/ directory, use:"
echo "  cd .. && python shakespeare_word.py [options]"
echo ""

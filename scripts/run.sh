#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=ShakespeareWord
#SBATCH --array=0-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3976@nyu.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Example: Run with custom hyperparameters
# You can modify these arguments as needed:
# --n_heads: Number of attention heads (default: 10)
# --n_layers: Number of transformer layers (default: 10)
# --embedding_size: Embedding dimension (default: 128)
# --block_size: Context block size (default: 80)
# --dropout: Dropout rate (default: 0.1)
# --batch_size: Training batch size (default: 50)
# --lr: Learning rate (default: 1e-4)
# --iters: Number of training iterations (default: 5000)

# Run the Shakespeare training script with custom parameters
singularity exec --nv \
    --overlay /scratch/hl3976/singularity/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd .. && python shakespeare_word.py --iters 50000 --n_heads 10 --n_layers 10 --embedding_size 256 --batch_size 64 --lr 1e-3"

# Alternative: Run with default parameters
# singularity exec --nv \
#     --overlay /scratch/hl3976/singularity/overlay-50G-10M.ext3:ro \
#     /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
#     /bin/bash -c "source /ext3/env.sh; cd .. && python shakespeare_word.py"
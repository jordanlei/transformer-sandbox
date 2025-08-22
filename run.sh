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

# Run the Shakespeare training script
singularity exec --nv \
    --overlay /scratch/hl3976/singularity/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python script_shakespeare.py"
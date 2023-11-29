#!/bin/bash

#SBATCH --job-name=transcribe
#SBATCH --output=%x.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=34
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project

srun python transcribe.py --index 1 --max_index 1 --n 500 --batch_size 10 --lang ell

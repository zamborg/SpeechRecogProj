#!/bin/bash

#SBATCH --job-name=transcribe
#SBATCH --output=%x.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00
# #SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate project

./download.sh

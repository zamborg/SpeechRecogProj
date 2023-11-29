#!/bin/bash

#SBATCH --job-name=download
#SBATCH --output=%x_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32

source ~/miniconda3/etc/profile.d/conda.sh
conda activate project

./download.sh

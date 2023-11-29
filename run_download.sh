#!/bin/bash

#SBATCH --job-name=download
#SBATCH --output=%x.out
#SBATCH --partition=RM-shared
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate project

./download.sh

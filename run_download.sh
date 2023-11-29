#!/bin/bash

#SBATCH --job-name=download
#SBATCH --output=%x.out
#SBATCH --partition=RM-shared
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12 # 12 tasks with 1 cpu each meaning 12 workers with 1cpu
#SBATCH --cpus-per-task=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate project

./download.sh

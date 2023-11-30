#!/bin/bash

#SBATCH --job-name=transcribe
#SBATCH --output=%x.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
# source ~/project/miniconda3/etc/profile.d/conda.sh
source /jet/home/zaysola/project/etc/profile.d/conda.sh
conda activate project

export HF_DATASETS_CACHE="../HFCACHE/"

# srun python transcribe.py --index 1 --max_index 1 --n -1 --batch_size 20 --lang ell
srun python transcribe.py --index 1 --max_index 1 --n -1 --batch_size 20 --lang mlt

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=gpujob.out
#SBATCH --gres=gpu:v100:1
#SBATCH --error=gpujob.err

module purge
module load nvidia-hpc-sdk

# ./hetero_miner 24 4 8 1 1 512 512 64
./hetero_miner 16 2 16 1 1 128 128 4

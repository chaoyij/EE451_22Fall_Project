#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=hetero_job.out
#SBATCH --gres=gpu:v100:1
#SBATCH --error=hetero_job.err

module purge
module load nvidia-hpc-sdk

./hetero_miner 24 2 8 1 1 512 512 64

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=cpujob_pthread.out
#SBATCH --gres=gpu:v100:1
#SBATCH --error=cpujob_pthread.err

module purge
module load nvidia-hpc-sdk

./pthread_cpu_miner 24 2 8

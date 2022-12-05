#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=serial_job.out
#SBATCH --gres=gpu:v100:1
#SBATCH --error=serial_job.err

module purge
module load nvidia-hpc-sdk

./cpu_miner 24 2 8

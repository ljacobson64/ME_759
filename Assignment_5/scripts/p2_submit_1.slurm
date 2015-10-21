#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jacobson/p2_%J.out
#SBATCH --error=/home/jacobson/p2_%J.err

export CUDA_PROFILE=1
../bld/p2 0 4096
cat cuda_profile_0.log
rm -f cuda_profile_0.log

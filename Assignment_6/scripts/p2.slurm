#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jacobson/p2_%J.out
#SBATCH --error=/home/jacobson/p2_%J.err

declare -a N_arr=(16 32 64 128 256 512 1024 2048 4096 8192 13904)
time=1.0

for N in "${N_arr[@]}"; do
  ../bld/p2 $N $time
done

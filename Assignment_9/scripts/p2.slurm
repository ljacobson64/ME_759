#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jacobson/p2_%J.out
#SBATCH --error=/home/jacobson/p2_%J.err

declare -a N_arr=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
                  65536 131072 262144 524288 1048576 2097152 4194304 8388608 \
                  16777216 33554432 67108864 134217728 191758336)
M=5
dur=1

for N in "${N_arr[@]}"; do
  ../bld/p2 $N $M $dur
done

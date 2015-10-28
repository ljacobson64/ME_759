#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jacobson/p1_%J.out
#SBATCH --error=/home/jacobson/p1_%J.err

for ((i = 3; i <= 8; i++)); do
  ../bld/p1 4 $i
done
for ((i = 3; i <= 8; i++)); do
  ../bld/p1 5 $i
done
for ((i = 3; i <= 8; i++)); do
  ../bld/p1 6 $i
done

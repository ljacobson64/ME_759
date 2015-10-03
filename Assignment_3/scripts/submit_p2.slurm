#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/$USER/%J.out
#SBATCH --error=/home/$USER/%J.err

for i in $(seq -f "%02g" 5 20); do
  ../bld/p2 $i 32
done
for i in $(seq -f "%02g" 10 25); do
  ../bld/p2 $i 1024
done

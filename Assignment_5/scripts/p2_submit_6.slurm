#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jacobson/p2_%J.out
#SBATCH --error=/home/jacobson/p2_%J.err

for i in $(seq -f "%2g" 4 11); do
  ../bld/p2 1 $((2**$i))
done
../bld p2 0 4096
../bld p2 0 8192
../bld p2 0 11360

#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=/home/jacobson/p3_1_8_%J.out
#SBATCH --error=/home/jacobson/p3_1_8_%J.err

mpiexec -np 8 ../bld/p3

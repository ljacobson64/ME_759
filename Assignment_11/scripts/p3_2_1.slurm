#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/jacobson/p3_2_1_%J.out
#SBATCH --error=/home/jacobson/p3_2_1_%J.err

mpiexec -np 2 ../bld/p3

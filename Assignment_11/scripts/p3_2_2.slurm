#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=/home/jacobson/p3_2_2_%J.out
#SBATCH --error=/home/jacobson/p3_2_2_%J.err

mpiexec -np 4 ../bld/p3

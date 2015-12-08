#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --output=/home/jacobson/p1_%J.out
#SBATCH --error=/home/jacobson/p1_%J.err

mpiexec -np 16 ../bld/p1

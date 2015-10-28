#!/bin/bash

#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jacobson/p1_%J.out
#SBATCH --error=/home/jacobson/p1_%J.err

declare -a N_arr=(      100       316 \
                       1000      3162 \
                      10000     31623 \
                     100000    316228 \
                    1000000   3162278 \
                   10000000  31622777 \
                  100000000 193134592)
time=1.0

for N in "${N_arr[@]}"; do
  ../bld/p1 4 $N $time
done
for N in "${N_arr[@]}"; do
  ../bld/p1 5 $N $time
done
for N in "${N_arr[@]}"; do
  ../bld/p1 6 $N $time
done

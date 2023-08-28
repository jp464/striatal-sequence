#!/bin/bash

#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 20G

srun --preserve-env --multi-prog ./learning.conf

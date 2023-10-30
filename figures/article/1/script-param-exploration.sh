#!/bin/bash
#SBATCH --job-name=param-exploration
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --array=1-43201
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$jp464@duke.edu  

srun $(head -n $SLURM_ARRAY_TASK_ID jobs-param-exploration.txt | tail -n 1)

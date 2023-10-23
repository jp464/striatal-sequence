#!/bin/bash
#SBATCH --job-name=param-exploration
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=15G
#SBATCH --array=1-1859
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$jp464@duke.edu  

srun $(head -n $SLURM_ARRAY_TASK_ID jobs-param-exploration.txt | tail -n 1)

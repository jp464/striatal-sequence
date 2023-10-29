#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --output=R-%x/%j.out
#SBATCH --error=R-%x/%j.err
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$jp464@duke.edu  

srun $(head -n $SLURM_ARRAY_TASK_ID jobs-learning.txt | tail -n 1)

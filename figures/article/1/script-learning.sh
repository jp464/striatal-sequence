#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=05:00:00
#SBATCH --array=1-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jp464@duke.edu  
#SBATCH --output=R-%x/%j.out
#SBATCH --error=R-%x/%j.err


srun $(head -n $SLURM_ARRAY_TASK_ID jobs-learning.txt | tail -n 1)
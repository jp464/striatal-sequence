#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x/%j.err
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$jp464@duke.edu  

srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)

#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH -e slurm.err
#SBATCH -array=0-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$jp464@duke.edu  



readarray -t FILES < learning.txt
FILENAME=${FILES[${SLURM_ARRAY_TASK_ID}]}
python learning.py $FILENAME



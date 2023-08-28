#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH -a 1-4
#SBATCH -e slurm.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$jp464@duke.edu  



readarray -t FILES < learning.txt
FILENAME=${FILES[(($SLURM_ARRAY_TASK_ID - 1))]}
python learning.py $FILENAME



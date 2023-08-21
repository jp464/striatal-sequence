#!/bin/bash
#SBATCH -p brunellab-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=learning
#SBATCH --output=learning.out
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH -a 1-4
#SBATCH -e slurm.err

readarray -t FILES < learning.txt
FILENAME=${FILES[(($SLURM_ARRAY_TASK_ID - 1))]}
python learning.py $FILENAME



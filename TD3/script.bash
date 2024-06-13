#!/bin/bash

#SBATCH --job-name=iLL_240613
#SBATCH --account=Project_2006500
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN #uncomment to enable mail
#SBATCH --mail-type=FAIL #uncomment to enable mail
#SBATCH --mail-type=END #uncomment to enable mail
#SBATCH --gres=gpu:v100:1

export PATH="/users/mshokrne/envs/conda/ActorCritic/bin:$PATH"

python3 -u main.py > output.txt
#python test.py > output1.txt
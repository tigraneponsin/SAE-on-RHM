#!/bin/bash

#SBATCH --job-name=ipython
#SBATCH --partition h100
#SBATCH --time 2:00:00
#SBATCH --mem 90G
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
#SBATCH --account pcsl
#SBATCH --output jupyter-log-%J.out

###### #SBATCH --begin=2025-12-17T10:00:00

cd ..
 
ipnport=$(shuf -i8000-9999 -n1)

jupyter lab --no-browser --port=${ipnport} --ip=$(hostname -i)

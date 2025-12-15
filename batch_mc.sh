#!/bin/bash

#SBATCH --job-name=MC_soccer_datagen_5
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6Gb
#SBATCH --time=06:00:00
#SBATCH --account=bckiedro0
#SBATCH --export=ALL
#SBATCH --output=xg_datagen_5.out

srun --cpu-bind=cores python Soccer.py

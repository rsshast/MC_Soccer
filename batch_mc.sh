#!/bin/bash

#SBATCH --job-name=MC_soccer
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16Gb
#SBATCH --time=00:30:00
#SBATCH --account=bckiedro0
#SBATCH --export=ALL
#SBATCH --output=game_state.out

srun --cpu-bind=cores python Hockey.py

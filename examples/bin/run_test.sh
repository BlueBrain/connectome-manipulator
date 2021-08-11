#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate
python -u test.py

# EXAMPLE HOW TO RUN: sbatch run_test.sh
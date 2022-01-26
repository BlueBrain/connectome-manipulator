#!/bin/bash -l
#SBATCH --job-name=sscx_extract
#SBATCH --time=12:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

module purge
module load archive/2021-07
source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate
python -u connectome_manipulator_SSCx_extract.py

# EXAMPLE HOW TO RUN: sbatch run_SSCx_extract.sh
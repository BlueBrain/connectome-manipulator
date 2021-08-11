#!/bin/bash -l
#SBATCH --job-name=VPMsub10
#SBATCH --time=8:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate
python -u connectome_manipulator_SSCxVPMsub10.py

# EXAMPLE HOW TO RUN: sbatch run_VPMsub10.sh
#!/bin/bash -l
#SBATCH --job-name=model_build
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

module purge
module load archive/2021-07
source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate
python -u /gpfs/bbp.cscs.ch/home/pokorny/JupyterLab/git/connectome_manipulator/connectome_manipulator/model_building/model_building.py $1 $2 $3

# EXAMPLE HOW TO RUN: sbatch run_model_building.sh <model_config.json> [force_reextract] [force_rebuild]
# e.g. sbatch run_model_building.sh model_config.json 1 1
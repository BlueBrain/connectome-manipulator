#!/bin/bash -l
#SBATCH --job-name=struct_comp
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

module purge
module load unstable
source /gpfs/bbp.cscs.ch/home/pokorny/ReWiringKernel/bin/activate
connectome-manipulator compare-connectomes $1 $2 $3

# EXAMPLE HOW TO RUN: sbatch run_struct_comparison.sh <model_config.json> [--force-recomp-circ1] [--force-recomp-circ2]
# e.g. sbatch run_struct_comparison.sh structcomp_config.json --force-recomp-circ1 --force-recomp-circ2

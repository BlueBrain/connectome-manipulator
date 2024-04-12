#!/bin/bash -l
#SBATCH --job-name=conn_rewire
#SBATCH --partition=prod
#SBATCH --nodes=5
#SBATCH --tasks-per-node=18
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --out=logs/%j.txt
#SBATCH --err=logs/%j.txt

module load archive/2023-07 parquet-converters/0.8.0 py-mpi4py
source /gpfs/bbp.cscs.ch/home/pokorny/ReWiringKernel/bin/activate

srun dplace parallel-manipulator -v manipulate-connectome $1 --output-dir=$2 --parallel --profile --convert-to-sonata --splits=$3

# EXAMPLE HOW TO RUN: sbatch run_rewiring_parallel.sh <model_config.json> <output_dir> <num_splits>
# e.g. sbatch run_rewiring.sh manip_config.json /gpfs/.../O1v5-SONATA__Rewired 100
#
# IMPORTANT: Don't launch from within another SLURM allocation!!

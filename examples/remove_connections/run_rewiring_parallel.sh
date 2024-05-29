#!/bin/bash -l
#SBATCH --job-name=conn_rewire
#SBATCH --partition=prod
#SBATCH --nodes=10
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --account=proj142
#SBATCH --out=logs/%j.txt
#SBATCH --err=logs/%j.txt

rm -r output/*

module purge
module load unstable parquet-converters  py-mpi4py
#module load unstable parquet-converters
source /gpfs/bbp.cscs.ch/project/proj112/home/kurban/christoph_paper/github/venv_3_10_8/bin/activate

srun dplace parallel-manipulator -v manipulate-connectome $1 --output-dir=$2 --parallel --profile --convert-to-sonata --splits=$3

# EXAMPLE HOW TO RUN: sbatch run_rewiring_parallel.sh <model_config.json> <output_dir> <num_splits>
# e.g. sbatch run_rewiring.sh manip_config.json /gpfs/.../O1v5-SONATA__Rewired 100
#
# IMPORTANT: Don't launch from within another SLURM allocation!!

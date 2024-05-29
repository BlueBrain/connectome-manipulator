#!/bin/bash -l
#SBATCH --job-name=conn_rewire
#SBATCH --partition=prod
#SBATCH --nodes=64
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --account=proj112
#SBATCH --out=logs/%j.txt
#SBATCH --err=logs/%j.txt

. /etc/profile.d/modules.sh
unset MODULEPATH
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/config/modules.sh
module purge
module load archive/2023-07 python-dev parquet-converters/0.8.0 py-mpi4py
# module load py-connectome-manipulator
#source /gpfs/bbp.cscs.ch/home/pokorny/ReWiringKernel/bin/activate
source /gpfs/bbp.cscs.ch/project/proj112/home/kurban/christoph_paper/github/venv_3_10_8/bin/activate

set -x

srun dplace parallel-manipulator -v manipulate-connectome $1 --output-dir=$2 --parallel --profile --convert-to-sonata --splits=$3

# EXAMPLE HOW TO RUN: sbatch run_rewiring_parallel.sh <model_config.json> <output_dir> <num_splits>
# e.g. sbatch run_rewiring.sh manip_config.json /gpfs/.../O1v5-SONATA__Rewired 100
#
# sbatch run_rewiring_parallel_working_ex.sh manip_config.json /gpfs/bbp.cscs.ch/project/proj112/home/kurban/christoph_paper/github/connectome-manipulator/examples/remove_connections/output/ 100
#
# IMPORTANT: Don't launch from within another SLURM allocation!!

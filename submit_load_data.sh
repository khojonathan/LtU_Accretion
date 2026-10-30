#!/bin/bash
#SBATCH -J load_data
#SBATCH -A torrey-group
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=7500
#SBATCH --time=1-00:00:00
#SBATCH -p standard
#SBATCH --output=logs/loading.out
#SBATCH --error=logs/loading.err
#SBATCH --mail-user=yja6qa@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Batch shells don't load conda's shell function, so set it up explicitly
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kho_env || exit 1

# One thread per worker; load_data_updated.py runs SLURM_CPUS_PER_TASK worker processes
export OMP_NUM_THREADS=1

python load_data_updated.py

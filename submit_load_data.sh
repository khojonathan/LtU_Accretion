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

# Call kho_env's Python directly: conda activate in batch jobs can leave the
# miniforge module's base Python ahead of the env on PATH
PYTHON=$HOME/.conda/envs/kho_env/bin/python

# One thread per worker; load_data_updated.py runs SLURM_CPUS_PER_TASK worker processes
export OMP_NUM_THREADS=1

$PYTHON load_data_updated.py

#!/bin/bash
# Generic pull-based worker (see worker.py). Submit many short ones; each lands on whichever partition frees first.
#   sbatch --array=1-40 slurm_worker.sbatch
#   for p in gpuA100x4-interactive gpuA40x4-interactive; do sbatch -p $p slurm_worker.sbatch; done
#SBATCH --job-name=Topo_worker
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8,gpuA100x4-preempt,gpuA40x4-preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
unset PYTHONPATH; export PYTHONNOUSERSITE=1
T=${TOPO_HOME:?set TOPO_HOME to the folder that contains code/ and venv/}; export TOPO_ROOT=$T; cd $T/code; source $T/venv/bin/activate
export TOPO_BUDGET_MIN=${TOPO_BUDGET_MIN:-55} TOPO_SLOTS=${TOPO_SLOTS:-3}
python worker.py

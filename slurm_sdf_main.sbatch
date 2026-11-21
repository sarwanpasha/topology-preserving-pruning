#!/bin/bash
#SBATCH --job-name=Topo_sdf_main
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-34
#SBATCH --output=logs/%x_%A_%a.out
module load anaconda; export PYTHONNOUSERSITE=1   # adjust module for your cluster
T=${TOPO_HOME:?set TOPO_HOME to the folder that contains code/ and venv/}; export TOPO_ROOT=$T; cd $T/code; source $T/venv/bin/activate
TAG=${TAG:-main}
SH=(torus thin_torus double_torus triple_torus hopf_link holey_cube sphere)
i=$SLURM_ARRAY_TASK_ID; s=$((i % 5)); sh=${SH[$((i / 5))]}
echo "task $i shape=$sh seed=$s"
python run_sdf.py --shape $sh --seed $s --dense_only --tag $TAG
for m in random mag imp taylor synflow mag_topo imp_topo taylor_topo ts ts_iter ts_fisher ts_taskft synflow_topo alloc_mag_topo ts_alloc; do
  done_n=$(grep -c "\"seed\": $s," $T/results/$TAG/sdf_$sh/$m.jsonl 2>/dev/null || echo 0)
  if [ "$done_n" -ge 5 ]; then echo "skip $m"; continue; fi
  python run_sdf.py --shape $sh --seed $s --methods $m --tag $TAG &
done; wait
echo TASK_DONE

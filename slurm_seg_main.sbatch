#!/bin/bash
#SBATCH --job-name=Topo_seg_main
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
module load anaconda; export PYTHONNOUSERSITE=1   # adjust module for your cluster
T=${TOPO_HOME:?set TOPO_HOME to the folder that contains code/ and venv/}; export TOPO_ROOT=$T; cd $T/code; source $T/venv/bin/activate
TAG=${TAG:-main}
DS=(isbi stare roads); M="mag imp random taylor synflow mag_cl mag_topo imp_topo taylor_topo ts ts_iter ts_fisher ts_taskft"
i=$SLURM_ARRAY_TASK_ID; s=$((i % 5)); d=${DS[$((i / 5))]}
echo "task $i ds=$d seed=$s"
python run_seg.py --ds $d --seed $s --dense_only --workers 8 --tag $TAG
for m in $M; do
  done_n=$(grep -c "\"seed\": $s," $T/results/$TAG/$d/$m.jsonl 2>/dev/null || echo 0)
  if [ "$done_n" -ge 5 ]; then echo "skip $m (done)"; continue; fi
  python run_seg.py --ds $d --seed $s --methods $m --workers 2 --tag $TAG &
done; wait
echo TASK_DONE

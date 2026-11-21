#!/bin/bash
#SBATCH --job-name=Topo_setup_smoke
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
T=${TOPO_HOME:?set TOPO_HOME to the folder that contains code/ and venv/}; export TOPO_ROOT=$T; cd $T/code
module load anaconda  # adjust for your cluster
[ -x $T/venv/bin/python ] || python -m venv $T/venv
source $T/venv/bin/activate; export PYTHONNOUSERSITE=1
python -c "import torch,gudhi,skimage; assert torch.cuda.is_available()" 2>/dev/null || {
  pip install -q --upgrade pip; pip install -q numpy scipy gudhi scikit-image pillow torch
  python -c "import torch; assert torch.cuda.is_available()" || pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126; }
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import torch,gudhi,skimage; print('VERS', torch.__version__, torch.cuda.is_available(), gudhi.__version__, skimage.__version__)"
python -c "import common; print('GUDHI_ORDER', common.check_gudhi_order())"
python prep_data.py; ls -la $T/data
M="random mag imp taylor synflow mag_cl mag_topo imp_topo taylor_topo ts ts_iter ts_fisher ts_taskft"
python run_seg.py --ds isbi --seed 0 --smoke --dense_only --workers 8 --tag smoke
for m in $M; do python run_seg.py --ds isbi --seed 0 --smoke --methods $m --sparsities 0.9 --workers 2 --tag smoke & done
for d in stare roads; do python run_seg.py --ds $d --seed 0 --smoke --methods mag,ts --sparsities 0.9 --workers 2 --tag smoke & done
wait
for s in torus thin_torus double_torus hopf_link holey_cube triple_torus sphere; do
  python -c "import run_sdf as R; sh=R.Shape('$s','cpu',0); print('$s', 'gt64', sh.beta, 'gt40', sh.beta_topo)" &
done; wait
python run_sdf.py --shape thin_torus --seed 0 --smoke --dense_only --tag smoke
for m in mag imp taylor synflow mag_topo ts ts_iter ts_fisher random; do
  python run_sdf.py --shape thin_torus --seed 0 --smoke --methods $m --sparsities 0.95 --tag smoke &
done; wait
python agg.py smoke
n1=$(cat $T/results/smoke/isbi/*.jsonl | grep -vc '"method": "dense"'); n2=$(cat $T/results/smoke/sdf_thin_torus/*.jsonl | grep -vc '"method": "dense"')
echo "SMOKE_ROWS seg=$n1 sdf=$n2"
[ "$n1" -ge 13 ] && [ "$n2" -ge 9 ] && [ -f $T/data/roads.npz ] && [ -f $T/data/stare.npz ] || { echo SMOKE_FAILED; exit 1; }
echo SMOKE_DONE

# Topology-preserving neural network compression

Research code for compressing neural networks whose **outputs must be topologically correct**: connected
vessels and neuron membranes, unbroken road networks, 3D shapes with the right number of handles.
Standard pruning criteria (weight magnitude, IMP, Taylor/Fisher saliency, SynFlow) keep pixel accuracy high
while the topology of the prediction collapses. This repository contains:

* **TopoSaliency (`ts`, `ts_iter`, `ts_fisher`)**: pruning scores that add an empirical-Fisher
  (OBD-style) *topological* sensitivity term to magnitude / task saliency. The topological term is the
  squared gradient of a persistent-homology loss on the network output, so it is supported only on the
  critical pixels/voxels that create or destroy topological features.
* **Topological fine-tuning losses**
  * 2D (segmentation): persistence loss on the superlevel filtration of the probability map
    (Hu et al., NeurIPS 2019 matching to ground-truth Betti numbers), computed on CPU worker pools.
  * 3D (neural implicit shapes / SDFs): a *zero-level margin loss*. Only persistence pairs that can change
    the topology of `{f <= 0}` are used; the `beta_k` most persistent ones must straddle `[-c, c]`, all other
    relevant pairs are pushed out of the band.
* **Benchmark harness** with 13 methods (random, magnitude, IMP, Taylor, SynFlow, magnitude+clDice,
  magnitude/IMP/Taylor + topological fine-tuning, and the TopoSaliency variants), 5 sparsities
  (80, 90, 95, 97, 99 %), 5 seeds, equal fine-tuning budgets.
* **Applications**
  * Neuron membranes in electron microscopy (ISBI 2012), retinal vessels (STARE), road networks
    (Massachusetts Roads): U-Net segmentation.
  * Neural signed-distance fields of shapes with known genus (torus, thin torus, double/triple torus,
    Hopf link, cube with tunnels, sphere).
* **Metrics**: Dice, clDice, Betti-0/Betti-1 errors (64x64 patches and whole image), topological fidelity
  to the dense model, and for SDFs exact Betti numbers of the zero level set, IoU and surface error.

Status: experiments are running; see `RESULTS.md` for the numbers available so far. Nothing here should be
cited as a final result yet.

## Install

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                      # pick the torch build that matches your CUDA driver
```

Optional (Windows, integrated GPUs): a second environment with `torch-directml` lets `orchestrate.py`
add a DirectML worker (`--dml_workers 1`).

## Run

```bash
export TOPO_ROOT=$PWD/../work            # data, models, results, logs go here
python prep_data.py                      # downloads ISBI12, STARE, Massachusetts Roads (resumable)
python run_seg.py  --ds isbi  --seed 0 --methods mag,ts --workers 4
python run_sdf.py  --shape torus --seed 0 --methods mag,ts
python agg.py main                       # mean ± std tables -> results/main_summary.csv
```

* Slurm clusters: `slurm_setup_smoke.sbatch` (environment, data, smoke test of every method),
  `slurm_seg_main.sbatch` and `slurm_sdf_main.sbatch` (job arrays; set `--account` and `TOPO_HOME`).
* Single workstation: `python orchestrate.py --gpu_workers 1 --cpu_workers 1` runs the whole sweep with a
  GPU pool and a CPU pool in parallel and resumes after interruption (`run_all_windows.cmd` on Windows).

## Files

| file | content |
|---|---|
| `common.py` | data pipeline, U-Net, losses (task, clDice, persistence), Betti metrics, masks, saliency scores, SynFlow |
| `run_seg.py` | one (dataset, seed, method) run over all sparsities; trains/caches the dense model |
| `run_sdf.py` | neural-SDF application: analytic shapes, SDF MLP, 3D topological margin loss, evaluation |
| `prep_data.py` | robust, resumable dataset download and preprocessing |
| `orchestrate.py` | multi-device scheduler (CUDA / CPU / DirectML workers) with resume |
| `agg.py` | aggregation into mean ± std tables |

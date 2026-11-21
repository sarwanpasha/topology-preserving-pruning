# Results so far (pilot, updated as runs finish)

**Status (2026-11-21):** one seed (seed 0) on one dataset is partially complete. These are pilot numbers,
not paper results. TopoSaliency (our method) has not finished on any dataset yet.

## ISBI 2012 neuron membranes, U-Net, seed 0, 99 % sparsity

| method | Dice | clDice | Betti-0 error | Betti-1 error |
|---|---|---|---|---|
| dense (0 %) | 0.818 | 0.939 | 0.42 | 0.21 |
| magnitude | 0.731 | 0.828 | 4.32 | 1.84 |
| IMP | 0.653 | 0.704 | 12.17 | 6.54 |
| magnitude + clDice fine-tuning | 0.737 | 0.848 | 2.99 | 0.99 |
| magnitude + persistence fine-tuning | 0.714 | 0.824 | 3.15 | 1.15 |
| Taylor (Fisher) | 0.769 | 0.882 | 1.95 | 0.54 |
| SynFlow | 0.762 | 0.891 | 0.61 | 0.22 |

Observation: at high sparsity, topology fails much faster than Dice (magnitude pruning: Dice -11 %,
Betti-0 error x10). SynFlow is the strongest baseline so far and is the bar our method must clear.

## Neural SDF, torus, seed 0

Dense SDF reproduces the torus exactly (Betti (1,1,0), IoU 0.995). Magnitude pruning keeps the topology up to
97 % sparsity and loses it at 99 %. Earlier topological fine-tuning runs used a 3D loss that acted on
far-field persistence pairs and destroyed the shape; that loss was replaced by the zero-level margin loss
and those runs were discarded.

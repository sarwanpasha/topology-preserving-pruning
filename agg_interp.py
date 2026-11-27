"""Summarise results/interp/*.json: causal-ablation table + where the protected weights live."""
import json, glob, os, sys, numpy as np
R = os.environ.get('TOPO_ROOT', '..'); files = sorted(f for f in glob.glob(f'{R}/results/interp/*.json') if not os.path.basename(f).startswith('sdf_'))
by = {}
for f in files:
    d = json.load(open(f))
    for sp, r in d['sp'].items(): by.setdefault((d['ds'], sp), []).append((d, r))
def m(v): v = np.asarray(v, float); return f'{v.mean():.3f}' + (f'+-{v.std():.3f}' if len(v) > 1 else '')
for (ds, sp), L in sorted(by.items()):
    d0 = L[0][0]; tot = sum(d0['numel'])
    print(f'== {ds} sp={sp} seeds={len(L)} dense dice={m([d["dense"]["dice"] for d, _ in L])}  |A|/N={m([sum(r["nA"]) / tot for _, r in L])}'
          f'  |w|A/B={m([r["mag_mean_A"] / r["mag_mean_B"] for _, r in L])}  |w_eff|A/B={m([r["eff_mean_A"] / r["eff_mean_B"] for _, r in L])}'
          f'  Fisher-mass A/B={m([r["topo_mass_A"] / max(r["topo_mass_B"], 1e-30) for _, r in L])}')
    for k in ('A_protected', 'B_counterpart', 'R_random', 'A_vs_eff'):
        if k not in L[0][1]['abl']: continue
        a = [r['abl'][k] for _, r in L]
        print(f'  {k:14s} dice={m([x["dice"] for x in a])} b0={m([x["b0"] for x in a])} fid_b0={m([x["fid_b0"] for x in a])} '
              f'dp_crit={m([x["dp_crit"] for x in a])} dp_all={m([x["dp_all"] for x in a])} crit/all={m([x["dp_crit"] / x["dp_all"] for x in a])} '
              f'flip_crit={m([x["flip_crit"] for x in a])} flip_all={m([x["flip_all"] for x in a])}')
    # where do protected weights live: share of A per block
    blocks = {}
    for (d, r) in L:
        nA = np.asarray(r['nA'], float); s = nA.sum() or 1
        for nm, x in zip(d['layers'], nA):
            b = nm.split('.')[0] + nm.split('.')[1]; blocks.setdefault(b, []).append(x / s)
    nb = len(L)
    print('  share of A by block: ' + ' '.join(f'{b}={np.sum(v) / nb:.2f}' for b, v in blocks.items()))
    print('  rho(topoFisher, |w|) by layer: ' + ' '.join(f'{x:.2f}' for x in np.mean([d['rho_topo_mag'] for d, _ in L], 0)))

# ----------------------------------------------------------------------------- 3D shapes (interp_sdf.py)
by3 = {}
for f in sorted(glob.glob(f'{R}/results/interp/sdf_*.json')):
    d = json.load(open(f))
    for sp, r in d['sp'].items(): by3.setdefault(sp, []).append((d, r))
for sp, L in sorted(by3.items()):
    print(f'== SDF shapes sp={sp}: {len(L)} (shape,seed) runs; |w|A/B={m([r["mag_mean_A"] / r["mag_mean_B"] for _, r in L])}')
    for k in ('A_protected', 'B_counterpart', 'R_random'):
        a = [r['abl'][k] for _, r in L]
        print(f'  {k:14s} keeps dense Betti={m([x["fid_correct"] for x in a])} iou={m([x["iou"] for x in a])} '
              f'dp_crit={m([x["dp_crit"] for x in a])} dp_near={m([x["dp_near"] for x in a])} crit/near={m([x["dp_crit"] / x["dp_near"] for x in a])} '
              f'flip_crit={m([x["flip_crit"] for x in a])} flip_near={m([x["flip_near"] for x in a])}')
    for d, r in L:
        print(f'    {d["shape"]:13s} s{d["seed"]} ' + ' '.join(f'{k[0]}:{int(r["abl"][k]["fid_correct"]) if k != "R_random" else round(r["abl"][k]["fid_correct"], 2)}/{r["abl"][k]["iou"]:.2f}' for k in ('A_protected', 'B_counterpart', 'R_random')))

"""Interpretability of TopoSaliency on the dense 2D U-Nets (no fine-tuning, so effects are attributable to the weights).

For one (dataset, seed) and sparsities S:
  1. per-layer statistics: kept fraction under global magnitude / global TopoSaliency / SynFlow allocation,
     rank correlation of topological Fisher vs magnitude, Jaccard(ts_alloc, alloc_mag) inside the SynFlow budget;
  2. 'protected' set A = kept by ts_alloc but dropped by alloc_mag, its counterpart B = kept by alloc_mag but dropped
     by ts_alloc (|A_l| = |B_l| in every layer), and a size-matched random set R in the same layers;
  3. causal ablation: zero A, B or R in the DENSE model and measure (a) task metrics, (b) Betti fidelity to the dense
     prediction, (c) how much the probability map moves at the critical cells (birth/death cells of persistent bars of
     the dense prediction) vs everywhere else, and how many critical cells cross the 0.5 threshold.
Writes ROOT/results/interp/<ds>_s<seed>.json
"""
import os, json, time, argparse, copy, numpy as np, torch, torch.nn as nn, multiprocessing as mp
import common as C

def layer_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and m is not model.out]

def global_mask(scores, sp):
    allS = torch.cat([s.flatten() for s in scores]); k = max(1, int(round((1 - sp) * allS.numel())))
    thr = torch.topk(allS, k).values[-1]
    return [(s >= thr) for s in scores]

def count_mask(scores, counts):
    out = []
    for s, k in zip(scores, counts):
        f = s.flatten(); m = torch.zeros_like(f, dtype=torch.bool); m[torch.topk(f, int(max(1, min(k, f.numel())))).indices] = True
        out.append(m.view(s.shape))
    return out

def spearman(a, b, n=200000, g=None):
    a = a.flatten(); b = b.flatten()
    if a.numel() > n:
        idx = torch.randperm(a.numel(), generator=g)[:n]; a, b = a[idx], b[idx]
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))

def critical_cells(args):
    """Birth/death cells of bars (dims 0,1) of the superlevel filtration of p with persistence > tau."""
    import gudhi
    C.ensure_order()
    p, tau = args
    f = (1.0 - p).astype(np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=f)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=tau)
    reg, ess = cc.cofaces_of_persistence_pairs()
    fl = f.ravel(order=C.ORDER); cells = []
    for dim in (0, 1):
        if len(reg) > dim and len(reg[dim]):
            for bi, di in reg[dim]:
                if fl[di] - fl[bi] > tau: cells += [int(bi), int(di)]
        if len(ess) > dim and len(ess[dim]): cells += [int(b) for b in ess[dim]]
    return np.unique(np.asarray(cells, dtype=np.int64))

def crit_stats(pr, dn, crit):
    a = b = c = d = 0.; nc = nall = 0
    for i in range(len(dn)):
        diff = np.abs(pr[i] - dn[i]).ravel(order=C.ORDER); cr = crit[i]
        flip = ((pr[i] > .5) != (dn[i] > .5)).ravel(order=C.ORDER)
        a += diff[cr].sum(); b += diff.sum(); c += flip[cr].sum(); d += flip.sum(); nc += len(cr); nall += diff.size
    return dict(dp_crit=a / max(nc, 1), dp_all=b / nall, flip_crit=c / max(nc, 1), flip_all=d / nall, n_crit=nc / len(dn))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sps', default='0.97,0.99'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tau', type=float, default=0.1); ap.add_argument('--n_img', type=int, default=10)
    ap.add_argument('--crop', type=int, default=512); ap.add_argument('--n_rand', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.workers))
    pool = mp.get_context('fork' if os.name != 'nt' else 'spawn').Pool(a.workers)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = C.get_device(); data = C.Data(a.ds, dev, seed=a.seed)
    # evaluation subset: first n_img test images, centre crop (keeps persistence affordable on Roads)
    X = data.Xte[:a.n_img]; Y = data.Yte[:a.n_img]; H, W = X.shape[2:]
    r0, c0 = max(0, (H - a.crop) // 2), max(0, (W - a.crop) // 2)
    data.Xte = X[:, :, r0:r0 + a.crop, c0:c0 + a.crop].contiguous(); data.Yte = Y[:, r0:r0 + a.crop, c0:c0 + a.crop]
    dense = C.UNet(data.inch).to(dev); dense.load_state_dict(torch.load(f'{C.ROOT}/models/{a.ds}_s{a.seed}.pt', map_location='cpu'))
    names = layer_names(dense)
    dense_res, dn = C.evaluate(dense, data, pool)
    crit = pool.map(critical_cells, [(dn[i], a.tau) for i in range(len(dn))])
    t0 = time.time()
    m = copy.deepcopy(dense); mk = C.Masker(m)
    mag = [p.detach().abs().float().cpu() for p in mk.params]
    mag2n = [x.float().cpu() for x in C._norm([p.detach() ** 2 for p in mk.params], mk.masks)]
    topon = [x.float().cpu() for x in C._norm(C.fisher_scores(m, mk, data, 'topo', pool), mk.masks)]
    ts = [x + y for x, y in zip(mag2n, topon)]
    # BN-aware ('effective') magnitude: a conv followed by BatchNorm is rescaled per output channel by |gamma|/sqrt(var+eps)
    mods = dict(dense.named_modules()); eff = []
    for nm, w in zip(names, mag):
        bn = mods.get(nm.rsplit('.', 1)[0] + '.1') if nm.endswith('.0') else None
        if isinstance(bn, nn.BatchNorm2d):
            sc = (bn.weight.detach().abs() / torch.sqrt(bn.running_var.detach() + bn.eps)).float().cpu()
            eff.append(w * sc.view(-1, *([1] * (w.dim() - 1))))
        else: eff.append(w.clone())
    g = torch.Generator().manual_seed(a.seed)
    out = dict(ds=a.ds, seed=a.seed, tau=a.tau, n_img=len(dn), dense=dense_res, layers=names,
               numel=[int(x.numel()) for x in mag], rho_topo_mag=[spearman(x, y, g=g) for x, y in zip(topon, mag)],
               topo_share=[float(x.sum() / sum(t.sum() for t in topon)) for x in topon],
               crit_dense=dict(n_crit=float(np.mean([len(c) for c in crit]))), t_scores=time.time() - t0, sp={})
    for sp in map(float, a.sps.split(',')):
        counts = C.synflow_allocation(lambda: C.UNet(data.inch), dense.state_dict(), (1, data.inch, 64, 64), sp, dev)
        M = dict(mag=global_mask(mag, sp), ts=global_mask(ts, sp), alloc_mag=count_mask(mag, counts), ts_alloc=count_mask(ts, counts),
                 alloc_eff=count_mask(eff, counts))
        A = [x & ~y for x, y in zip(M['ts_alloc'], M['alloc_mag'])]; B = [y & ~x for x, y in zip(M['ts_alloc'], M['alloc_mag'])]
        rec = dict(counts=counts,
                   keep={k: [float(x.float().mean()) for x in v] for k, v in M.items()},
                   jaccard=[float((x & y).sum() / max(1, (x | y).sum())) for x, y in zip(M['ts_alloc'], M['alloc_mag'])],
                   nA=[int(x.sum()) for x in A],
                   topo_mass_A=float(sum((t * x).sum() for t, x in zip(topon, A))),
                   topo_mass_B=float(sum((t * x).sum() for t, x in zip(topon, B))),
                   mag_mean_A=float(sum((t * x).sum() for t, x in zip(mag, A)) / max(1, sum(x.sum() for x in A))),
                   mag_mean_B=float(sum((t * x).sum() for t, x in zip(mag, B)) / max(1, sum(x.sum() for x in B))),
                   eff_mean_A=float(sum((t * x).sum() for t, x in zip(eff, A)) / max(1, sum(x.sum() for x in A))),
                   eff_mean_B=float(sum((t * x).sum() for t, x in zip(eff, B)) / max(1, sum(x.sum() for x in B))),
                   jacc_ts_effalloc=[float((x & y).sum() / max(1, (x | y).sum())) for x, y in zip(M['ts_alloc'], count_mask(eff, counts))],
                   abl={})
        def ablate(Z):
            mm = copy.deepcopy(dense)
            with torch.no_grad():
                for p, z in zip(C.prunable(mm), Z): p[z.to(p.device)] = 0
            res, pr = C.evaluate(mm, data, pool, dn)
            res.update(crit_stats(pr, dn, crit)); return res
        rec['abl']['A_protected'] = ablate(A)
        rec['abl']['B_counterpart'] = ablate(B)
        # same comparison with BN-aware magnitude: weights kept by ts_alloc but dropped by BN-aware-magnitude allocation
        A2 = [x & ~y for x, y in zip(M['ts_alloc'], M['alloc_eff'])]; rec['nA_eff'] = [int(x.sum()) for x in A2]
        rec['abl']['A_vs_eff'] = ablate(A2)
        rs = []
        for r in range(a.n_rand):
            Z = []
            for x in A:
                f = torch.zeros(x.numel(), dtype=torch.bool); f[torch.randperm(x.numel(), generator=g)[:int(x.sum())]] = True
                Z.append(f.view(x.shape))
            rs.append(ablate(Z))
        rec['abl']['R_random'] = {k: float(np.mean([q[k] for q in rs])) for k in rs[0]}
        out['sp'][str(sp)] = rec
        print(f'[{a.ds} s{a.seed} sp={sp}] |A|={sum(rec["nA"])} ' + ' | '.join(
            f'{k}: dice={v["dice"]:.3f} fid_b0={v["fid_b0"]:.3f} dp_crit={v["dp_crit"]:.4f} dp_all={v["dp_all"]:.4f} flip_crit={v["flip_crit"]:.4f}'
            for k, v in rec['abl'].items()), flush=True)
    od = f'{C.ROOT}/results/interp'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/{a.ds}_s{a.seed}.json.tmp{os.getpid()}'
    with open(tmp, 'w') as f: json.dump(out, f, default=float)
    os.replace(tmp, f'{od}/{a.ds}_s{a.seed}.json')   # atomic: a crash never leaves a 'done' marker
    print('done', time.time() - t0, flush=True)

if __name__ == '__main__':
    main()

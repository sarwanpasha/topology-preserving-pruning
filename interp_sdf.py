"""Interpretability for neural implicit shapes (SDF MLPs): the 3D counterpart of interp.py.

For one (shape, seed): fix the SynFlow layer budget, compare ts_alloc vs alloc_mag masks, and zero in the DENSE SDF
  A = kept by TopoSaliency, dropped by magnitude;  B = the size-matched counterpart;  R = size-matched random,
then measure whether the zero level set keeps the dense Betti numbers (b0,b1,b2), IoU, and how much the field moves at
the critical voxels (birth/death voxels of persistent bars that can affect {f<=0}) vs near-surface / all voxels.
Writes ROOT/results/interp/sdf_<shape>_s<seed>.json (atomically).
"""
import os, json, time, argparse, copy, numpy as np, torch
import common as C
import run_sdf as S
from interp import spearman, global_mask, count_mask

def critical_voxels(f, window=0.1, tau=0.01):
    import gudhi
    C.ensure_order()
    f64 = f.astype(np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=f64)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=tau)
    reg, ess = cc.cofaces_of_persistence_pairs()
    fl = f64.ravel(order=C.ORDER); cells = []
    for dim in range(3):
        if len(reg) > dim and len(reg[dim]):
            for bi, di in reg[dim]:
                if fl[bi] < window and fl[di] > -window and fl[di] - fl[bi] > tau: cells += [int(bi), int(di)]
        if len(ess) > dim and len(ess[dim]):
            cells += [int(b) for b in ess[dim] if fl[b] < window]
    return np.unique(np.asarray(cells, dtype=np.int64))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sps', default='0.97,0.99'); ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--n_rand', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('TOPO_THREADS', '4')))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = C.get_device(); sh = S.Shape(a.shape, dev, a.seed)
    dense = S.SDFNet().to(dev); dense.load_state_dict(torch.load(f'{C.ROOT}/models/sdf_{a.shape}_s{a.seed}.pt', map_location='cpu'))
    t0 = time.time()
    dense_res, dense_f = S.eval_sdf(dense, sh)
    dn = dense_f.view(64, 64, 64).cpu().numpy()
    crit = critical_voxels(dn)
    near = (np.abs(dn) < 0.03).ravel(order=C.ORDER)
    m = copy.deepcopy(dense); mk = C.Masker(m)
    mag = [p.detach().abs().float().cpu() for p in mk.params]
    mag2n = [x.float().cpu() for x in C._norm([p.detach() ** 2 for p in mk.params], mk.masks)]
    topon = [x.float().cpu() for x in C._norm(S.fisher_sdf(m, mk, sh, 'topo', None), mk.masks)]
    ts = [x + a.lam * y for x, y in zip(mag2n, topon)]
    g = torch.Generator().manual_seed(a.seed)
    names = [n for n, mm in dense.named_modules() if isinstance(mm, torch.nn.Linear) and mm is not dense.out]
    out = dict(shape=a.shape, seed=a.seed, dense=dense_res, layers=names, numel=[int(x.numel()) for x in mag],
               rho_topo_mag=[spearman(x, y, g=g) for x, y in zip(topon, mag)],
               topo_share=[float(x.sum() / sum(t.sum() for t in topon)) for x in topon],
               n_crit=int(len(crit)), t_scores=time.time() - t0, sp={})

    def ablate(Z):
        mm = copy.deepcopy(dense)
        with torch.no_grad():
            for p, z in zip(C.prunable(mm), Z): p[z.to(p.device)] = 0
        res, f = S.eval_sdf(mm, sh, dense_f)
        fa = f.view(64, 64, 64).cpu().numpy()
        d = np.abs(fa - dn).ravel(order=C.ORDER); flip = (np.sign(fa) != np.sign(dn)).ravel(order=C.ORDER)
        res.update(dp_crit=float(d[crit].mean()) if len(crit) else float('nan'), dp_near=float(d[near].mean()),
                   dp_all=float(d.mean()), flip_crit=float(flip[crit].mean()) if len(crit) else float('nan'),
                   flip_near=float(flip[near].mean()), flip_all=float(flip.mean()))
        res.pop('betti_gt', None)
        return res

    for sp in map(float, a.sps.split(',')):
        counts = C.synflow_allocation(S.SDFNet, dense.state_dict(), (1, 3), sp, dev)
        M = dict(mag=global_mask(mag, sp), ts=global_mask(ts, sp), alloc_mag=count_mask(mag, counts), ts_alloc=count_mask(ts, counts))
        A = [x & ~y for x, y in zip(M['ts_alloc'], M['alloc_mag'])]; B = [y & ~x for x, y in zip(M['ts_alloc'], M['alloc_mag'])]
        rec = dict(counts=counts, keep={k: [float(x.float().mean()) for x in v] for k, v in M.items()},
                   jaccard=[float((x & y).sum() / max(1, (x | y).sum())) for x, y in zip(M['ts_alloc'], M['alloc_mag'])],
                   nA=[int(x.sum()) for x in A],
                   topo_mass_A=float(sum((t * x).sum() for t, x in zip(topon, A))),
                   topo_mass_B=float(sum((t * x).sum() for t, x in zip(topon, B))),
                   mag_mean_A=float(sum((t * x).sum() for t, x in zip(mag, A)) / max(1, sum(x.sum() for x in A))),
                   mag_mean_B=float(sum((t * x).sum() for t, x in zip(mag, B)) / max(1, sum(x.sum() for x in B))),
                   abl={})
        rec['abl']['A_protected'] = ablate(A)
        rec['abl']['B_counterpart'] = ablate(B)
        rs = []
        for r in range(a.n_rand):
            Z = []
            for x in A:
                f = torch.zeros(x.numel(), dtype=torch.bool); f[torch.randperm(x.numel(), generator=g)[:int(x.sum())]] = True
                Z.append(f.view(x.shape))
            rs.append(ablate(Z))
        rec['abl']['R_random'] = {k: float(np.mean([q[k] for q in rs])) for k in rs[0] if not isinstance(rs[0][k], list)}
        out['sp'][str(sp)] = rec
        print(f'[{a.shape} s{a.seed} sp={sp}] |A|={sum(rec["nA"])} ' + ' | '.join(
            f'{k}: fid_correct={v["fid_correct"]:.2f} iou={v["iou"]:.3f} dp_crit={v["dp_crit"]:.4f} dp_near={v["dp_near"]:.4f} '
            f'flip_crit={v["flip_crit"]:.3f}' for k, v in rec['abl'].items()), flush=True)
    od = f'{C.ROOT}/results/interp'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/sdf_{a.shape}_s{a.seed}.json.tmp{os.getpid()}'
    with open(tmp, 'w') as f: json.dump(out, f, default=float)
    os.replace(tmp, f'{od}/sdf_{a.shape}_s{a.seed}.json')
    print('done', time.time() - t0, flush=True)

if __name__ == '__main__':
    main()

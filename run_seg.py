"""One (dataset, seed, method) run over a list of sparsities. Dense model trained/cached on first use."""
import os, sys, json, time, argparse, copy, numpy as np, torch, multiprocessing as mp
import common as C
try:
    import fcntl
except ImportError:  # Windows: dense model is trained by a separate step first, so no lock needed
    fcntl = None

METHODS = {
    # name:        score,      iterative, finetune loss
    'random':      ('random',   False, 'task'),
    'mag':         ('mag',      False, 'task'),
    'imp':         ('mag',      True,  'task'),
    'taylor':      ('taylor',   False, 'task'),
    'synflow':     ('synflow',  False, 'task'),
    'mag_cl':      ('mag',      False, 'cl'),
    'mag_topo':    ('mag',      False, 'topo'),
    'imp_topo':    ('mag',      True,  'topo'),
    'taylor_topo': ('taylor',   False, 'topo'),
    'ts':          ('tsmag',    False, 'topo'),    # ours: magnitude + topological-Fisher saliency
    'ts_iter':     ('tsmag',    True,  'topo'),    # ours, iterative
    'ts_fisher':   ('tsfisher', False, 'topo'),    # ours, task-Fisher + topo-Fisher variant
    'ts_taskft':   ('tsmag',    False, 'task'),    # ablation: our score, plain fine-tuning
    'synflow_topo':   ('synflow',     False, 'topo'),  # strongest baseline + topological fine-tuning
    'alloc_mag_topo': ('alloc:mag',   False, 'topo'),  # SynFlow layer allocation, magnitude within layers
    'ts_alloc':       ('alloc:tsmag', False, 'topo'),  # ours: SynFlow layer allocation, TopoSaliency within layers
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--methods', default='mag'); ap.add_argument('--sparsities', default='0.8,0.9,0.95,0.97,0.99')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--ft_iters', type=int, default=1500)
    ap.add_argument('--rounds', type=int, default=5); ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--lam_topo', type=float, default=0.005); ap.add_argument('--lr_ft', type=float, default=3e-4)
    ap.add_argument('--dense_only', action='store_true'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default='main')
    a = ap.parse_args()
    pool = mp.get_context('fork' if os.name != 'nt' else 'spawn').Pool(a.workers) if a.workers > 0 else None
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = C.get_device()
    data = C.Data(a.ds, dev, seed=a.seed)
    if a.smoke:
        C.DS_CFG[a.ds]['iters'] = 300; a.ft_iters = 100
    log = lambda s: print(f'[{a.ds} s{a.seed} {time.strftime("%H:%M:%S")}] {s}', flush=True)
    os.makedirs(f'{C.ROOT}/models', exist_ok=True)
    dpath = f'{C.ROOT}/models/{a.ds}_s{a.seed}{"_smoke" if a.smoke else ""}.pt'
    outdir = f'{C.ROOT}/results/{a.tag}/{a.ds}'; os.makedirs(outdir, exist_ok=True)

    # ---- dense model (file lock so parallel processes train it once)
    with open(dpath + '.lock', 'w') as lf:
        fcntl and fcntl.flock(lf, fcntl.LOCK_EX)
        if not os.path.exists(dpath):
            log('training dense')
            m = C.UNet(data.inch).to(dev)
            C.train_loop(m, data, C.DS_CFG[a.ds]['iters'], 1e-3, log=log)
            torch.save(m.state_dict(), dpath)
            res, _ = C.evaluate(m, data, pool)
            with open(f'{outdir}/dense.jsonl', 'a') as f:
                f.write(json.dumps(dict(ds=a.ds, seed=a.seed, method='dense', sparsity=0.0, **res)) + '\n')
            log(f'dense {res}')
        fcntl and fcntl.flock(lf, fcntl.LOCK_UN)
    if a.dense_only: return
    dense = C.UNet(data.inch).to(dev); dense.load_state_dict(torch.load(dpath, map_location='cpu'))
    dense_pred = C.predict(dense, data.Xte)

    for meth in a.methods.split(','):
        score, iterative, ft = METHODS[meth]
        donef = f'{outdir}/{meth}.jsonl'
        done = {json.loads(l)['sparsity'] for l in open(donef) if f'"seed": {a.seed},' in l} if os.path.exists(donef) else set()
        for sp in map(float, a.sparsities.split(',')):
            if sp in done: log(f'skip {meth} sp={sp}'); continue
            t0 = time.time()
            torch.manual_seed(1000 * a.seed + int(sp * 1000)); data.rng = np.random.RandomState(a.seed * 7 + 1)
            m = C.UNet(data.inch).to(dev); m.load_state_dict(dense.state_dict()); mk = C.Masker(m)
            if score.startswith('alloc:'):
                counts = C.synflow_allocation(lambda: C.UNet(data.inch), dense.state_dict(), (1, data.inch, 64, 64), sp, dev)
                mk.prune_to_counts(C.compute_scores(score.split(':')[1], m, mk, data, pool, a.lam), counts)
                C.train_loop(m, data, a.ft_iters, a.lr_ft, pool, mk, ft, a.lam_topo)
            elif score == 'synflow':
                C.prune_synflow(m, mk, (1, data.inch, 64, 64), sp, dev)
                C.train_loop(m, data, a.ft_iters, a.lr_ft, pool, mk, ft, a.lam_topo, log=log)
            elif iterative:
                for r in range(1, a.rounds + 1):
                    s_r = 1 - (1 - sp) ** (r / a.rounds)
                    mk.prune_to(C.compute_scores(score, m, mk, data, pool, a.lam), s_r)
                    C.train_loop(m, data, a.ft_iters // a.rounds, a.lr_ft, pool, mk, ft, a.lam_topo)
            else:
                mk.prune_to(C.compute_scores(score, m, mk, data, pool, a.lam), sp, layerwise=(score == 'random'))
                C.train_loop(m, data, a.ft_iters, a.lr_ft, pool, mk, ft, a.lam_topo, log=log if a.smoke else None)
            res, _ = C.evaluate(m, data, pool, dense_pred)
            rec = dict(ds=a.ds, seed=a.seed, method=meth, sparsity=sp, actual_sparsity=mk.sparsity(),
                       lam=a.lam, lam_topo=a.lam_topo, ft_iters=a.ft_iters, time=time.time() - t0, **res)
            with open(f'{outdir}/{meth}.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
            log(f'{meth} sp={sp} dice={res["dice"]:.4f} cl={res["cldice"]:.4f} b0={res["b0"]:.3f} b1={res["b1"]:.3f} '
                f'fid_b0={res["fid_b0"]:.3f} ({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()

"""Application 2: compressing neural implicit shapes (SDF MLPs) while preserving genus / Betti numbers."""
import os, json, time, argparse, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multiprocessing as mp
import common as C
try:
    import fcntl
except ImportError:  # Windows: dense model is trained by a separate step first, so no lock needed
    fcntl = None

# ----------------------------------------------------------------------------- analytic shapes
def _torus(p, c, R, r, axis=2):
    q = p - torch.tensor(c, dtype=p.dtype, device=p.device)
    ax = [0, 1, 2]; ax.remove(axis)
    a = torch.sqrt(q[:, ax[0]] ** 2 + q[:, ax[1]] ** 2) - R
    return torch.sqrt(a ** 2 + q[:, axis] ** 2) - r

def _box(p, h):
    q = p.abs() - h
    return q.clamp(min=0).norm(dim=1) + q.max(1).values.clamp(max=0)

def _cyl(p, r, axis):
    ax = [0, 1, 2]; ax.remove(axis)
    return torch.sqrt(p[:, ax[0]] ** 2 + p[:, ax[1]] ** 2) - r

SHAPES = {
    'sphere':      lambda p: p.norm(dim=1) - 0.6,
    'torus':       lambda p: _torus(p, (0, 0, 0), 0.5, 0.2),
    'thin_torus':  lambda p: _torus(p, (0, 0, 0), 0.55, 0.07),
    'double_torus': lambda p: torch.minimum(_torus(p, (-0.38, 0, 0), 0.4, 0.13), _torus(p, (0.38, 0, 0), 0.4, 0.13)),
    'triple_torus': lambda p: torch.minimum(torch.minimum(_torus(p, (-0.6, 0, 0), 0.32, 0.1), _torus(p, (0, 0, 0), 0.32, 0.1)),
                                            _torus(p, (0.6, 0, 0), 0.32, 0.1)),
    'hopf_link':   lambda p: torch.minimum(_torus(p, (-0.22, 0, 0), 0.42, 0.09, 2), _torus(p, (0.22, 0, 0), 0.42, 0.09, 1)),
    'holey_cube':  lambda p: torch.maximum(_box(p, 0.6), -torch.minimum(torch.minimum(_cyl(p, 0.25, 0), _cyl(p, 0.25, 1)), _cyl(p, 0.25, 2))),
}

def grid(n, device, jitter=0.0):
    t = torch.linspace(-1, 1, n, device=device)
    g = torch.stack(torch.meshgrid(t, t, t, indexing='ij'), -1).view(-1, 3)
    if jitter: g = g + (torch.rand(1, 3, device=device) - .5) * jitter
    return g

# ----------------------------------------------------------------------------- model
class SDFNet(nn.Module):
    def __init__(self, L=6, width=256, depth=5):
        super().__init__()
        self.register_buffer('freq', (2.0 ** torch.arange(L)) * np.pi)
        d = 3 + 6 * L; layers = []
        for _ in range(depth): layers.append(nn.Linear(d, width)); d = width
        self.hidden = nn.ModuleList(layers); self.out = nn.Linear(d, 1)

    def forward(self, x):
        e = x[..., None] * self.freq
        h = torch.cat([x, torch.sin(e).flatten(-2), torch.cos(e).flatten(-2)], -1)
        for l in self.hidden: h = F.softplus(l(h), beta=100)
        return self.out(h)[..., 0]

def evalf(model, pts, chunk=131072):
    return torch.cat([model(pts[i:i + chunk]) for i in range(0, len(pts), chunk)])

# ----------------------------------------------------------------------------- topology on 3D grids
def betti_level0(f):
    """Betti numbers (b0,b1,b2) of the sublevel set {f <= 0} of a 3D voxel field (gudhi cubical)."""
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=f.astype(np.float64))
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    b = [0, 0, 0]
    for dim, (bi, de) in cc.persistence():
        if dim < 3 and bi <= 0 < de: b[dim] += 1
    return tuple(b)

def topo_pairs3d(args, window=0.1):
    """Persistence pairs of the sublevel filtration of the SDF on a grid, restricted to bars that can affect the
    zero level set (they intersect [-window, window]); far-field bars cannot change the topology of {f<=0}.
    The beta_k most persistent relevant bars are 'kept' (should straddle 0), all other relevant bars are 'unwanted'."""
    import gudhi
    C.ensure_order()
    f, beta = args
    f64 = f.astype(np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=f64)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    reg, ess = cc.cofaces_of_persistence_pairs()
    fl = f64.ravel(order=C.ORDER)
    kb, kd, xb, xd = [], [], [], []
    for dim in range(3):
        bars = []
        if len(reg) > dim and len(reg[dim]):
            for bi, di in reg[dim]:
                if fl[bi] < window and fl[di] > -window: bars.append((fl[di] - fl[bi], int(bi), int(di)))
        if len(ess) > dim and len(ess[dim]):
            for bi in ess[dim]:
                if fl[bi] < window: bars.append((np.inf, int(bi), -1))
        bars.sort(key=lambda t: -t[0])
        for j, (pers, bi, di) in enumerate(bars):
            if j < beta[dim]: kb.append(bi); kd.append(di)
            else: xb.append(bi); xd.append(di)
    a = lambda v: np.asarray(v, dtype=np.int64)
    return a(kb), a(kd), a(xb), a(xd)

class Shape:
    def __init__(self, name, device, seed, ntopo=40):
        self.name, self.dev = name, device; self.fn = SHAPES[name]
        g = torch.Generator(device='cpu').manual_seed(seed)
        u = torch.rand(400000, 3, generator=g) * 2 - 1
        near = []
        while sum(len(n) for n in near) < 300000:
            c = torch.rand(400000, 3, generator=g) * 2 - 1; fc = self.fn(c); near.append(c[fc.abs() < 0.08])
        pts = torch.cat([u, torch.cat(near)[:300000]]).to(device)
        self.pts, self.sdf = pts, self.fn(pts)
        self.rng = np.random.RandomState(seed)
        self.ntopo = ntopo
        self.beta_topo = betti_level0(self.fn(grid(ntopo, 'cpu')).view(ntopo, ntopo, ntopo).numpy())
        self.ge = grid(64, device); self.fe = self.fn(self.ge)
        self.beta = betti_level0(self.fe.view(64, 64, 64).cpu().numpy())
        near_e = self.fe.abs() < 0.03; self.near_e = near_e

    def batch(self, bs=None):
        bs = bs or int(os.environ.get('TOPO_SDF_BS', '16384'))
        i = torch.as_tensor(self.rng.randint(0, len(self.pts), bs), device=self.dev)
        return self.pts[i], self.sdf[i]

def fit_loss(model, x, y, clamp=0.1):
    # truncated-SDF L1 that keeps gradients when the prediction is outside the band:
    # inside the band |y|<c regress y; outside it only require the correct sign with margin c
    f = model(x)
    return torch.where(y.abs() < clamp, (f - y).abs(), F.relu(torch.full_like(f, clamp) - torch.sign(y) * f)).mean()

def topo_loss3d(model, shape, pool, c=0.02):
    """Topological margin loss on the zero level set: kept bars must contain [-c, c] (feature robustly present),
    every other relevant bar must avoid [-c, c] (pushed to the nearer side), so Betti numbers of {f<=0} equal beta."""
    n = shape.ntopo; g = grid(n, shape.dev, jitter=2.0 / (n - 1))
    f = evalf(model, g)
    fn = f.detach().view(n, n, n).cpu().numpy()
    kb, kd, xb, xd = topo_pairs3d((fn, shape.beta_topo))
    dev = f.device; loss = f.sum() * 0
    cc = lambda t: torch.full_like(t, c)
    if len(kb):
        fb = f[torch.as_tensor(kb, device=dev)]; loss = loss + (F.relu(fb + cc(fb)) ** 2).sum()
        kd2 = kd[kd >= 0]
        if len(kd2):
            fd = f[torch.as_tensor(kd2, device=dev)]; loss = loss + (F.relu(cc(fd) - fd) ** 2).sum()
    if len(xb):
        fb = f[torch.as_tensor(xb, device=dev)]; up = F.relu(cc(fb) - fb) ** 2        # move birth above +c
        xdt = torch.as_tensor(xd, device=dev)
        fd = torch.where(xdt >= 0, f[xdt.clamp(min=0)], torch.full_like(fb, 1e3))
        down = F.relu(fd + cc(fd)) ** 2                                                # or death below -c
        loss = loss + torch.minimum(up, down).sum()
    return loss

def train_sdf(model, shape, iters, lr, pool=None, masker=None, ft='task', lam_topo=1.0, topo_every=1):
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=False if C.is_dml(next(model.parameters()).device) else None); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    model.train()
    for it in range(iters):
        x, y = shape.batch(); loss = fit_loss(model, x, y)
        if ft == 'topo' and it % topo_every == 0: loss = loss + lam_topo * topo_loss3d(model, shape, pool)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if masker is not None: masker.apply()
        if masker is None and (it % 500 == 0 or it == iters - 1):
            print(f'  dense it {it}/{iters} loss {loss.item():.5f}', flush=True)
    return model

def fisher_sdf(model, masker, shape, kind, pool, nb=16):
    acc = [torch.zeros_like(p) for p in masker.params]
    for _ in range(nb):
        model.zero_grad()
        if kind == 'task': x, y = shape.batch(); loss = fit_loss(model, x, y)
        else: loss = topo_loss3d(model, shape, pool)
        loss.backward()
        for a, p in zip(acc, masker.params):
            if p.grad is not None: a += (p.detach() * p.grad.detach()) ** 2
    model.zero_grad(); return acc

def scores_sdf(kind, model, mk, shape, pool, lam):
    P = mk.params
    if kind == 'mag': return [p.detach().abs() for p in P]
    if kind == 'random': return [torch.rand_like(p) for p in P]
    if kind == 'taylor': return fisher_sdf(model, mk, shape, 'task', pool)
    topo = C._norm(fisher_sdf(model, mk, shape, 'topo', pool), mk.masks)
    if kind == 'tsmag':
        mag2 = C._norm([p.detach() ** 2 for p in P], mk.masks); return [a + lam * b for a, b in zip(mag2, topo)]
    if kind == 'tsfisher':
        task = C._norm(fisher_sdf(model, mk, shape, 'task', pool), mk.masks); return [a + lam * b for a, b in zip(task, topo)]
    raise ValueError(kind)

@torch.no_grad()
def eval_sdf(model, shape, dense_f=None):
    f = evalf(model, shape.ge); fn = f.view(64, 64, 64).cpu().numpy()
    b = betti_level0(fn); g = shape.beta
    ins, gin = f <= 0, shape.fe <= 0
    iou = ((ins & gin).sum() / ((ins | gin).sum() + 1e-9)).item()
    surf = (f - shape.fe)[shape.near_e].abs().mean().item()
    out = dict(b0=abs(b[0] - g[0]), b1=abs(b[1] - g[1]), b2=abs(b[2] - g[2]), betti_pred=list(b), betti_gt=list(g),
               topo_correct=int(tuple(b) == tuple(g)), iou=iou, surf_l1=surf)
    if dense_f is not None:
        bd = betti_level0(dense_f.view(64, 64, 64).cpu().numpy())
        out.update(fid_betti=int(sum(abs(x - y) for x, y in zip(b, bd))), fid_correct=int(tuple(b) == tuple(bd)),
                   linf=(f - dense_f).abs().max().item(), betti_dense=list(bd))
    return out, f

METHODS = {
    'random': ('random', False, 'task'), 'mag': ('mag', False, 'task'), 'imp': ('mag', True, 'task'),
    'taylor': ('taylor', False, 'task'), 'synflow': ('synflow', False, 'task'),
    'mag_topo': ('mag', False, 'topo'), 'imp_topo': ('mag', True, 'topo'), 'taylor_topo': ('taylor', False, 'topo'),
    'ts': ('tsmag', False, 'topo'), 'ts_iter': ('tsmag', True, 'topo'), 'ts_fisher': ('tsfisher', False, 'topo'),
    'ts_taskft': ('tsmag', False, 'task'),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--methods', default='mag'); ap.add_argument('--sparsities', default='0.8,0.9,0.95,0.97,0.99')
    ap.add_argument('--iters', type=int, default=4000); ap.add_argument('--ft_iters', type=int, default=1000)
    ap.add_argument('--rounds', type=int, default=5); ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--lam_topo', type=float, default=0.01); ap.add_argument('--lr_ft', type=float, default=1e-4)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tag', default='main')
    ap.add_argument('--dense_only', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('TOPO_THREADS', '1')))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = C.get_device()
    if a.smoke: a.iters, a.ft_iters = 300, 60
    sh = Shape(a.shape, dev, a.seed)
    log = lambda s: print(f'[{a.shape} s{a.seed} {time.strftime("%H:%M:%S")}] {s}', flush=True)
    os.makedirs(f'{C.ROOT}/models', exist_ok=True)
    dpath = f'{C.ROOT}/models/sdf_{a.shape}_s{a.seed}{"_smoke" if a.smoke else ""}.pt'
    outdir = f'{C.ROOT}/results/{a.tag}/sdf_{a.shape}'; os.makedirs(outdir, exist_ok=True)
    with open(dpath + '.lock', 'w') as lf:
        fcntl and fcntl.flock(lf, fcntl.LOCK_EX)
        if not os.path.exists(dpath):
            m = SDFNet().to(dev); t0 = time.time(); train_sdf(m, sh, a.iters, 5e-4)
            torch.save(m.state_dict(), dpath); res, _ = eval_sdf(m, sh)
            with open(f'{outdir}/dense.jsonl', 'a') as f:
                f.write(json.dumps(dict(shape=a.shape, seed=a.seed, method='dense', sparsity=0.0, **res)) + '\n')
            log(f'dense ({time.time()-t0:.0f}s) {res} gt_topo40={sh.beta_topo}')
        fcntl and fcntl.flock(lf, fcntl.LOCK_UN)
    if a.dense_only: return
    dense = SDFNet().to(dev); dense.load_state_dict(torch.load(dpath, map_location='cpu'))
    with torch.no_grad(): dense_f = evalf(dense, sh.ge)
    for meth in a.methods.split(','):
        score, iterative, ft = METHODS[meth]
        donef = f'{outdir}/{meth}.jsonl'
        done = {json.loads(l)['sparsity'] for l in open(donef) if f'"seed": {a.seed},' in l} if os.path.exists(donef) else set()
        for sp in map(float, a.sparsities.split(',')):
            if sp in done: log(f'skip {meth} sp={sp}'); continue
            t0 = time.time(); torch.manual_seed(1000 * a.seed + int(sp * 1000)); sh.rng = np.random.RandomState(a.seed * 7 + 1)
            m = SDFNet().to(dev); m.load_state_dict(dense.state_dict()); mk = C.Masker(m)
            if score == 'synflow':
                C.prune_synflow(m, mk, (1, 3), sp, dev); train_sdf(m, sh, a.ft_iters, a.lr_ft, None, mk, ft, a.lam_topo)
            elif iterative:
                for r in range(1, a.rounds + 1):
                    mk.prune_to(scores_sdf(score, m, mk, sh, None, a.lam), 1 - (1 - sp) ** (r / a.rounds))
                    train_sdf(m, sh, a.ft_iters // a.rounds, a.lr_ft, None, mk, ft, a.lam_topo)
            else:
                mk.prune_to(scores_sdf(score, m, mk, sh, None, a.lam), sp, layerwise=(score == 'random'))
                train_sdf(m, sh, a.ft_iters, a.lr_ft, None, mk, ft, a.lam_topo)
            res, _ = eval_sdf(m, sh, dense_f)
            rec = dict(shape=a.shape, seed=a.seed, method=meth, sparsity=sp, actual_sparsity=mk.sparsity(),
                       lam=a.lam, lam_topo=a.lam_topo, time=time.time() - t0, **res)
            with open(f'{outdir}/{meth}.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
            log(f'{meth} sp={sp} betti={res["betti_pred"]} gt={res["betti_gt"]} iou={res["iou"]:.4f} '
                f'surf={res["surf_l1"]:.4f} ({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()

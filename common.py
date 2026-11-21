"""Core library: data, U-Net, losses (task / clDice / persistence), topology metrics, pruning."""
import os, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi

ROOT = os.environ.get('TOPO_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work'))
DATA = os.path.join(ROOT, 'data')

def get_device():
    """TOPO_DEVICE = cuda | cpu | dml (DirectML: any DX12 GPU on Windows, e.g. Intel UHD)."""
    d = os.environ.get('TOPO_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    if d == 'dml':
        import torch_directml as dml
        idx = os.environ.get('TOPO_DML_INDEX')
        if idx is None:   # default: the integrated Intel GPU (the NVIDIA card is used through CUDA)
            names = [dml.device_name(i) for i in range(dml.device_count())]
            idx = next((i for i, n in enumerate(names) if 'intel' in n.lower()), 0)
        return dml.device(int(idx))
    return torch.device(d)

def is_dml(dev):
    return getattr(dev, 'type', str(dev)) == 'privateuseone'

DS_CFG = {
    'isbi':  dict(patch=128, iters=4000, bs=16),
    'stare': dict(patch=128, iters=4000, bs=16),
    'roads': dict(patch=192, iters=6000, bs=16),
}

# ----------------------------------------------------------------------------- data
class Data:
    """Train images kept as uint8 on CPU (memory-light); patches normalised on the fly on `device`."""
    def __init__(self, name, device, seed=0):
        z = np.load(os.path.join(DATA, name + '.npz'))
        Xtr, Xte = z['Xtr'], z['Xte']
        if Xtr.ndim == 3: Xtr, Xte = Xtr[..., None], Xte[..., None]
        s1 = np.zeros(Xtr.shape[-1]); s2 = np.zeros(Xtr.shape[-1]); n = 0
        for im in Xtr:
            v = im.reshape(-1, im.shape[-1]).astype(np.float64) / 255.; s1 += v.sum(0); s2 += (v ** 2).sum(0); n += len(v)
        mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.sd = torch.tensor(sd, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.Xtr = torch.from_numpy(np.ascontiguousarray(Xtr.transpose(0, 3, 1, 2)))          # uint8, CPU
        self.Ytr = torch.from_numpy(np.ascontiguousarray(z['Ytr'][:, None].astype(np.uint8)))  # uint8, CPU
        self.Xte = ((torch.from_numpy(np.ascontiguousarray(Xte.transpose(0, 3, 1, 2))).float() / 255.)
                    - self.mu.cpu()) / self.sd.cpu()                                           # float, CPU
        self.Yte = z['Yte'].astype(bool)
        self.inch = self.Xtr.shape[1]
        self.cfg = DS_CFG[name]; self.device = device
        self.rng = np.random.RandomState(seed)

    def batch(self, bs=None, patch=None):
        bs = bs or self.cfg['bs']; P = patch or self.cfg['patch']
        N, _, H, W = self.Xtr.shape
        xs, ys = [], []
        for _ in range(bs):
            i = self.rng.randint(N); r = self.rng.randint(H - P + 1); c = self.rng.randint(W - P + 1)
            x = self.Xtr[i, :, r:r + P, c:c + P]; y = self.Ytr[i, :, r:r + P, c:c + P]
            k = self.rng.randint(4); fl = self.rng.rand() < .5
            x = torch.rot90(x, k, (1, 2)); y = torch.rot90(y, k, (1, 2))
            if fl: x = x.flip(2); y = y.flip(2)
            xs.append(x); ys.append(y)
        x = torch.stack(xs).to(self.device).float() / 255.
        return (x - self.mu) / self.sd, torch.stack(ys).to(self.device).float()

# ----------------------------------------------------------------------------- model
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, inch, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList(); c = inch
        for o in ch:
            self.down.append(nn.Sequential(cbr(c, o), cbr(o, o))); c = o
        self.up = nn.ModuleList(); self.upc = nn.ModuleList()
        for o in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(c, o, 2, stride=2))
            self.upc.append(nn.Sequential(cbr(2 * o, o), cbr(o, o))); c = o
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x); x = F.max_pool2d(x, 2)
        for u, c in zip(self.up, self.upc):
            x = u(x); x = c(torch.cat([x, skips.pop()], 1))
        return self.out(x)

@torch.no_grad()
def predict(model, X, chunk=None):
    model.eval(); out = []; dev = next(model.parameters()).device
    chunk = chunk or int(os.environ.get('TOPO_PRED_CHUNK', '4'))
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].to(dev); H, W = x.shape[2:]
        ph, pw = (-H) % 8, (-W) % 8
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        out.append(torch.sigmoid(model(x))[:, 0, :H, :W].float().cpu())
    return torch.cat(out).numpy()

# ----------------------------------------------------------------------------- losses
def task_loss(logit, y):
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    # numerically stable BCE-with-logits written with elementary ops (runs natively on every backend, incl. DirectML)
    bce = (logit.clamp(min=0) - logit * y + torch.log1p(torch.exp(-logit.abs()))).mean()
    return bce + dice.mean()

def _erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))
def _dilate(x): return F.max_pool2d(x, 3, 1, 1)
def soft_skel(x, it=10):
    skel = F.relu(x - _dilate(_erode(x)))
    for _ in range(it):
        x = _erode(x); d = F.relu(x - _dilate(_erode(x)))
        skel = skel + F.relu(d - skel * d)
    return skel

def cldice_loss(p, y):
    sp, sy = soft_skel(p), soft_skel(y)
    tprec = ((sp * y).sum() + 1) / (sp.sum() + 1)
    tsens = ((sy * p).sum() + 1) / (sy.sum() + 1)
    return 1 - 2 * tprec * tsens / (tprec + tsens)

# ----------------------------------------------------------------------------- topology (CPU, run in worker pool)
S8 = np.ones((3, 3), bool)
def betti2d(b):
    """Betti numbers of a binary image: b0 = 8-connected foreground components,
    b1 = holes = 4-connected background components not touching the border.
    Matches the cubical complex with pixels as top cells (superlevel filtration)."""
    b = b.astype(bool)
    _, n0 = ndi.label(b, structure=S8)
    lab, nb = ndi.label(~b)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    n1 = nb - int((border > 0).sum())
    return n0, n1

ORDER = 'C'  # index order of gudhi cofaces; verified by check_gudhi_order() in smoke test

def check_gudhi_order():
    import gudhi
    rng = np.random.RandomState(0); f = rng.rand(17, 23)
    cc = gudhi.CubicalComplex(top_dimensional_cells=f); cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    reg, ess = cc.cofaces_of_persistence_pairs()
    diag = sorted((round(b, 9), round(d, 9)) for dim, (b, d) in cc.persistence() if dim == 0 and d != float('inf'))
    res = {}
    for o in 'CF':
        fl = f.ravel(order=o)
        res[o] = sorted((round(fl[i], 9), round(fl[j], 9)) for i, j in reg[0]) == diag
    return res

_ORDER_OK = False
def ensure_order():
    global ORDER, _ORDER_OK
    if not _ORDER_OK:
        r = check_gudhi_order(); ORDER = 'C' if r['C'] else 'F'
        assert r[ORDER], f'gudhi index order check failed: {r}'
        _ORDER_OK = True

def topo_pairs(args):
    """Persistence pairs of the superlevel filtration of probability map p (dims 0,1), and the
    Hu et al. (NeurIPS'19) assignment: the beta_k most persistent bars are pushed to (1,0),
    the rest to the diagonal; beta_k = Betti numbers of the ground-truth patch."""
    import gudhi
    ensure_order()
    p, y = args
    b0, b1 = betti2d(y)
    f = (1.0 - p).astype(np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=f)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    reg, ess = cc.cofaces_of_persistence_pairs()
    fl = f.ravel(order=ORDER)
    kb, kd, xb, xd = [], [], [], []
    for dim, beta in ((0, b0), (1, b1)):
        bars = []
        if len(reg) > dim and len(reg[dim]):
            for bi, di in reg[dim]: bars.append((fl[di] - fl[bi], int(bi), int(di)))
        if len(ess) > dim and len(ess[dim]):
            for bi in ess[dim]: bars.append((np.inf, int(bi), -1))
        bars.sort(key=lambda t: -t[0])
        for j, (pers, bi, di) in enumerate(bars):
            if j < beta: kb.append(bi); kd.append(di)
            elif pers > 1e-6: xb.append(bi); xd.append(di)
    a = lambda v: np.asarray(v, dtype=np.int64)
    return a(kb), a(kd), a(xb), a(xd)

def topo_loss(prob, y, pool):
    """prob, y: (B,1,H,W). Persistence computed on CPU worker pool, loss assembled in torch."""
    B = prob.shape[0]
    P = prob.detach().float().cpu().numpy()[:, 0]; Y = (y.detach().cpu().numpy()[:, 0] > .5)
    args = [(P[i], Y[i]) for i in range(B)]
    res = pool.map(topo_pairs, args) if pool is not None else list(map(topo_pairs, args))
    flat = prob.view(B, -1); dev = prob.device; loss = prob.sum() * 0
    for i, (kb, kd, xb, xd) in enumerate(res):
        v = flat[i]
        if len(kb):
            loss = loss + ((1 - v[torch.as_tensor(kb, device=dev)]) ** 2).sum()
            kd2 = kd[kd >= 0]
            if len(kd2): loss = loss + (v[torch.as_tensor(kd2, device=dev)] ** 2).sum()
        if len(xb):
            vb = v[torch.as_tensor(xb, device=dev)]
            xdt = torch.as_tensor(xd, device=dev)
            vd = torch.where(xdt >= 0, v[xdt.clamp(min=0)], torch.zeros_like(vb))
            loss = loss + ((vb - vd) ** 2).sum()
    return loss / B

# ----------------------------------------------------------------------------- metrics
def _img_metrics(args):
    from skimage.morphology import skeletonize
    pr, gt, dn, ps = args
    pb, gb = pr > .5, gt.astype(bool)
    dice = 2 * (pb & gb).sum() / (pb.sum() + gb.sum() + 1e-9)
    sp, sg = skeletonize(pb), skeletonize(gb)
    tprec = (sp & gb).sum() / (sp.sum() + 1e-9); tsens = (sg & pb).sum() / (sg.sum() + 1e-9)
    cl = 2 * tprec * tsens / (tprec + tsens + 1e-9)
    H, W = pb.shape; e0 = e1 = f0 = f1 = 0.; n = 0
    db = dn > .5 if dn is not None else None
    for r in range(0, H - ps + 1, ps):
        for c in range(0, W - ps + 1, ps):
            a = betti2d(pb[r:r + ps, c:c + ps]); g = betti2d(gb[r:r + ps, c:c + ps])
            e0 += abs(a[0] - g[0]); e1 += abs(a[1] - g[1]); n += 1
            if db is not None:
                d = betti2d(db[r:r + ps, c:c + ps]); f0 += abs(a[0] - d[0]); f1 += abs(a[1] - d[1])
    a, g = betti2d(pb), betti2d(gb)
    out = dict(dice=dice, cldice=cl, b0=e0 / n, b1=e1 / n, b0_img=abs(a[0] - g[0]), b1_img=abs(a[1] - g[1]))
    if db is not None:
        d = betti2d(db)
        out.update(fid_b0=f0 / n, fid_b1=f1 / n, fid_b0_img=abs(a[0] - d[0]), fid_b1_img=abs(a[1] - d[1]),
                   fid_dice=2 * (pb & db).sum() / (pb.sum() + db.sum() + 1e-9),
                   linf=float(np.abs(pr - dn).max()))
    return out

def evaluate(model, data, pool, dense_pred=None, ps=64):
    pr = predict(model, data.Xte)
    args = [(pr[i], data.Yte[i], None if dense_pred is None else dense_pred[i], ps) for i in range(len(pr))]
    res = pool.map(_img_metrics, args) if pool is not None else list(map(_img_metrics, args))
    agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    return agg, pr

# ----------------------------------------------------------------------------- pruning
def prunable(model):
    return [m.weight for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and m is not model.out]

class Masker:
    def __init__(self, model):
        self.params = prunable(model)
        self.masks = [torch.ones_like(p) for p in self.params]
        for p, m in zip(self.params, self.masks):
            p.register_hook(lambda g, m=m: g * m)

    @torch.no_grad()
    def apply(self):
        for p, m in zip(self.params, self.masks): p.mul_(m)

    def sparsity(self):
        t = sum(m.numel() for m in self.masks); z = sum((m.detach().cpu() == 0).sum().item() for m in self.masks)
        return z / t

    @torch.no_grad()
    def prune_to(self, scores, sparsity, layerwise=False, min_keep_frac=0.002):
        """Keep top-(1-sparsity) weights by score (global threshold unless layerwise).
        Already-pruned weights stay pruned. Each layer keeps >= min_keep_frac (anti layer-collapse)."""
        dev = self.masks[0].device; M = [m.detach().cpu() for m in self.masks]   # mask arithmetic on CPU (backend-agnostic)
        S = [torch.where(m > 0, s.detach().float().cpu(), torch.full(m.shape, -float('inf'))) for s, m in zip(scores, M)]
        if layerwise:
            for s, m, mt in zip(S, M, self.masks):
                k = max(1, int(round((1 - sparsity) * m.numel())))
                thr = torch.topk(s.flatten(), k).values[-1]
                mt.copy_(((s >= thr).float() * (m > 0).float()).to(dev))
        else:
            allS = torch.cat([s.flatten() for s in S]); total = allS.numel()
            k = max(1, int(round((1 - sparsity) * total)))
            thr = torch.topk(allS, k).values[-1]
            for s, m, mt in zip(S, M, self.masks):
                new = (s >= thr).float()
                kmin = max(1, int(min_keep_frac * m.numel()))
                if new.sum() < kmin:
                    idx = torch.topk(s.flatten(), kmin).indices; new = torch.zeros_like(new).flatten(); new[idx] = 1; new = new.view_as(s)
                mt.copy_((new * (m > 0).float()).to(dev))
        self.apply()

def _norm(xs, masks):
    tot = sum((x * (m > 0)).sum() for x, m in zip(xs, masks)); n = sum((m > 0).sum() for m in masks)
    mu = tot / n + 1e-30
    return [x / mu for x in xs]

def fisher_scores(model, masker, data, loss_kind, pool, nb=32):
    """Empirical-Fisher (OBD-style) saliency: sum_b (w * dL_b/dw)^2 over nb mini-batches."""
    model.eval(); acc = [torch.zeros_like(p) for p in masker.params]
    for _ in range(nb):
        x, y = data.batch(); model.zero_grad()
        logit = model(x)
        loss = task_loss(logit, y) if loss_kind == 'task' else topo_loss(torch.sigmoid(logit), y, pool)
        loss.backward()
        for a, p in zip(acc, masker.params):
            if p.grad is not None: a += (p.detach() * p.grad.detach()) ** 2
    model.zero_grad()
    return acc

def prune_synflow(model, masker, in_shape, sparsity, device, rounds=100):
    m64 = model.double().eval()
    signs = {}
    with torch.no_grad():
        for n, p in m64.state_dict().items():
            if p.dtype.is_floating_point: signs[n] = torch.sign(p); p.abs_()
    x = torch.ones(*in_shape, dtype=torch.float64, device=device)
    for r in range(1, rounds + 1):
        s_r = 1 - (1 - sparsity) ** (r / rounds)
        m64.zero_grad(); R = m64(x).sum(); R.backward()
        sc = [(p.detach() * p.grad.detach()).abs() for p in masker.params]
        masker.prune_to(sc, s_r)
    with torch.no_grad():
        for n, p in m64.state_dict().items():
            if n in signs: p.mul_(signs[n])
    model.float(); masker.apply()

def compute_scores(kind, model, masker, data, pool, lam=1.0):
    P = masker.params
    if kind == 'mag': return [p.detach().abs() for p in P]
    if kind == 'random': return [torch.rand_like(p) for p in P]
    if kind == 'taylor': return fisher_scores(model, masker, data, 'task', pool)
    mag2 = _norm([p.detach() ** 2 for p in P], masker.masks)
    topo = _norm(fisher_scores(model, masker, data, 'topo', pool), masker.masks)
    if kind == 'tsmag': return [a + lam * b for a, b in zip(mag2, topo)]
    if kind == 'tsfisher':
        task = _norm(fisher_scores(model, masker, data, 'task', pool), masker.masks)
        return [a + lam * b for a, b in zip(task, topo)]
    if kind == 'topo_only': return topo
    raise ValueError(kind)

# ----------------------------------------------------------------------------- training
def train_loop(model, data, iters, lr, pool=None, masker=None, ft='task', lam_topo=0.005, lam_cl=0.5, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=False if is_dml(next(model.parameters()).device) else None)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    model.train(); t0 = time.time()
    for it in range(iters):
        x, y = data.batch(); logit = model(x)
        lt = task_loss(logit, y); loss = lt; ltp = lc = torch.zeros(())
        if ft in ('topo', 'cl'):
            p = torch.sigmoid(logit)
            if ft == 'topo': ltp = topo_loss(p, y, pool); loss = loss + lam_topo * ltp
            if ft == 'cl': lc = cldice_loss(p, y); loss = loss + lam_cl * lc
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if masker is not None: masker.apply()
        if log is not None and (it % 500 == 0 or it == iters - 1):
            log(f'  it {it} task {lt.item():.4f} topo {float(ltp):.3f} cl {float(lc):.4f} ({time.time()-t0:.0f}s)')
    return model

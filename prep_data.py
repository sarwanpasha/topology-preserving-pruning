"""Download + preprocess datasets into npz files under $TOPO_ROOT/data."""
import os, re, io, sys, time, gzip, tarfile, ssl, urllib.request, numpy as np
from PIL import Image, ImageSequence
ROOT = os.environ.get('TOPO_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work'))
D = os.path.join(ROOT, 'data'); RAW = os.path.join(D, 'raw'); os.makedirs(RAW, exist_ok=True)
CTX = ssl._create_unverified_context()  # STARE host has an incomplete cert chain (public data)

def get(url, path, unverified=False, tries=12):
    """Robust download for slow/throttled hosts: streamed, resumes partial files (HTTP Range), long timeouts, backoff."""
    if os.path.exists(path) and os.path.getsize(path) > 0: return path
    part = path + '.part'
    for k in range(tries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research download)', **({'Range': f'bytes={have}-'} if have else {})})
            with urllib.request.urlopen(req, timeout=600, context=CTX if unverified else None) as r:
                mode = 'ab' if have and r.status == 206 else 'wb'
                with open(part, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        f.write(b)
            os.replace(part, path); return path
        except Exception as e:
            print('retry', k, url, e, flush=True); time.sleep(min(300, 15 * (k + 1)))
    raise RuntimeError(url)

def isbi():
    out = os.path.join(D, 'isbi.npz')
    if os.path.exists(out): return
    b = 'https://github.com/alexklibisz/isbi-2012/raw/master/data/'
    v = get(b + 'train-volume.tif', os.path.join(RAW, 'isbi_vol.tif'))
    l = get(b + 'train-labels.tif', os.path.join(RAW, 'isbi_lab.tif'))
    X = np.stack([np.array(f) for f in ImageSequence.Iterator(Image.open(v))]).astype(np.uint8)[..., None]
    L = np.stack([np.array(f) for f in ImageSequence.Iterator(Image.open(l))])
    Y = (L < 128).astype(np.uint8)  # membranes (thin curvilinear network) = foreground
    np.savez_compressed(out, Xtr=X[:20], Ytr=Y[:20], Xte=X[20:], Yte=Y[20:])
    print('isbi', X.shape, Y.mean())

def stare():
    out = os.path.join(D, 'stare.npz')
    if os.path.exists(out): return
    b = 'https://cecas.clemson.edu/~ahoover/stare/probing/'
    ti = get(b + 'stare-images.tar', os.path.join(RAW, 'stare-images.tar'), True)
    tl = get(b + 'labels-ah.tar', os.path.join(RAW, 'labels-ah.tar'), True)
    def load(tarp):
        res = {}
        with tarfile.open(tarp) as t:
            for m in t.getmembers():
                if not m.isfile(): continue
                raw = t.extractfile(m).read()
                if m.name.endswith('.gz'): raw = gzip.decompress(raw)
                key = re.search(r'(im\d+)', m.name).group(1)
                res[key] = np.array(Image.open(io.BytesIO(raw)))
        return res
    I, Lb = load(ti), load(tl)
    keys = sorted(Lb)
    X = np.stack([I[k] for k in keys]).astype(np.uint8)
    Y = (np.stack([Lb[k] for k in keys]) > 127).astype(np.uint8)
    if Y.ndim == 4: Y = Y[..., 0]
    np.savez_compressed(out, Xtr=X[:10], Ytr=Y[:10], Xte=X[10:], Yte=Y[10:])
    print('stare', X.shape, Y.mean())

def roads(ntr=300):
    out = os.path.join(D, 'roads.npz')
    if os.path.exists(out): return
    base = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/'
    rd = os.path.join(RAW, 'roads'); os.makedirs(rd, exist_ok=True)
    def names(split):
        p = get(base + split + '/sat/index.html', os.path.join(rd, f'{split}_index.html'))
        html = open(p, encoding='utf-8', errors='ignore').read()
        return sorted(set(re.findall(r'(\d+_\d+)\.tiff', html)))
    def load(split, ns):
        X, Y = [], []
        for n in ns:
            s = get(f'{base}{split}/sat/{n}.tiff', os.path.join(rd, f'{split}_{n}_s.tiff'))
            m = get(f'{base}{split}/map/{n}.tif', os.path.join(rd, f'{split}_{n}_m.tif'))
            x = Image.open(s).convert('RGB').resize((750, 750), Image.BILINEAR)
            y = Image.open(m).convert('L').resize((750, 750), Image.NEAREST)
            X.append(np.array(x)); Y.append((np.array(y) > 127).astype(np.uint8))
        return np.stack(X), np.stack(Y)
    tr = names('train'); rng = np.random.RandomState(0); tr = sorted(rng.choice(tr, ntr, replace=False))
    te = names('test')
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(f'{base}{sp}/sat/{n}.tiff', os.path.join(rd, f'{sp}_{n}_s.tiff')) for sp, ns in (('train', tr), ('test', te)) for n in ns]
    jobs += [(f'{base}{sp}/map/{n}.tif', os.path.join(rd, f'{sp}_{n}_m.tif')) for sp, ns in (('train', tr), ('test', te)) for n in ns]
    with ThreadPoolExecutor(int(os.environ.get('TOPO_DL_THREADS', '6'))) as ex: list(ex.map(lambda a: get(*a), jobs))
    Xtr, Ytr = load('train', tr); Xte, Yte = load('test', te)
    np.savez_compressed(out, Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte)
    print('roads', Xtr.shape, Xte.shape, Ytr.mean())

if __name__ == '__main__':
    which = sys.argv[1:] or ['isbi', 'stare', 'roads']
    for w in which:
        try: globals()[w]()
        except Exception as e: print('FAILED', w, repr(e), flush=True)

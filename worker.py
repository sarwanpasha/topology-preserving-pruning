"""Pull-based worker for short Slurm jobs on shared clusters.

Every job is a generic worker: it claims the next unfinished task -- one (dataset|shape, seed, method, sparsity) --
from a deterministic task list, runs it, and repeats until its time budget is used up. Claims are atomic mkdir's
on the shared filesystem; a claim older than TOPO_STALE_MIN (job killed / preempted) is taken over.
So jobs can be small, short (<= 1 h, fits interactive/backfill), land on any partition, and die at any time.

Env: TOPO_ROOT, TOPO_TAG (main), TOPO_BUDGET_MIN (55), TOPO_SLOTS (3 concurrent tasks per GPU),
     TOPO_SEG_EST_MIN (20) / TOPO_SDF_EST_MIN (12): minimum remaining minutes needed to start a task,
     TOPO_STALE_MIN (75), TOPO_SEEDS (5), TOPO_KINDS (seg,sdf)
"""
import os, sys, time, json, threading, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ['TOPO_ROOT']; TAG = os.environ.get('TOPO_TAG', 'main')
T_END = time.time() + float(os.environ.get('TOPO_BUDGET_MIN', '55')) * 60
SLOTS = int(os.environ.get('TOPO_SLOTS', '3'))
EST = {'seg': float(os.environ.get('TOPO_SEG_EST_MIN', '20')) * 60, 'sdf': float(os.environ.get('TOPO_SDF_EST_MIN', '12')) * 60,
       'segd': 15 * 60, 'sdfd': 10 * 60}
STALE = float(os.environ.get('TOPO_STALE_MIN', '75')) * 60
SEEDS = int(os.environ.get('TOPO_SEEDS', '5')); KINDS = os.environ.get('TOPO_KINDS', 'seg,sdf').split(',')
CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', '16'))
SEG_DS = ['isbi', 'stare', 'roads']
SHAPES = ['torus', 'thin_torus', 'double_torus', 'triple_torus', 'hopf_link', 'holey_cube', 'sphere']
SP = [0.8, 0.9, 0.95, 0.97, 0.99]
# key methods first (all seeds), then the rest
SEG_A = ['ts_alloc', 'ts', 'synflow', 'alloc_mag_topo', 'mag', 'mag_topo', 'taylor_topo', 'synflow_topo', 'mag_cl']
SEG_B = ['imp', 'random', 'taylor', 'imp_topo', 'ts_iter', 'ts_fisher', 'ts_taskft']
SDF_A = ['ts_alloc', 'ts', 'synflow', 'alloc_mag_topo', 'mag', 'mag_topo', 'taylor_topo', 'synflow_topo']
SDF_B = ['imp', 'random', 'taylor', 'imp_topo', 'ts_iter', 'ts_fisher', 'ts_taskft']
CL = os.path.join(ROOT, 'claims', TAG); FL = os.path.join(ROOT, 'claims_fail', TAG)
os.makedirs(CL, exist_ok=True); os.makedirs(FL, exist_ok=True); os.makedirs(os.path.join(ROOT, 'logs', 'tasks'), exist_ok=True)

def build():
    t = []
    if 'seg' in KINDS: t += [('segd', ds, s, None, None) for s in range(SEEDS) for ds in SEG_DS]
    if 'sdf' in KINDS: t += [('sdfd', sh, s, None, None) for s in range(SEEDS) for sh in SHAPES]
    for grp_seg, grp_sdf in ((SEG_A, SDF_A), (SEG_B, SDF_B)):
        for s in range(SEEDS):
            if 'seg' in KINDS: t += [('seg', ds, s, m, sp) for ds in SEG_DS for m in grp_seg for sp in SP]
            if 'sdf' in KINDS: t += [('sdf', sh, s, m, sp) for sh in SHAPES for m in grp_sdf for sp in SP]
    return t

def name(t): k, d, s, m, sp = t; return f'{k}_{d}_s{s}' + (f'_{m}_{sp}' if m else '')
def model_path(k, d, s): return os.path.join(ROOT, 'models', (f'{d}_s{s}.pt' if k in ('seg', 'segd') else f'sdf_{d}_s{s}.pt'))
_cache = {}
def done_set(k, d, m):
    f = os.path.join(ROOT, 'results', TAG, d if k == 'seg' else f'sdf_{d}', f'{m}.jsonl')
    try: st = os.stat(f)
    except OSError: return set()
    key = (f, st.st_mtime, st.st_size)
    if key not in _cache:
        out = set()
        for l in open(f):
            try: r = json.loads(l); out.add((r['seed'], r['sparsity']))
            except Exception: pass
        _cache[key] = out
    return _cache[key]
def is_done(t):
    k, d, s, m, sp = t
    if k in ('segd', 'sdfd'): return os.path.exists(model_path(k, d, s))
    return (s, sp) in done_set(k, d, m)
def fails(t):
    try: return int(open(os.path.join(FL, name(t))).read() or 0)
    except Exception: return 0

lock = threading.Lock()
def claim():
    with lock:
        left = T_END - time.time()
        for t in TASKS:
            k = t[0]
            if left < EST[k] or is_done(t) or fails(t) >= 2: continue
            if k in ('seg', 'sdf') and not os.path.exists(model_path(k, t[1], t[2])): continue
            c = os.path.join(CL, name(t))
            try:
                os.mkdir(c); return t
            except FileExistsError:
                try:
                    if time.time() - os.stat(c).st_mtime > STALE:
                        shutil.rmtree(c, ignore_errors=True); os.mkdir(c); return t
                except OSError: pass
        return None

def run(t, slot):
    k, d, s, m, sp = t
    w = max(1, CPUS // SLOTS - 1)
    if k in ('seg', 'segd'):
        cmd = [sys.executable, 'run_seg.py', '--ds', d, '--seed', str(s), '--workers', str(w), '--tag', TAG]
    else:
        cmd = [sys.executable, 'run_sdf.py', '--shape', d, '--seed', str(s), '--tag', TAG]
    cmd += ['--dense_only'] if k in ('segd', 'sdfd') else ['--methods', m, '--sparsities', str(sp)]
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOPO_THREADS=str(w))
    logf = os.path.join(ROOT, 'logs', 'tasks', name(t) + '.log'); t0 = time.time()
    with open(logf, 'a') as lf:
        lf.write(f'=== {time.ctime()} job {os.environ.get("SLURM_JOB_ID")} node {os.uname().nodename} slot {slot}\n'); lf.flush()
        try:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=max(60, T_END - time.time() + 240))
        except subprocess.TimeoutExpired:
            rc = 'timeout'
        lf.write(f'=== rc {rc} after {time.time()-t0:.0f}s\n')
    ok = is_done(t)
    if not ok and rc != 'timeout':
        with open(os.path.join(FL, name(t)), 'w') as f: f.write(str(fails(t) + 1))
    shutil.rmtree(os.path.join(CL, name(t)), ignore_errors=True)
    print(f'[{time.strftime("%H:%M:%S")}] slot{slot} {name(t)} rc={rc} ok={ok} {time.time()-t0:.0f}s', flush=True)

def slot_loop(i):
    idle = 0
    while time.time() < T_END:
        t = claim()
        if t is None:
            idle += 1
            if idle > 20: return         # nothing claimable for ~10 min: let the job end and free the GPU
            time.sleep(30); continue
        idle = 0; run(t, i)

TASKS = build()
if __name__ == '__main__':
    print(f'worker job={os.environ.get("SLURM_JOB_ID")} node={os.uname().nodename} slots={SLOTS} tasks={len(TASKS)} '
          f'remaining={sum(not is_done(t) for t in TASKS)}', flush=True)
    th = [threading.Thread(target=slot_loop, args=(i,)) for i in range(SLOTS)]
    for x in th: x.start(); time.sleep(5)
    for x in th: x.join()
    print('worker exit', time.ctime(), flush=True)

"""Laptop orchestrator: runs the full sweep with a GPU worker pool and a CPU worker pool in parallel.
Resume-safe (each runner skips finished sparsities). Usage:
    python orchestrate.py --gpu_workers 2 --cpu_workers 1
Progress: <root>/progress.json, per-job logs: <root>/logs/
"""
import os, sys, json, time, argparse, threading, subprocess, collections

HERE = os.path.dirname(os.path.abspath(__file__))
SEG_DS = ['isbi', 'stare', 'roads']
SEG_M = ['mag', 'imp', 'random', 'taylor', 'synflow', 'mag_cl', 'mag_topo', 'imp_topo', 'taylor_topo',
         'ts', 'ts_iter', 'ts_fisher', 'ts_taskft']
SHAPES = ['torus', 'thin_torus', 'double_torus', 'triple_torus', 'hopf_link', 'holey_cube', 'sphere']
SDF_M = ['random', 'mag', 'imp', 'taylor', 'synflow', 'mag_topo', 'imp_topo', 'taylor_topo',
         'ts', 'ts_iter', 'ts_fisher', 'ts_taskft']
SP = [0.8, 0.9, 0.95, 0.97, 0.99]

def n_done(root, tag, sub, meth, seed):
    f = os.path.join(root, 'results', tag, sub, f'{meth}.jsonl')
    if not os.path.exists(f): return 0
    k = f'"seed": {seed},'
    return len({json.loads(l)['sparsity'] for l in open(f) if k in l})

def build_jobs(a):
    jobs = []  # dict(name, kind seg|sdf, args, dep)
    for seed in range(a.seeds):
        for ds in SEG_DS:
            d = f'seg_{ds}_s{seed}_dense'
            jobs.append(dict(name=d, kind='seg', dep=None, args=['run_seg.py', '--ds', ds, '--seed', str(seed), '--dense_only', '--workers', '0', '--tag', a.tag],
                             done=lambda ds=ds, seed=seed: os.path.exists(os.path.join(a.root, 'models', f'{ds}_s{seed}.pt'))))
            for m in SEG_M:
                jobs.append(dict(name=f'seg_{ds}_s{seed}_{m}', kind='seg', dep=d,
                                 args=['run_seg.py', '--ds', ds, '--seed', str(seed), '--methods', m, '--workers', '0', '--tag', a.tag],
                                 done=lambda ds=ds, m=m, seed=seed: n_done(a.root, a.tag, ds, m, seed) >= len(SP)))
        for sh in SHAPES:
            d = f'sdf_{sh}_s{seed}_dense'
            jobs.append(dict(name=d, kind='sdf', dep=None, args=['run_sdf.py', '--shape', sh, '--seed', str(seed), '--dense_only', '--tag', a.tag],
                             done=lambda sh=sh, seed=seed: os.path.exists(os.path.join(a.root, 'models', f'sdf_{sh}_s{seed}.pt'))))
            for m in SDF_M:
                jobs.append(dict(name=f'sdf_{sh}_s{seed}_{m}', kind='sdf', dep=d,
                                 args=['run_sdf.py', '--shape', sh, '--seed', str(seed), '--methods', m, '--tag', a.tag],
                                 done=lambda sh=sh, m=m, seed=seed: n_done(a.root, a.tag, f'sdf_{sh}', m, seed) >= len(SP)))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu_workers', type=int, default=1); ap.add_argument('--cpu_workers', type=int, default=1)
    ap.add_argument('--dml_workers', type=int, default=0); ap.add_argument('--dml_python', default=os.path.join(HERE, '..', 'venv_dml', 'Scripts', 'python.exe'))
    ap.add_argument('--cpu_threads', type=int, default=2); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--tag', default='laptop'); ap.add_argument('--root', default=os.environ.get('TOPO_ROOT', os.path.join(HERE, '..', 'work')))
    a = ap.parse_args(); a.root = os.path.abspath(a.root)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    jobs = build_jobs(a)
    state = {j['name']: ('done' if j['done']() else 'todo') for j in jobs}
    lock = threading.Lock(); fails = collections.Counter(); t0 = time.time()

    def save():
        c = collections.Counter(state.values())
        prog = dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'), elapsed_h=round((time.time() - t0) / 3600, 2),
                    total=len(jobs), done=c['done'], running=[k for k, v in state.items() if v == 'running'],
                    failed=[k for k, v in state.items() if v == 'failed'], todo=c['todo'])
        with open(os.path.join(a.root, 'progress.json'), 'w') as f: json.dump(prog, f, indent=1)

    def pick(device):
        with lock:
            for j in jobs:
                if state[j['name']] != 'todo': continue
                if device == 'cpu' and j['kind'] == 'seg': continue      # U-Nets are GPU-only on this laptop
                if device == 'dml' and j['name'].endswith('_synflow'): continue   # DirectML has no float64 (SynFlow)
                if j['dep'] and state[j['dep']] == 'failed': state[j['name']] = 'failed'; continue
                if j['dep'] and state[j['dep']] != 'done': continue
                state[j['name']] = 'running'; save(); return j
            return None

    def worker(device, wid):
        while True:
            j = pick(device)
            if j is None:
                with lock:
                    left = any(v in ('todo', 'running') for v in state.values())
                    # stop when nothing this worker could ever run remains
                    can = any(state[x['name']] in ('todo', 'running') and not (device == 'cpu' and x['kind'] == 'seg')
                              and not (device == 'dml' and x['name'].endswith('_synflow')) for x in jobs)
                if not left or not can: return
                time.sleep(20); continue
            env = dict(os.environ, TOPO_ROOT=a.root, TOPO_DEVICE=device, TOPO_PRED_CHUNK='1',
                       TOPO_THREADS=str(a.cpu_threads if device == 'cpu' else 1), PYTHONUNBUFFERED='1')
            logf = os.path.join(a.root, 'logs', j['name'] + '.log')
            with open(logf, 'a') as lf:
                lf.write(f'\n=== {time.ctime()} on {device}{wid}\n'); lf.flush()
                py = a.dml_python if device == 'dml' else sys.executable
                rc = subprocess.call([py] + j['args'], cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
            with open(logf, 'a') as lf: lf.write(f'=== exit code {rc} at {time.ctime()}\n')
            with lock:
                if rc == 0 and j['done']():
                    state[j['name']] = 'done'
                else:
                    fails[j['name']] += 1
                    state[j['name']] = 'todo' if fails[j['name']] < 2 else 'failed'
                save()

    save()
    ths = [threading.Thread(target=worker, args=('cuda', i), daemon=True) for i in range(a.gpu_workers)]
    ths += [threading.Thread(target=worker, args=('cpu', i), daemon=True) for i in range(a.cpu_workers)]
    ths += [threading.Thread(target=worker, args=('dml', i), daemon=True) for i in range(a.dml_workers)]
    for t in ths: t.start()
    for t in ths: t.join()
    save(); print('ALL_DONE', json.load(open(os.path.join(a.root, 'progress.json')))['done'], 'of', len(jobs))

if __name__ == '__main__':
    main()

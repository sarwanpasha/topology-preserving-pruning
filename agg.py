"""Aggregate results/<tag>/*/*.jsonl -> mean ± std tables (csv + markdown)."""
import os, sys, json, glob, collections, numpy as np
ROOT = os.environ.get('TOPO_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')); tag = sys.argv[1] if len(sys.argv) > 1 else 'main'
rows = []
for f in glob.glob(f'{ROOT}/results/{tag}/*/*.jsonl'):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
G = collections.defaultdict(list)
for r in rows:
    G[(r.get('ds') or 'sdf_' + r['shape'], r['method'], r['sparsity'])].append(r)
keys_seg = ['dice', 'cldice', 'b0', 'b1', 'b0_img', 'b1_img', 'fid_b0', 'fid_b1', 'fid_dice']
keys_sdf = ['topo_correct', 'b0', 'b1', 'b2', 'iou', 'surf_l1', 'fid_correct']
out = open(f'{ROOT}/results/{tag}_summary.csv', 'w')
out.write('dataset,method,sparsity,n,' + ','.join(f'{k}_mean,{k}_std' for k in keys_seg + keys_sdf) + '\n')
for (d, m, s) in sorted(G):
    rs = G[(d, m, s)]; cells = []
    for k in keys_seg + keys_sdf:
        v = [r[k] for r in rs if k in r]
        cells += [f'{np.mean(v):.4f}', f'{np.std(v):.4f}'] if v else ['', '']
    out.write(f'{d},{m},{s},{len(rs)},' + ','.join(cells) + '\n')
out.close()
# compact console view
for d in sorted({k[0] for k in G}):
    seg = not d.startswith('sdf_')
    ks = ['dice', 'cldice', 'b0', 'b1', 'fid_b0'] if seg else ['topo_correct', 'b1', 'iou', 'surf_l1']
    print(f'\n== {d}  ({" | ".join(ks)})')
    for s in sorted({k[2] for k in G if k[0] == d}):
        print(f' sparsity {s}')
        for m in sorted({k[1] for k in G if k[0] == d and k[2] == s}):
            rs = G[(d, m, s)]
            vals = [[r[k] for r in rs if k in r] for k in ks]
            print(f'   {m:12s} n={len(rs)} ' + ' '.join(f'{np.mean(v):.4f}±{np.std(v):.4f}' if v else '   -   ' for v in vals))

"""Per-cell median excess log loss of each Gaussian filter from the synthetic run (read-only)."""
import csv, glob, json, math, sys
import numpy as np

# usage: cell_medians.py <run dir with synthetic/> <theory_regions.json> <out.csv>
H, REGIONS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

regions = {(r['cell'], float(r['qc']), float(r['dt'])): r
           for r in json.load(open(REGIONS, encoding='utf8'))['rows']}
loss = {}
for p in glob.glob(H + '/synthetic/SW*.json'):
    for x in json.load(open(p, encoding='utf8'))['records']:
        if x['method'] in ('E', 'I', 'E_cap', 'I_cap'):
            l = x['loss']
            v = l['sign'] * 10.0 ** min(l['log10'], 300) if l['sign'] else 0.0
            u = l['sign'] * l['log10'] if l['sign'] else -math.inf  # signed log10 magnitude
            loss.setdefault((x['cell'], x['qc'], x['dt'], x['method']), []).append((l['sign'], l['log10']))

def median_log10(vals):
    # median of extended reals given as (sign, log10|x|); returns (sign, log10)
    key = lambda t: (t[0], t[0] * t[1]) if t[0] else (0, 0.0)
    s = sorted(vals, key=key)
    return s[len(s) // 2 - 1] if len(s) % 2 == 0 else s[len(s) // 2]  # nearest-rank lower median

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['cell', 'L', 'rate', 'qc', 'dt', 'kappa', 'region', 'method', 'median_sign', 'median_log10', 'n'])
    for (cell, qc, dt, m), v in sorted(loss.items()):
        r = regions[(cell, qc, dt)]
        sg, lg = median_log10(v)
        w.writerow([cell, r['L'], r['rate'], qc, dt, r['kappa'], r['region'], m, sg, lg, len(v)])

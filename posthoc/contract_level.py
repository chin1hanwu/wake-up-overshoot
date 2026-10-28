"""Descriptive robustness check for R1/R2: contract-level aggregation of the
frozen real-data scores (read only; no filter is rerun)."""
import json, math, pathlib, sys
from collections import defaultdict
from scipy.stats import binomtest

root = pathlib.Path(sys.argv[1])
score = {}  # (symbol, day, dt, method) -> score as float (nats)
for f in sorted(root.glob('*.json')):
    for r in json.loads(f.read_text(encoding='utf8'))['records']:
        s = r['score']
        v = s['sign'] * 10 ** s['log10'] if isinstance(s, dict) else float(s)
        score[(r['symbol'], r['day'], r['dt'], r['method'])] = v

syms = sorted({k[0] for k in score}); days = sorted({k[1] for k in score})
dts = sorted({k[2] for k in score}); methods = sorted({k[3] for k in score})
print('symbols', len(syms), 'days', days, 'dts', dts, 'methods', methods)
for dt in dts:
    diff = {(s, d): score[(s, d, dt, 'I')] - score[(s, d, dt, 'E')] for s in syms for d in days}
    wins = sum(v > 0 for v in diff.values()); ties = sum(v == 0 for v in diff.values())
    p20 = binomtest(wins, len(diff) - ties, 0.5, alternative='greater').pvalue
    agg = {s: sum(diff[(s, d)] for d in days) for s in syms}
    w10 = sum(v > 0 for v in agg.values())
    p10 = binomtest(w10, len(agg), 0.5, alternative='greater').pvalue
    # both days must favour I for a contract to count
    both = sum(all(diff[(s, d)] > 0 for d in days) for s in syms)
    loss = [s for s in syms if agg[s] <= 0]
    print(f'dt={dt}: contract-days I>E {wins}/{len(diff)} (ties {ties}) p={p20:.2e}; '
          f'contracts (sum of 2 days) {w10}/10 p={p10:.2e}; both days {both}/10; '
          f'contracts with I<=E: {loss}; total I-E {sum(diff.values()):.4g}')
    for d in days:
        print('   ', d, 'I>E', sum(diff[(s, d)] > 0 for s in syms), '/10')

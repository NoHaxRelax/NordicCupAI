"""Per-phase effects from whole-game checkpoint rounds (lategame.py): for each variant and checkpoint, paired survival
difference against 'none' forked from the identical state. usage: phase_ana.py rows.jsonl [...]"""
import sys, json, statistics as st, math, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
key = lambda r: (r['seed'], r['t0'])
ref = {key(r): r for r in rows if r['label'] == 'none' and 'skip' not in r}
labs = sorted({r['label'] for r in rows} - {'none'})
t0s = sorted({r['t0'] for r in rows})
print('checkpoint:        ' + ''.join(f'{int(t):>12d}' for t in t0s))
print('games alive:       ' + ''.join(f'{sum(1 for k in ref if k[1]==t):>12d}' for t in t0s))
print('baseline lives on: ' + ''.join(f"{st.mean(ref[k]['surv']-t for k in ref if k[1]==t) if any(k[1]==t for k in ref) else 0:>11.0f}s" for t in t0s))
for lab in labs:
    cells = []
    for t in t0s:
        d = [r['surv'] - ref[key(r)]['surv'] for r in rows if r['label'] == lab and r['t0'] == t and 'skip' not in r and key(r) in ref]
        cells.append(f"{st.mean(d):+6.0f}+-{st.pstdev(d)/math.sqrt(len(d)):3.0f}" if len(d) > 1 else f"{'-':>12s}")
    print(f'{lab:18s} ' + ''.join(f'{c:>12s}' for c in cells))

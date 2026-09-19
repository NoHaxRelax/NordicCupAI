"""Summarise supply.py rows: paired deltas vs 'none' per checkpoint, hold, supply and colony capacity.
usage: python3 supply_ana.py rows.jsonl"""
import sys, json, collections, statistics as S

def ms(x):
    if not x: return '   -'
    se = S.stdev(x) / len(x) ** .5 if len(x) > 1 else 0.
    return f'{S.mean(x):+6.0f}±{se:3.0f}'

rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in rows: by[(r['seed'], r['t0'])][r['label']] = r
labels = list(dict.fromkeys(r['label'] for r in rows))
t0s = sorted({r['t0'] for r in rows})
print('colony at the checkpoint (none rows): n alive-at-t0, agents, births/100 s before t0, agents <30 s / <60 s old, mean energy, agents >300 energy, predators')
for t0 in t0s:
    N = [v['none'] for k, v in by.items() if k[1] == t0 and 'none' in v and 'surv' in v['none']]
    dead = sum(1 for k, v in by.items() if k[1] == t0 and v.get('none', {}).get('skip') == 'dead_before_t0')
    if not N: continue
    f = lambda key: S.mean(r[key] for r in N if r.get(key) is not None)
    print(f'  t0={t0:5.0f} alive {len(N):3d} (dead {dead:3d})  agents {f("alive0"):5.1f}  births/100s {f("births_prev100"):5.1f}  '
          f'<30s {f("young30"):4.1f} <60s {f("young60"):4.1f}  e {f("e_mean"):5.0f}  rich {f("rich"):4.1f}  preds {f("preds0"):4.1f}  none surv {S.mean(r["surv"] for r in N):6.0f}')
print()
print('paired vs none: dSurv, dScore, share reaching 3000, held predators at 30/60/120/300 s, kills/starved in first 300 s (variant vs none), supply')
for t0 in t0s:
    print(f't0={t0:.0f}')
    for l in labels:
        if l == 'none': continue
        P = [(v[l], v['none']) for k, v in by.items() if k[1] == t0 and l in v and 'none' in v and 'surv' in v[l] and 'surv' in v['none']]
        if not P: continue
        skips = collections.Counter(v[l].get('skip') for k, v in by.items() if k[1] == t0 and l in v and 'skip' in v[l] and v[l]['skip'] != 'dead_before_t0')
        ds = [a['surv'] - b['surv'] for a, b in P]; dc = [a['score'] - b['score'] for a, b in P]
        full = sum(1 for a, b in P if a['surv'] >= 2999.9) / len(P); fulln = sum(1 for a, b in P if b['surv'] >= 2999.9) / len(P)
        def held(i):
            h = [a['probe'][i][1] for a, b in P if a.get('probe') and len(a['probe']) > i]
            return f'{S.mean(h):4.1f}' if h else '   -'
        k3 = S.mean(a['kills_300'] for a, b in P); k3n = S.mean(b['kills_300'] for a, b in P)
        s3 = S.mean(a['starved_300'] for a, b in P); s3n = S.mean(b['starved_300'] for a, b in P)
        sup = [a['sup'] for a, b in P]
        extra = ''
        if any(x.get('n') for x in sup):
            ages = [g for x in sup for g in x.get('ages', [])]
            extra += f' supplied {S.mean(x.get("n", 0) for x in sup):4.1f} (age {S.mean(ages) if ages else 0:4.1f}s, dry {S.mean(x.get("dry", 0) for x in sup):3.1f})'
        if any(x.get('fed') for x in sup): extra += f' fuel {S.mean(x.get("fed", 0) for x in sup):5.0f}'
        nz = [x['nest'] for x in sup if x.get('nest')]
        if nz and any(z['cadets'] for z in nz):
            extra += (f' nest: cadets {S.mean(z["cadets"] for z in nz):4.1f} sent {S.mean(z["sent"] for z in nz):4.1f} arrived {S.mean(z["arrived"] for z in nz):4.1f}'
                      f' timeouts {S.mean(z["timeouts"] for z in nz):3.1f} e@send {S.mean(z["sent_e"] for z in nz if z["sent"]) if any(z["sent"] for z in nz) else 0:4.0f}'
                      f' age@send {S.mean(z["sent_age"] for z in nz if z["sent"]) if any(z["sent"] for z in nz) else 0:4.1f}')
        print(f'  {l:11s} n={len(P):3d} dSurv {ms(ds)} dScore {ms(dc)}  3000: {full:4.0%} vs {fulln:4.0%}  held {held(2)}/{held(5)}/{held(11)}/{held(29)}'
              f'  kills300 {k3:4.1f} vs {k3n:4.1f} starved300 {s3:5.1f} vs {s3n:5.1f}{extra}' + (f'  skips {dict(skips)}' if skips else ''))

"""Analyse nightsim rows. usage: ana.py FILES... [--ref LABEL] [--bins] [--labels a b]
Dedupes (label, seed); paired diffs vs --ref on common seeds; --bins prints harvest per 250 s."""
import sys, json, statistics as st, math, collections, argparse
ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--ref'); ap.add_argument('--bins', action='store_true')
ap.add_argument('--labels', nargs='*'); ap.add_argument('--top', type=int, default=40)
a = ap.parse_args()
R = {}
for f in a.files:
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        R[(r['label'], r['seed'])] = r
by = collections.defaultdict(dict)
for (l, s), r in R.items():
    if a.labels and l not in a.labels: continue
    by[l][s] = r
def se(x): return st.stdev(x)/math.sqrt(len(x)) if len(x) > 1 else float('nan')
rows = []
for l, d in by.items():
    v = list(d.values()); surv = [r['surv'] for r in v]
    full = sum(1 for r in v if r['surv'] >= 2999.9)
    tr = [r['trees_d'] for r in v if r['surv'] < 2999.9 and r['trees_d'] is not None]
    wall = sum(1 for r in v if r['surv'] < 2999.9 and r['trees_d'] is not None and r['trees_d'] <= 2 and (r['fruits_d'] or 0) <= 5)
    pd = sum(r.get('pdeaths', 0) for r in v)/len(v)
    rows.append((st.mean(r['score'] for r in v), l, len(v), st.mean(surv), se(surv), full, wall, st.median(tr) if tr else float('nan'),
                 st.mean(r['fruit'] for r in v), st.mean(r['peak'] for r in v), pd))
rows.sort(reverse=True)
ref = by.get(a.ref) if a.ref else None
print(f"{'label':<14}{'n':>5}{'score':>8}{'surv':>7}{'se':>5}{'full':>5}{'wall':>5}{'trD':>5}{'fruit':>7}{'peak':>5}{'pdth':>5}  paired vs ref (surv, fruit, wins, n)")
for sc, l, n, m, s, full, wall, trd, fr, pk, pd in rows[:a.top]:
    extra = ''
    if ref is not None and l != a.ref:
        com = [k for k in by[l] if k in ref]
        if len(com) > 1:
            ds = [by[l][k]['surv']-ref[k]['surv'] for k in com]; df = [by[l][k]['fruit']-ref[k]['fruit'] for k in com]
            dsc = [by[l][k]['score']-ref[k]['score'] for k in com]
            extra = f"  {st.mean(ds):+6.0f}±{se(ds):3.0f}  {st.mean(df):+5.1f}  score {st.mean(dsc):+6.0f}±{se(dsc):3.0f}  {sum(1 for x in ds if x > 0)}/{len(com)}"
    print(f"{l:<14}{n:>5}{sc:8.0f}{m:7.0f}{s:5.0f}{full:5d}{wall:5d}{trd:5.1f}{fr:7.1f}{pk:5.0f}{pd:5.1f}{extra}")
if a.bins:
    for l in (a.labels or [r[1] for r in rows[:3]]):
        print(f'\n## {l}: per 250 s bin (runs alive at bin end): alive, trees, fruit spawned, eaten, harvest%, energy/eaten, mean agent energy, runs')
        agg = collections.defaultdict(list)
        for r in by[l].values():
            prev = None
            for row in r['traj']:
                if len(row) < 8: continue
                if prev is not None:
                    agg[row[0]].append((row[1], row[2], row[4]-prev[4], row[5]-prev[5], row[6]-prev[6], row[7]))
                prev = row
        for t in sorted(agg):
            x = agg[t]; sp = sum(y[2] for y in x); ea = sum(y[3] for y in x); en = sum(y[4] for y in x)
            print(f"  t={t:5d} alive {st.mean(y[0] for y in x):5.1f} trees {st.mean(y[1] for y in x):5.1f} spawned {sp/len(x):6.1f} eaten {ea/len(x):6.1f} "
                  f"harvest {100*ea/max(1,sp):5.1f}% e/eaten {en/max(1,ea):5.1f} meanE {st.mean(y[5] for y in x):5.0f} runs {len(x)}")

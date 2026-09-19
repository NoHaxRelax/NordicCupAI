"""Failure diagnosis for predator games from per-death records (run.py with NIGHT_DEATHS=1).
usage: diag_ana.py rows.jsonl [--label rf_ref]
Record: cause(0 starve,1 predator), t, age, e, maxe, speed, sprint, x, y, npred150, dpred, prest, nearwall, pop, npred, evading, old, haspost"""
import sys, json, collections, statistics as st
args = sys.argv[1:]
lab = args[args.index('--label') + 1] if '--label' in args else None
files = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] != '--label')]
rows = [json.loads(l) for f in files for l in open(f)]
rows = [r for r in rows if r.get('deaths') is not None and (lab is None or r['label'] == lab)]
G = len(rows); D = [d for r in rows for d in r['deaths']]
print(f"games {G}  mean survival {st.mean(r['surv'] for r in rows):.0f}  deaths/game {len(D)/G:.0f}  kills/game {sum(1 for d in D if d[0]==1)/G:.1f}  starvation/game {sum(1 for d in D if d[0]==0)/G:.1f}  created/game {st.mean(r['created'] for r in rows):.0f}")
def agecls(a): return 'child<60' if a < 60 else 'adult' if a < 120 else 'old>120'
def twin(t): return '0-500' if t < 500 else '500-1000' if t < 1000 else '1000-1500' if t < 1500 else '1500+'
print("\n== 1. deaths per game by cause x age class")
for cause, name in ((0, 'starvation'), (1, 'predator')):
    c = collections.Counter(agecls(d[2]) for d in D if d[0] == cause)
    print(f"  {name:10s}", {k: round(v / G, 1) for k, v in sorted(c.items())})
print("== 1b. deaths per game by cause x time window (and mean population in that window)")
for w in ('0-500', '500-1000', '1000-1500', '1500+'):
    ks = sum(1 for d in D if d[0] == 1 and twin(d[1]) == w) / G; ss = sum(1 for d in D if d[0] == 0 and twin(d[1]) == w) / G
    pops = [d[13] for d in D if twin(d[1]) == w]
    print(f"  {w:10s} kills {ks:5.1f}  starvation {ss:5.1f}  pop at deaths {st.mean(pops) if pops else 0:5.1f}  games alive at window start {sum(1 for r in rows if r['surv'] >= int(w.split('-')[0].rstrip('+')))}")
K = [d for d in D if d[0] == 1]
print(f"\n== 2. kills ({len(K)/G:.1f} per game) by category (share of kills)")
def share(name, f):
    n = sum(1 for d in K if f(d)); print(f"  {name:55s} {100*n/max(1,len(K)):5.1f}%  ({n/G:.1f}/game)")
share('walk-capped (energy < 20% of max, no sprint allowed)', lambda d: d[3] < 0.2 * d[4])
share('cannot outrun even sprinting (sprint <= 15)', lambda d: d[6] <= 15.0 and d[3] >= 0.2 * d[4])
share('could outrun (sprint > 15, not capped)', lambda d: d[6] > 15.0 and d[3] >= 0.2 * d[4])
share('was evading in the last second (saw it coming)', lambda d: d[15] == 1)
share('NOT evading (ambushed / never reacted)', lambda d: d[15] == 0)
share('2+ predators within 150', lambda d: d[9] >= 2)
share('3+ predators within 150', lambda d: d[9] >= 3)
share('cornered: obstacle within 15', lambda d: d[12] == 1)
share('old agent (senescent)', lambda d: d[16] == 1)
share('child (< 60 s)', lambda d: d[2] < 60)
share('energy at death > 200 (valuable)', lambda d: d[3] > 200)
print(f"  mean energy at death {st.mean(d[3] for d in K):.0f}, mean age {st.mean(d[2] for d in K):.0f}, mean sprint {st.mean(d[6] for d in K):.1f}, mean speed {st.mean(d[5] for d in K):.1f}")
print("  cross: evading x escape capability (share of kills)")
for ev in (1, 0):
    for name, f in (('capped', lambda d: d[3] < 0.2 * d[4]), ('slow', lambda d: d[6] <= 15 and d[3] >= 0.2 * d[4]), ('fast', lambda d: d[6] > 15 and d[3] >= 0.2 * d[4])):
        n = sum(1 for d in K if d[15] == ev and f(d)); print(f"     evading={ev} {name:6s} {100*n/max(1,len(K)):5.1f}%")
S = [d for d in D if d[0] == 0]
print(f"\n== 3. starvation ({len(S)/G:.1f} per game)")
c = collections.Counter(agecls(d[2]) for d in S); print("  by age class", {k: f"{100*v/len(S):.0f}%" for k, v in sorted(c.items())})
print(f"  children starving: mean age {st.mean(d[2] for d in S if d[2] < 60) if any(d[2] < 60 for d in S) else 0:.0f} s; with a post {100*sum(1 for d in S if d[2] < 60 and d[17]) / max(1, sum(1 for d in S if d[2] < 60)):.0f}%")
print(f"  adults/old starving with a post {100*sum(1 for d in S if d[2] >= 60 and d[17]) / max(1, sum(1 for d in S if d[2] >= 60)):.0f}%, old share {100*sum(1 for d in S if d[16]) / len(S):.0f}%")
print("\n== 4. last 300 s of each game: deaths by cause, population 300 s before the end")
last_k = last_s = 0; pop300 = []
for r in rows:
    T = r['surv']; ds = [d for d in r['deaths'] if d[1] >= T - 300]
    last_k += sum(1 for d in ds if d[0] == 1); last_s += sum(1 for d in ds if d[0] == 0)
    first = [d for d in ds if d[1] >= T - 300]
    if first: pop300.append(first[0][13])
print(f"  kills {last_k/G:.1f}/game, starvation {last_s/G:.1f}/game, population 300 s before the end {st.mean(pop300) if pop300 else 0:.1f}, predators at the end {st.mean(r['preds'] for r in rows):.1f}, trees at the end {st.mean(r['trees_d'] or 0 for r in rows):.1f}")
print("\n== 5. economy from the 250 s trajectory: fruit energy eaten per agent-250 s, births (created) per 250 s")
acc = collections.defaultdict(lambda: [0., 0., 0, 0.])
for r in rows:
    tr = r['traj']; prev_e = 0.
    for i, x in enumerate(tr):
        t, alive, trees, preds, fsp, eaten, fe = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        a = acc[t]; a[0] += fe - prev_e; a[1] += alive; a[2] += 1; a[3] += preds; prev_e = fe
for t in sorted(acc):
    a = acc[t]
    if a[2] >= 10: print(f"  t={t:5d}  games {a[2]:3d}  alive {a[1]/a[2]:5.1f}  predators {a[3]/a[2]:4.1f}  fruit energy per agent per 250 s {a[0]/max(1,a[1]):6.1f}")

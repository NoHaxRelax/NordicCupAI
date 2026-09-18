"""Merge optimizer journals and rank configurations. Usage: opt_report.py journal1.jsonl [journal2 ...] [--top 10]"""
import json, sys, statistics, collections
top = int(sys.argv[sys.argv.index('--top')+1]) if '--top' in sys.argv else 10
paths = [p for i, p in enumerate(sys.argv[1:], 1) if not p.startswith('--') and sys.argv[i-1] != '--top']
import glob, pathlib
def credit(r, horizon=3000.):
    """Add tree-wall credit fields to a journal row, from the row itself or its run JSON."""
    if 'eff_score' in r: return r
    T = r['survival']; trees_d = r.get('trees_death'); fruits_d = r.get('fruits_death')
    if trees_d is None and r.get('_runfile'):
        rr = json.load(open(r['_runfile'])); last = [x for x in rr['samples'] if x['time'] <= T+1e-6]
        if last: trees_d, fruits_d = last[-1]['trees'], last[-1]['fruits']
    wall = T < horizon-1e-6 and trees_d is not None and trees_d <= 2 and (fruits_d or 0) <= 5
    r['trees_death'] = trees_d; r['fruits_death'] = fruits_d; r['wall_death'] = wall
    r['eff_survival'] = horizon if wall else T; r['eff_score'] = r['eff_survival']+r['fruit']
    return r
by = collections.OrderedDict()
for path in paths:
    runs_dir = pathlib.Path(path).parent/'runs'
    for line in open(path):
        r = json.loads(line); key = json.dumps(r['cfg'], sort_keys=True)
        if runs_dir.exists():
            cand = list(runs_dir.glob(f"*-{r['label']}-seed{r['seed']}.json"))
            if cand: r['_runfile'] = str(cand[0])
        r = credit(r)
        by.setdefault(key, {'cfg': r['cfg'], 'label': r['label'], 'runs': []})['runs'].append(r)
rows = []
for v in by.values():
    rs = v['runs']
    rows.append(dict(label=v['label'], cfg=v['cfg'], n=len(rs), score=statistics.mean(r['eff_score'] for r in rs),
                     raw=statistics.mean(r['score'] for r in rs),
                     surv=statistics.mean(r['survival'] for r in rs), surv_min=min(r['survival'] for r in rs),
                     fruit=statistics.mean(r['fruit'] for r in rs), full=sum(1 for r in rs if r['survival'] >= 2999),
                     walls=sum(1 for r in rs if r['wall_death']),
                     fail=sum(1 for r in rs if r['survival'] < 2999 and not r['wall_death']),
                     trees_d=statistics.mean(r['trees_death'] for r in rs if r['trees_death'] is not None) if any(r['trees_death'] is not None for r in rs) else float('nan')))
rows.sort(key=lambda r: -r['score'])
print(f"{len(rows)} configurations, {sum(r['n'] for r in rows)} runs; ranking = survival with tree-wall credit + fruit score")
print(f"{'label':<10}{'n':>3}{'effScore':>9}{'rawScore':>9}{'surv':>6}{'min':>6}{'full':>5}{'wall':>5}{'fail':>5}{'trees@d':>8}{'fruit':>7}  cfg")
for r in rows[:top]:
    short = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r['cfg'].items()}
    print(f"{r['label']:<10}{r['n']:>3}{r['score']:>9.1f}{r['raw']:>9.1f}{r['surv']:>6.0f}{r['surv_min']:>6.0f}{r['full']:>5}{r['walls']:>5}{r['fail']:>5}{r['trees_d']:>8.1f}{r['fruit']:>7.1f}  {json.dumps(short)}")
# parameter-wise view: mean score of top quartile vs bottom quartile per numeric parameter
if len(rows) >= 8:
    q = max(2, len(rows)//4); hi, lo = rows[:q], rows[-q:]
    print("\nparameter: mean in top quartile vs bottom quartile (numeric) / share in top vs bottom (choices)")
    for k in rows[0]['cfg']:
        vals_hi = [r['cfg'][k] for r in hi]; vals_lo = [r['cfg'][k] for r in lo]
        if isinstance(vals_hi[0], (int, float)) and not isinstance(vals_hi[0], bool):
            print(f"  {k:<18} top {statistics.mean(vals_hi):8.2f}   bottom {statistics.mean(vals_lo):8.2f}")
        else:
            c_hi = collections.Counter(map(str, vals_hi)); c_lo = collections.Counter(map(str, vals_lo))
            print(f"  {k:<18} top {dict(c_hi)}   bottom {dict(c_lo)}")

"""Summarize a directory of harness result JSONs: per-run table, trajectories, late deaths, metrics.
Usage: analyze.py RESULTS_DIR [--every 300] [--label LABEL]"""
import json, glob, sys, argparse, statistics as st, pathlib, collections
p = argparse.ArgumentParser(); p.add_argument('dir'); p.add_argument('--every', type=float, default=300.); p.add_argument('--label', default=None)
p.add_argument('--traj', action='store_true', help='print per-run trajectories')
p.add_argument('--pair', nargs=2, metavar=('A', 'B'), help='paired per-seed comparison of two labels')
a = p.parse_args()
rows = []
for f in sorted(glob.glob(str(pathlib.Path(a.dir)/'*.json'))):
    if f.endswith('summary.json'): continue
    r = json.load(open(f))
    if a.label and r.get('label') != a.label: continue
    rows.append(r)
by = collections.defaultdict(list)
for r in rows: by[r.get('label', '')].append(r)
def trees_at(r, t):
    pts = [s for s in r['samples'] if s['time'] <= t]
    return pts[-1]['trees'] if pts else None
def t_trees_below(r, n):
    for s in r['samples']:
        if s['trees'] < n: return s['time']
    return None
for label, rs in by.items():
    rs.sort(key=lambda r: r['seed'])
    surv = [r['survival_seconds'] for r in rs]; H = rs[0]['horizon']
    print(f"\n### {label}: n={len(rs)} survival mean {st.mean(surv):.0f} min {min(surv):.0f} max {max(surv):.0f} "
          f"full {sum(1 for x in surv if x >= H-1e-6)}/{len(rs)} score {st.mean(r['score'] for r in rs):.1f} fruit {st.mean(r['fruit_score'] for r in rs):.1f}")
    print(f"{'seed':>5}{'surv':>7}{'score':>8}{'fruit':>7}{'eaten':>7}{'peak':>6}{'made':>6}{'births':>7}{'oldb':>6}{'stuck':>6}{'trDeath':>8}{'t<8tr':>7}{'wall':>6}")
    for r in rs:
        m = r.get('policy_metrics') or {}
        td = trees_at(r, r['survival_seconds']); t8 = t_trees_below(r, 8)
        print(f"{r['seed']:>5}{r['survival_seconds']:>7.0f}{r['score']:>8.1f}{r['fruit_score']:>7.1f}{r['fruit_eaten']:>7}{r['peak_agents']:>6}{r['total_agents_created']:>6}"
              f"{m.get('births', 0):>7}{m.get('old_births', 0):>6}{m.get('stuck_events', 0):>6}{str(td):>8}{('%.0f' % t8) if t8 else '-':>7}{r['wall_seconds']:>6.0f}")
    # survival beyond the moment the map first had fewer than 8 trees: removes most of the tree-supply luck
    extra = [r['survival_seconds']-t_trees_below(r, 8) for r in rs if t_trees_below(r, 8) is not None]
    if extra: print(f"  survival past the <8-trees moment: mean {st.mean(extra):.0f} s over {len(extra)} runs")
    if a.traj:
        for r in rs:
            pts = [s for s in r['samples'] if abs((s['time']/a.every)-round(s['time']/a.every)) < 1e-6]
            print(f"  seed {r['seed']} t/alive/trees/fruits/meanE:", ' '.join(f"{s['time']:.0f}:{s['alive']}/{s['trees']}/{s['fruits']}/{s['mean_energy']:.0f}" for s in pts))
            T = r['survival_seconds']; late = [d for d in r['deaths'] if d['t'] > T-300]
            print(f"     last-300s deaths {len(late)}: young {sum(1 for d in late if d['age'] < 60)} old {sum(1 for d in late if d['age'] >= 60)}")

if a.pair:
    A, B = a.pair; ra = {r['seed']: r for r in by.get(A, [])}; rb = {r['seed']: r for r in by.get(B, [])}
    seeds = sorted(set(ra) & set(rb))
    if seeds:
        d = [rb[s_]['survival_seconds']-ra[s_]['survival_seconds'] for s_ in seeds]
        df = [rb[s_]['fruit_score']-ra[s_]['fruit_score'] for s_ in seeds]
        wins = sum(1 for x in d if x > 0)
        print(f"\n### paired {B} minus {A} on {len(seeds)} seeds: survival mean {st.mean(d):+.0f} s (sd {st.pstdev(d):.0f}, wins {wins}/{len(seeds)}), fruit {st.mean(df):+.1f}")
        print('   per seed:', ' '.join(f"{s_}:{x:+.0f}" for s_, x in zip(seeds, d)))

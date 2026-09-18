"""Summarize a directory of harness result JSONs: per-run table, trajectories, late deaths, metrics.
Usage: analyze.py RESULTS_DIR [--every 300] [--label LABEL]"""
import json, glob, sys, argparse, statistics as st, pathlib, collections
p = argparse.ArgumentParser(); p.add_argument('dir'); p.add_argument('--every', type=float, default=300.); p.add_argument('--label', default=None)
p.add_argument('--traj', action='store_true', help='print per-run trajectories')
a = p.parse_args()
rows = []
for f in sorted(glob.glob(str(pathlib.Path(a.dir)/'*.json'))):
    if f.endswith('summary.json'): continue
    r = json.load(open(f))
    if a.label and r.get('label') != a.label: continue
    rows.append(r)
by = collections.defaultdict(list)
for r in rows: by[r.get('label', '')].append(r)
for label, rs in by.items():
    rs.sort(key=lambda r: r['seed'])
    surv = [r['survival_seconds'] for r in rs]; H = rs[0]['horizon']
    print(f"\n### {label}: n={len(rs)} survival mean {st.mean(surv):.0f} min {min(surv):.0f} max {max(surv):.0f} "
          f"full {sum(1 for x in surv if x >= H-1e-6)}/{len(rs)} score {st.mean(r['score'] for r in rs):.1f} fruit {st.mean(r['fruit_score'] for r in rs):.1f}")
    print(f"{'seed':>5}{'surv':>7}{'score':>8}{'fruit':>7}{'eaten':>7}{'peak':>6}{'made':>6}{'births':>7}{'oldb':>6}{'stuck':>6}{'wall':>6}")
    for r in rs:
        m = r.get('policy_metrics') or {}
        print(f"{r['seed']:>5}{r['survival_seconds']:>7.0f}{r['score']:>8.1f}{r['fruit_score']:>7.1f}{r['fruit_eaten']:>7}{r['peak_agents']:>6}{r['total_agents_created']:>6}"
              f"{m.get('births', 0):>7}{m.get('old_births', 0):>6}{m.get('stuck_events', 0):>6}{r['wall_seconds']:>6.0f}")
    if a.traj:
        for r in rs:
            pts = [s for s in r['samples'] if abs((s['time']/a.every)-round(s['time']/a.every)) < 1e-6]
            print(f"  seed {r['seed']} t/alive/trees/fruits/meanE:", ' '.join(f"{s['time']:.0f}:{s['alive']}/{s['trees']}/{s['fruits']}/{s['mean_energy']:.0f}" for s in pts))
            T = r['survival_seconds']; late = [d for d in r['deaths'] if d['t'] > T-300]
            print(f"     last-300s deaths {len(late)}: young {sum(1 for d in late if d['age'] < 60)} old {sum(1 for d in late if d['age'] >= 60)}")

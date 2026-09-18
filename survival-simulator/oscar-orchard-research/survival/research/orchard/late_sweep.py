"""Late-game experiments from checkpoints: configs x snapshots on a process pool.
Usage: late_sweep.py --snaps DIR --configs '{"label":{"cap_min":3}, ...}' --workers 8 --out DIR [--record]
Each run resumes the engine and controller from the snapshot (taken at 1500 s under the base
configuration), applies the config's parameter overrides to the controller, and runs to 3000 s."""
import os, sys, json, argparse, pathlib, glob, statistics, time
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research'/'orchard'))


def _run(job):
    snap, kw, label, out = job
    import snapshot
    return snapshot.run(snap, kw, label, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snaps', required=True); ap.add_argument('--configs', default='{"base":{}}')
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: configs = json.loads(a.configs)
    except json.JSONDecodeError: configs = json.load(open(a.configs))
    snaps = sorted(glob.glob(str(pathlib.Path(a.snaps)/'seed*.pkl.gz')))
    jobs = [(s, kw, label, a.out) for label, kw in configs.items() for s in snaps]
    print(f"{len(configs)} configs x {len(snaps)} snapshots = {len(jobs)} late-game runs", flush=True)
    from multiprocessing import Pool
    t0 = time.perf_counter(); rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_run, jobs): rows.append(r)
    by = {}
    for r in rows: by.setdefault(r['label'], []).append(r)
    print(f"\n{len(rows)} runs in {time.perf_counter()-t0:.0f}s   (from t={rows[0]['from_t']:.0f})")
    print(f"{'label':<16}{'n':>3}{'surv':>7}{'eff':>7}{'min':>6}{'full':>5}{'walls':>6}{'trees@d':>8}{'score':>8}")
    for label, rs in sorted(by.items(), key=lambda x: -statistics.mean(r['eff_survival'] for r in x[1])):
        print(f"{label:<16}{len(rs):>3}{statistics.mean(r['survival'] for r in rs):>7.0f}{statistics.mean(r['eff_survival'] for r in rs):>7.0f}"
              f"{min(r['survival'] for r in rs):>6.0f}{sum(1 for r in rs if r['survival']>=2999):>5}{sum(1 for r in rs if r['wall_death']):>6}"
              f"{statistics.mean(r['trees_death'] for r in rs):>8.1f}{statistics.mean(r['score'] for r in rs):>8.1f}")
    pathlib.Path(a.out, 'summary.json').write_text(json.dumps(dict(configs=configs, rows=[{k: v for k, v in r.items() if k != 'samples'} for r in rows]), indent=1))

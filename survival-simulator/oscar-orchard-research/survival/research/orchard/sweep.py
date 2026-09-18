"""Parallel no-predator sweeps: configs x seeds in a process pool, summary table at the end.

Usage: sweep.py --configs '{"base":{}, "slots2":{"tree_slots":2}}' --seeds 1 2 3 --horizon 3000 --workers 8 --out results/orchard/sweep-x
Each run is one harness.run call (same JSON output as the society harness) with predators disabled.
"""
import os, sys, json, argparse, pathlib, time, statistics
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research'/'orchard'))


def _run(job):
    label, kw, seed, horizon, out, policy, record = job
    import harness_np  # patches predators off
    import harness
    t0 = time.perf_counter()
    r = harness.run(policy, seed, horizon, out, label, record, False, sample_every=100., **kw)
    return dict(label=label, seed=seed, survival=r['survival_seconds'], score=r['score'], fruit=r['fruit_score'],
                eaten=r['fruit_eaten'], alive=r['alive'], peak=r['peak_agents'], created=r['total_agents_created'],
                wall=round(time.perf_counter()-t0, 1), metrics=r.get('policy_metrics'))


def summarize(rows, horizon):
    by = {}
    for r in rows: by.setdefault(r['label'], []).append(r)
    print(f"{'label':<14}{'n':>3}{'surv_mean':>11}{'surv_min':>10}{'full':>6}{'score':>9}{'fruit':>8}{'eaten':>8}{'peak':>6}{'wall':>7}")
    for label, rs in by.items():
        surv = [r['survival'] for r in rs]; full = sum(1 for x in surv if x >= horizon-1e-6)
        print(f"{label:<14}{len(rs):>3}{statistics.mean(surv):>11.1f}{min(surv):>10.1f}{full:>4}/{len(rs):<2}"
              f"{statistics.mean(r['score'] for r in rs):>9.1f}{statistics.mean(r['fruit'] for r in rs):>8.1f}"
              f"{statistics.mean(r['eaten'] for r in rs):>8.0f}{statistics.mean(r['peak'] for r in rs):>6.0f}{statistics.mean(r['wall'] for r in rs):>7.0f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--configs', default='{"base":{}}')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--horizon', type=float, default=3000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--policy', default='orchard:OrchardPolicy')
    p.add_argument('--out', required=True)
    p.add_argument('--record', action='store_true', help='state-only replay per run (fallback for large sweeps, not organizer rendering)')
    a = p.parse_args()
    configs = json.loads(a.configs)
    if isinstance(configs, str): configs = json.load(open(configs))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(label, kw, seed, a.horizon, str(out), a.policy, a.record) for label, kw in configs.items() for seed in a.seeds]
    from multiprocessing import Pool
    t0 = time.perf_counter(); rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_run, jobs):
            rows.append(r); print(json.dumps({k: v for k, v in r.items() if k != 'metrics'}), flush=True)
    (out/'summary.json').write_text(json.dumps(dict(configs=configs, seeds=a.seeds, horizon=a.horizon, rows=rows), indent=1))
    print(f"\n{len(rows)} runs in {time.perf_counter()-t0:.0f}s")
    summarize(rows, a.horizon)

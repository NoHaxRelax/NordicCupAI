"""Parallel batch: society vs trapper over seeds x repeats, with a summary table.

    ../.venv/bin/python scripts/trapper/batch.py --seeds 1 8 --repeats 2 --seconds 600 --workers 4 --label v1

Each run is a separate process (the engine keeps no global state, but this also
isolates crashes). Results are the per-run JSON files from run_game.run plus a
batch summary under results/trapper/batches/.
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'trapper'))


def one(args):
    seed, trap, seconds, label, params, world = args
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import run_game
    r = run_game.run(seed, seconds, trap, record=False, native=False, label=label, verbose=False, params=params, world=world)
    keep = {k: r[k] for k in ('mode', 'seed', 'survival', 'score', 'alive', 'peak', 'predators', 'predator_deaths',
                               'starvation_deaths', 'predator_penalty', 'held_fraction', 'wall_seconds')}
    if trap:
        keep['trap'] = r['metrics']
        keep['role_deaths'] = r['role_deaths']
    return keep


def summarize(rows):
    out = {}
    for mode in ('society', 'trapper'):
        rs = [r for r in rows if r['mode'] == mode]
        if not rs:
            continue
        sc = [r['score'] for r in rs]
        out[mode] = dict(runs=len(rs), score_mean=round(statistics.mean(sc), 1), score_sd=round(statistics.pstdev(sc), 1),
                         survival_mean=round(statistics.mean(r['survival'] for r in rs), 1),
                         extinct=sum(r['alive'] == 0 for r in rs),
                         predator_deaths=round(statistics.mean(r['predator_deaths'] for r in rs), 1),
                         starvation=round(statistics.mean(r['starvation_deaths'] for r in rs), 1),
                         penalty=round(statistics.mean(r['predator_penalty'] for r in rs), 1),
                         held=round(statistics.mean(r['held_fraction'] or 0 for r in rs), 3))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs=2, default=[1, 6])
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--seconds', type=float, default=600)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--label', default='batch')
    ap.add_argument('--modes', default='both')
    ap.add_argument('--params', default='{}')
    ap.add_argument('--preset', default=None, help='named parameter set (avoids JSON quoting over ssh)')
    ap.add_argument('--world', default='oracle', help="'oracle' (full knowledge) or 'estimator' (observations only)")
    a = ap.parse_args()
    params = json.loads(a.params)
    PRESETS = {
        'v5': dict(max_turn=0.8727, turn_bonus=0.8727, lead_max=900.0, gap_reserve=False),
        'v6': dict(),
        'noreserve': dict(gap_reserve=False),
        'wideturn': dict(max_turn=0.8727, turn_bonus=0.8727, lead_max=900.0),
        'nodeliver': dict(explicit_deliveries=False),      # refuge runs and flybys only
        'shortlead': dict(max_turn=0.5236, turn_bonus=0.35, lead_max=500.0),
    }
    if a.preset:
        params = dict(PRESETS[a.preset], **params)
    jobs = []
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        for rep in range(a.repeats):
            if a.modes in ('both', 'society'):
                jobs.append((seed, False, a.seconds, f'{a.label}-r{rep}', params, a.world))
            if a.modes in ('both', 'trapper'):
                jobs.append((seed, True, a.seconds, f'{a.label}-r{rep}', params, a.world))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, jobs))
    summ = summarize(rows)
    out = ROOT / 'results' / 'trapper' / 'batches'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{a.label}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}.json'
    path.write_text(json.dumps(dict(args=vars(a), summary=summ, rows=rows), indent=1))
    print(json.dumps(summ, indent=1))
    # paired per-seed table
    by = {}
    for r in rows:
        by.setdefault((r['seed'], r['mode']), []).append(r['score'])
    print('seed  society(mean)  trapper(mean)  diff')
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        s = by.get((seed, 'society')); t = by.get((seed, 'trapper'))
        if s and t:
            print(f'{seed:4d}  {statistics.mean(s):9.1f}      {statistics.mean(t):9.1f}   {statistics.mean(t)-statistics.mean(s):+7.1f}')
    print(f'wall time {time.time()-t0:.0f}s; saved {path}')

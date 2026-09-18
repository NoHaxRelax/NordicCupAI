"""Paired 600-second games: society, stock trapper, dodge trapper, dodge trapper with the low
energy gate, on the same seeds, in parallel processes.

    ../.venv/bin/python scripts/trapper/batch_dodge.py --seeds 1 8 --seconds 600 --workers 8 --label dodge-v1
"""
import argparse, json, os, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts' / 'trapper'))


def one(args):
    seed, guide, low_gate, seconds, label, record = args
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import run_game_dodge
    r = run_game_dodge.run(seed, seconds, guide, low_gate, record=record, native=False, label=label)
    keep = {k: r[k] for k in ('guide', 'mode', 'seed', 'survival', 'score', 'alive', 'peak', 'predators', 'predator_deaths',
                               'starvation_deaths', 'predator_penalty', 'held_fraction', 'wall_seconds')}
    if r.get('metrics'):
        keep['trap'] = {k: v for k, v in r['metrics'].items() if k.startswith('trap_')}
        keep['role_deaths'] = r['role_deaths']
    if 'dodge' in r:
        keep['dodge'] = r['dodge']
    keep['replay'] = r.get('replay')
    return keep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs=2, default=[1, 8]); ap.add_argument('--seconds', type=float, default=600)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--label', default='dodge')
    ap.add_argument('--variants', default='society,stock,dodge,dodge-lowgate'); ap.add_argument('--record', action='store_true')
    a = ap.parse_args()
    jobs = []
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        for v in a.variants.split(','):
            guide = 'society' if v == 'society' else ('stock' if v == 'stock' else 'dodge')
            jobs.append((seed, guide, v == 'dodge-lowgate', a.seconds, a.label, a.record))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, jobs))
    out = ROOT / 'results' / 'trapper' / 'batches'; out.mkdir(parents=True, exist_ok=True)
    path = out / f'{a.label}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}.json'
    summary = {}
    for v in a.variants.split(','):
        rs = [r for r in rows if r['guide'] == v]
        if not rs: continue
        summary[v] = dict(runs=len(rs), score_mean=round(statistics.mean(r['score'] for r in rs), 1),
                          survival_mean=round(statistics.mean(r['survival'] for r in rs), 1), extinct=sum(r['alive'] == 0 for r in rs),
                          predator_deaths=round(statistics.mean(r['predator_deaths'] for r in rs), 1),
                          held=round(statistics.mean(r['held_fraction'] or 0 for r in rs), 3),
                          deliveries=sum(r.get('trap', {}).get('trap_deliveries', 0) for r in rs),
                          delivered=sum(r.get('trap', {}).get('trap_delivered', 0) for r in rs),
                          guide_lost=sum(r.get('trap', {}).get('trap_guide_lost', 0) for r in rs),
                          guide_deaths=sum((r.get('role_deaths') or {}).get('guide', 0) for r in rs))
    path.write_text(json.dumps(dict(args=vars(a), summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
    print('seed ' + ' '.join(f'{v:>14s}' for v in a.variants.split(',')))
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        print(f'{seed:4d} ' + ' '.join(f"{next((r['score'] for r in rows if r['seed'] == seed and r['guide'] == v), float('nan')):14.1f}" for v in a.variants.split(',')))
    print(f'wall {time.time() - t0:.0f}s; saved {path}')

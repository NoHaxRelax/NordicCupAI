"""Full generated-map games with the turning-circle dodge guide (oracle world).

    ../.venv/bin/python scripts/trapper/run_game_dodge.py --seeds 1 2 --seconds 600 --guide dodge [--low-gate] [--record]

Same runner as run_game.py; ``--guide dodge`` installs models/trapper/dodge.DodgeLure for every
delivery, ``--low-gate`` also lets guides from 160 energy lead (DodgeManager). Results go to
results/trapper/ like the stock runner, labelled with the guide variant.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts' / 'trapper'))


def run(seed, seconds, guide='stock', low_gate=False, record=False, native=False, label='', params=None):
    import run_game
    if guide == 'dodge':
        from models.trapper.dodge import install, DodgeLure
        install(low_gate=low_gate)
        DodgeLure.metrics.update(dodge_ticks=0, dodge_sprint_ticks=0, dodge_energy=0.0)
    tag = 'society' if guide == 'society' else (f'dodge{"-lowgate" if low_gate else ""}' if guide == 'dodge' else 'stock')
    r = run_game.run(seed, seconds, guide != 'society', record=record, native=native, label=(label + '-' if label else '') + tag, params=params)
    r['guide'] = tag
    if guide == 'dodge':
        from models.trapper.dodge import DodgeLure
        r['dodge'] = dict(DodgeLure.metrics)
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1]); ap.add_argument('--seconds', type=float, default=600)
    ap.add_argument('--guide', choices=['stock', 'dodge', 'society'], default='dodge'); ap.add_argument('--low-gate', action='store_true')
    ap.add_argument('--record', action='store_true'); ap.add_argument('--native', action='store_true'); ap.add_argument('--label', default='')
    ap.add_argument('--params', default='{}')
    a = ap.parse_args()
    for seed in a.seeds:
        r = run(seed, a.seconds, a.guide, a.low_gate, a.record, a.native, a.label, json.loads(a.params))
        print(json.dumps({k: r[k] for k in ('guide', 'seed', 'survival', 'score', 'predator_deaths', 'held_fraction', 'wall_seconds')} | ({'dodge': r['dodge']} if 'dodge' in r else {})))

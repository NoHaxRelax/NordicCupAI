"""Leash arena (scripts/trapper/test_leash.py) driven by the turning-circle dodge guide instead of
the stock leash. Same seeds, obstacles, placements and success criterion, so the two are paired.

    ../.venv/bin/python scripts/trapper/test_leash_dodge.py --cases 24 --second 0 --seconds 60 [--stock]
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts' / 'trapper'))
import test_leash as tl                                   # noqa: E402
from models.trapper.dodge import DodgeLure                # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=24); ap.add_argument('--second', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=60); ap.add_argument('--obstacles', type=int, default=6)
    ap.add_argument('--staffed', type=int, default=-1); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--energy', type=float, default=500); ap.add_argument('--verbose-seed', type=int, default=None)
    ap.add_argument('--stock', action='store_true', help='run the stock leash instead (paired baseline)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if not a.stock:
        tl.Lure = DodgeLure
    label = 'stock leash' if a.stock else 'dodge guide'
    if a.verbose_seed is not None:
        staffed = (a.verbose_seed % 2 == 0) if a.staffed < 0 else bool(a.staffed)
        r = tl.run_case(a.verbose_seed, a.obstacles, staffed, bool(a.second), a.seconds, verbose=True, energy=a.energy)
        print(r); sys.exit(0)
    ok = 0; fails = []; results = []; t0 = time.time()
    for seed in range(a.start, a.start + a.cases):
        staffed = (seed % 2 == 0) if a.staffed < 0 else bool(a.staffed)
        DodgeLure.metrics.update(dodge_ticks=0, dodge_sprint_ticks=0, dodge_energy=0.0)
        r = tl.run_case(seed, a.obstacles, staffed, bool(a.second), a.seconds, energy=a.energy)
        r.update(dodge=dict(DodgeLure.metrics), staffed=staffed, label=label)
        ok += r['ok']; results.append(r)
        print(f"seed {seed:3d} staffed={int(staffed)} chasing={int(r['chasing'])}: {'OK   ' if r['ok'] else 'FAIL '} t={r.get('t')} {r.get('why','')} "
              f"{('used=' + str(r.get('used')) + ' e_left=' + str(r.get('guide_energy'))) if r['ok'] else ('| ' + str(r.get('last', '')))[:110]} sprint_ticks={r['dodge']['dodge_sprint_ticks']}", flush=True)
        if not r['ok']:
            fails.append(seed)
    print(f'{label}: success {ok}/{a.cases}; failed seeds {fails}; wall {time.time()-t0:.0f}s')
    used = sorted(r['used'] for r in results if r.get('ok') and r.get('used') is not None)
    if used:
        print(f'energy used: mean {sum(used)/len(used):.0f}, median {used[len(used)//2]}, p90 {used[int(len(used)*0.9)]}, max {used[-1]}; '
              f'time to hold: median {sorted(r["t"] for r in results if r.get("ok"))[len(used)//2]}')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(args=vars(a), label=label, results=results), indent=1))

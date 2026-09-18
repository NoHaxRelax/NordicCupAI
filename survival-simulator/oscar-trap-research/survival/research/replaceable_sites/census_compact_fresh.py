"""Frozen 512-map census of staged selection plus the compact fallback.

Map seeds30000–30511 are fresh geometry-only cases. This does not run actors,
change simulator physics, or measure predator delivery. Complete native static
rectangles are retained so every eligibility decision can be inspected.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research'), str(ROOT / 'vendor/survival-simulator')]
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
from src.core import SimulationCore
from replaceable_sites.selector import enumerate_sites
from replaceable_sites.selector_compact import enumerate_sites as compact_sites

OUT = ROOT / 'results/replaceable_sites/compact_fresh512'


def one(seed):
    if Path('/tmp/predator-intake-stop').exists():
        return dict(seed=seed, status='not_started_stop', supported=False)
    env = SimulationCore(starting_agents=0, starting_predators=0, seed=seed).env
    static = dict(width=env.width, height=env.height, obstacles=[
        dict(x=o.x, y=o.y, width=o.width, height=o.height) for o in env.obstacles])
    sites, stage = [], None
    for index, (gap, overlap) in enumerate(((10.9, 20.), (10.1, 20.), (10.1, 10.3))):
        sites = enumerate_sites(static, min_gap=gap, min_overlap=overlap)
        if sites:
            stage = index
            break
    if not sites:
        sites = compact_sites(static)
        if sites:
            stage = 3
    row = dict(seed=seed, status='complete', supported=bool(sites), stage=stage,
               sites=len(sites),
               distinct_corridors=len({tuple(sorted(s['obstacle_indices'])) for s in sites}),
               boundary_supported=any(s['boundary_indices'] for s in sites),
               static_map=static, eligible_sites=sites,
               scope='static geometry only; no predator delivery test')
    (OUT / f'map-{seed}.json').write_text(json.dumps(row, indent=2) + '\n')
    return {k: v for k, v in row.items() if k not in ('static_map', 'eligible_sites')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = OUT / 'PLAN.json'
    if plan.exists():
        raise SystemExit('immutable census already started')
    sources = [Path(__file__), ROOT / 'research/replaceable_sites/selector.py',
               ROOT / 'research/replaceable_sites/selector_compact.py',
               ROOT / 'vendor/survival-simulator/src/elements/environment.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    seeds = list(range(30000, 30512))
    plan.write_text(json.dumps(dict(seeds=seeds, source_sha256=hashes, workers=2,
        depth=5, radius=5.01, tiers=[[10.9, 20], [10.1, 20], [10.1, 10.3], 'compact'],
        scope='fresh geometry census; full rectangles and eligible sites saved; no steps'),
        indent=2) + '\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, seeds))
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
               for path, digest in hashes.items()), 'frozen census sources changed'
    summary = dict(maps=len(rows), complete=sum(r['status'] == 'complete' for r in rows),
        supported=sum(r['supported'] for r in rows),
        boundary_supported=sum(r.get('boundary_supported', False) for r in rows),
        compact_fallback_supported=sum(r.get('stage') == 3 for r in rows), rows=rows,
        limitation='Static replaceable-gap availability is not guiding reliability.')
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}))


if __name__ == '__main__':
    main()

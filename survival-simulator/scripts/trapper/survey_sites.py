"""Count usable wall and gap trap sites on generated maps (geometry only, no simulation).

    ../.venv/bin/python scripts/trapper/survey_sites.py --seeds 1 30
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import SimulationCore                                   # noqa: E402
from models.trapper.geometry import Rect                               # noqa: E402
from models.trapper.sites import find_wall_sites, find_gap_sites      # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs=2, default=[1, 20])
    a = ap.parse_args()
    walls = gaps = maps_wall = maps_gap = maps_any = 0
    rows = []
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        env = SimulationCore(seed=seed).env
        rects = [Rect(o.x, o.y, o.width, o.height) for o in env.obstacles]
        w = find_wall_sites(rects, env.width, env.height)
        g = find_gap_sites(rects, env.width, env.height)
        walls += len(w); gaps += len(g)
        maps_wall += bool(w); maps_gap += bool(g); maps_any += bool(w or g)
        rows.append((seed, len(w), len(g), [round(s.thickness, 1) for s in w], [round(s.thickness, 1) for s in g]))
        print(f'seed {seed:4d}: walls {len(w)} (thick {[round(s.thickness,1) for s in w]}) gaps {len(g)} (gap {[round(s.thickness,1) for s in g]}, len {[round(s.length) for s in g]})', flush=True)
    n = a.seeds[1] - a.seeds[0] + 1
    print(f'maps with wall site: {maps_wall}/{n}, with gap site: {maps_gap}/{n}, with any: {maps_any}/{n}; mean walls {walls/n:.2f}, mean gaps {gaps/n:.2f}')

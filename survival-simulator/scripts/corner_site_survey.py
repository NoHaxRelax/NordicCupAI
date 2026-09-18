"""Survey opt-in, contact-excluding short corner pockets on native maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def survey(seed):
    from src.core import SimulationCore
    from models.entrapment.entrapment_sites import enumerate_corner_sites
    from models.entrapment.observed_trap_sites import our_sites
    env = SimulationCore(seed=seed, starting_agents=0, starting_trees=0, starting_fruits=1).env
    static = dict(width=env.width, height=env.height,
                  obstacles=[(o.x,o.y,o.width,o.height) for o in env.obstacles])
    corners = enumerate_corner_sites(static)
    return dict(seed=seed, crevices=len(our_sites(static)), corners=corners,
                static=static if corners else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps', type=int, default=100)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a new output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(survey, range(args.maps)))
    args.output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(maps=len(rows),maps_with_corners=sum(bool(r['corners']) for r in rows),
                         extra_maps=sum(bool(r['corners']) and not r['crevices'] for r in rows))),flush=True)

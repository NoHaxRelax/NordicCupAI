"""Survey native maps for safe bait pockets missed by pairwise site detection."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union, nearest_points


def components(geometry, minimum_area=0.02):
    if geometry.is_empty:
        return []
    parts = list(geometry.geoms) if geometry.geom_type in ("MultiPolygon", "GeometryCollection") else [geometry]
    return [g for g in parts if g.geom_type == "Polygon" and g.area >= minimum_area]


def free_space(width, height, rects, radius):
    arena = box(radius, radius, width-radius, height-radius)
    blocked = unary_union([box(x-radius, y-radius, x+w+radius, y+h+radius)
                           for x, y, w, h in rects])
    return arena.difference(blocked)


def candidate_point(poly):
    p = poly.representative_point()
    # Bias away from the safe-region boundary when it has meaningful interior.
    eroded = poly.buffer(-0.05)
    return (eroded.representative_point() if not eroded.is_empty else p)


def survey(seed):
    from src.core import SimulationCore
    from models.entrapment.observed_trap_sites import our_sites
    env = SimulationCore(seed=seed, starting_agents=0, starting_trees=0, starting_fruits=1).env
    rects = [(o.x, o.y, o.width, o.height) for o in env.obstacles]
    static = dict(width=env.width, height=env.height, obstacles=rects)
    ordinary = our_sites(static)
    agent_free = free_space(env.width, env.height, rects, 5.01)
    predator_free = free_space(env.width, env.height, rects, 10.01)
    # All points closer than native kill contact to any predator-valid centre
    # are excluded. The surviving set is globally contact-safe.
    safe = agent_free.difference(predator_free.buffer(15.05, resolution=24))
    agent_components = components(agent_free, 1.0)
    predator_components = components(predator_free, 1.0)
    rows = []
    for region in components(safe):
        p = candidate_point(region)
        xy = (p.x, p.y)
        distance = p.distance(predator_free)
        if distance < 15.049:
            continue
        front_on_bait, front = nearest_points(p, predator_free)
        front_xy = (front.x, front.y)
        front_distance = math.dist(xy, front_xy)
        if front_distance > 40.0:
            continue
        containing = next((c for c in agent_components if c.covers(p)), None)
        front_component = next((c for c in predator_components if c.covers(front)), None)
        if containing is None or front_component is None:
            continue
        # Seek a rear point at least 25 units from bait and predominantly away
        # from the nearest predator component. Straight access is stronger than
        # mere connectedness and matches the replacement fixture.
        vx, vy = xy[0]-front_xy[0], xy[1]-front_xy[1]
        norm = math.hypot(vx, vy) or 1.0
        rear = None
        for distance_back in (25., 35., 50., 70.):
            for angular in (0., .25, -.25, .5, -.5):
                c, s = math.cos(angular), math.sin(angular)
                ux, uy = (vx*c-vy*s)/norm, (vx*s+vy*c)/norm
                q = (xy[0]+ux*distance_back, xy[1]+uy*distance_back)
                if (containing.covers(Point(q)) and containing.covers(LineString([xy, q]))
                        and not front_component.covers(Point(q))):
                    rear = q
                    break
            if rear: break
        if rear is None:
            continue
        nearest_ordinary = min((math.dist(xy, site['goal']) for site in ordinary), default=None)
        rows.append(dict(goal=list(xy), front=list(front_xy), rear=list(rear),
                         minimum_predator_distance=distance,
                         front_distance=front_distance, safe_area=region.area,
                         nearest_ordinary_goal=nearest_ordinary,
                         genuinely_new=nearest_ordinary is None or nearest_ordinary > 20.0))
    rows.sort(key=lambda r: (not r['genuinely_new'], -r['safe_area']))
    return dict(seed=seed, ordinary_sites=len(ordinary), candidates=rows,
                obstacles=rects if rows else None,
                width=env.width, height=env.height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('seeds', type=int, nargs='+')
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(survey, args.seeds))
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(dict(maps=len(rows), maps_with_candidates=sum(bool(r['candidates']) for r in rows),
                          no_ordinary_with_candidates=sum(not r['ordinary_sites'] and bool(r['candidates']) for r in rows),
                          candidates=sum(len(r['candidates']) for r in rows)), indent=2))
    for row in rows:
        print(row['seed'], 'ordinary', row['ordinary_sites'], 'candidates', len(row['candidates']),
              'new', sum(c['genuinely_new'] for c in row['candidates']))


if __name__ == '__main__':
    main()

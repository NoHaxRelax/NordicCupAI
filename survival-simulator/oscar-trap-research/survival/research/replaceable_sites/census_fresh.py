"""Fresh static native-map geometry census; no simulation steps or actors."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'research'), str(ROOT/'vendor/survival-simulator')]
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORTPT','1')
from src.core import SimulationCore
from replaceable_sites.selector import enumerate_sites

OUT = ROOT/'results/replaceable_sites/fresh128'


def one(seed):
    if Path('/tmp/predator-intake-stop').exists():
        return dict(seed=seed, status='not_started_stop', supported=False)
    env = SimulationCore(starting_agents=0, starting_predators=0, seed=seed).env
    static = dict(width=env.width, height=env.height, obstacles=[
        dict(x=o.x,y=o.y,width=o.width,height=o.height) for o in env.obstacles])
    levels = ((10.9,20.), (10.1,20.), (10.1,10.3))
    stage = None
    sites = []
    for index, (gap, overlap) in enumerate(levels):
        sites = enumerate_sites(static,min_gap=gap,min_overlap=overlap)
        if sites:
            stage = index
            break
    row = dict(seed=seed,status='complete',supported=bool(sites),stage=stage,
        sites=len(sites),boundary_supported=any(s['boundary_indices'] for s in sites),
        static_map=static,scope='static geometry only; no predator delivery test')
    (OUT/f'map-{seed}.json').write_text(json.dumps(row,indent=2)+'\n')
    return {k:v for k,v in row.items() if k!='static_map'}


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    plan=OUT/'PLAN.json'
    if plan.exists():
        raise SystemExit('immutable census already started')
    seeds=list(range(20000,20128))
    selector=ROOT/'research/replaceable_sites/selector.py'
    plan.write_text(json.dumps(dict(seeds=seeds,selector_sha256=hashlib.sha256(selector.read_bytes()).hexdigest(),
        workers=2,depth=5,radius=5.01,tiers=[[10.9,20],[10.1,20],[10.1,10.3]],
        scope='fresh geometry census; full rectangles saved; no simulation steps'),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(one,seeds))
    summary=dict(maps=len(rows),supported=sum(r['supported'] for r in rows),
        boundary_supported=sum(r.get('boundary_supported',False) for r in rows),rows=rows,
        limitation='Static replaceable-gap availability is not guiding reliability.')
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))

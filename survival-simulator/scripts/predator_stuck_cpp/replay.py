"""Controlled replays of a recorded trapped predator; the baseline batch is unchanged."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import predator_stuck_scan as scan
from predator_stuck_cpp import _stuck,make_engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with gzip.open(args.trace,'rt',encoding='utf-8') as f:
        event=json.load(f)
    engine=make_engine(event['seed'],100)
    _stuck.ensure_predators(engine,100)
    variants=[('unchanged',0.,0.,0.,-1)]
    for deg in (-45,-10,-1,1,10,45,90,180):
        variants.append((f'heading {deg:+g} deg',0.,0.,math.radians(deg),-1))
    for shift in (-1.,-.1,.1,1.):
        variants.extend([(f'x {shift:+g}',shift,0.,0.,-1),(f'y {shift:+g}',0.,shift,0.,-1)])
    nearest=event.get('detection_geometry',{}).get('nearest_obstacles',[])[:3]
    for obstacle in nearest:
        oid=obstacle['obstacle_id']
        variants.append((f'remove obstacle {oid}',0.,0.,0.,oid))
    px,py=event['end_position']
    obstacles=engine.obstacles()
    distant=max(range(4,len(obstacles)),key=lambda i:(obstacles[i][0]+obstacles[i][2]/2-px)**2+(obstacles[i][1]+obstacles[i][3]/2-py)**2)
    variants.append((f'remove distant obstacle {distant} (control)',0.,0.,0.,distant))
    tick=event['samples'][-1][0]
    results=_stuck.probe_pose(engine,event['predator_id'],tick,round(args.seconds/.1),[tuple(v[1:]) for v in variants])
    # Baseline checkpoint must match the recorded event before interpreting variants.
    expected=event['samples'][-1]
    actual=results[0]['path'][0]
    if actual != tuple(expected):
        raise RuntimeError('Checkpoint differs from saved trace; replay on the source runtime/CPU before drawing conclusions')
    for variant,result in zip(variants,results):
        result['variant']=variant[0]
        result['intervention']=dict(dx=variant[1],dy=variant[2],heading_delta_radians=variant[3],remove_obstacle=variant[4])
        result['first_exit_seconds']=result['first_exit_tick']*.1 if result['first_exit_tick']>=0 else None
        print(json.dumps({k:v for k,v in result.items() if k!='path'}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scan.write_json(args.output,dict(seed=event['seed'],predator_id=event['predator_id'],checkpoint_tick=tick,
        seconds=args.seconds,escape_radius=15,results=results,
        sources=scan.source_hashes('cpp'),build=json.loads((Path(__file__).parent/'build-info.json').read_text()),
        replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Each variant starts from the same complete world/RNG checkpoint. Position variants recenter the test circle at the perturbed pose. Invalid overlapping poses are marked. Only one obstacle is removed per geometry variant. This tests the recorded case, not a universal rule.'))


if __name__=='__main__':
    main()

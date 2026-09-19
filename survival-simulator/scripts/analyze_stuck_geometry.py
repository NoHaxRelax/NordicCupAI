"""Offline geometry study of the frozen census; never used as policy input."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,pathlib,sys,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastsim.fastpolicy import PolicySimulationCore

def one(row):
    sim=PolicySimulationCore(seed=row['seed'],predators=True)
    rect=np.array(sim._engine.obstacles())
    def distances(xy):
        delta=np.maximum(np.maximum(rect[None,:,:2]-xy[:,None,:],xy[:,None,:]-rect[None,:,:2]-rect[None,:,2:]),0)
        return np.sort(np.linalg.norm(delta,axis=2),axis=1)[:,:2]
    boxes=np.array([c['bbox'] for c in row['cases']]).reshape(-1,4)
    nearest=distances((boxes[:,:2]+boxes[:,2:])/2)
    rng=np.random.default_rng(row['seed'])
    control=rng.uniform([30,30],[1570,1170],size=(500,2));control=distances(control)
    control=control[control[:,0]>=10.]
    return dict(seed=row['seed'],cases=[dict(duration=c['duration'],nearest=d.tolist()) for c,d in zip(row['cases'],nearest)],
                control_count=len(control),control_two35=int(np.sum(control[:,1]<35.)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--games',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    rows=[json.loads(s) for s in a.games.read_text().splitlines()]
    with ProcessPoolExecutor(a.workers) as pool:results=list(pool.map(one,rows))
    cases=[c for r in results for c in r['cases']]
    report=dict(maps=len(rows),cases=len(cases),two35=sum(c['nearest'][1]<35 for c in cases),
                long_cases=sum(c['duration']>=300 for c in cases),long_two35=sum(c['duration']>=300 and c['nearest'][1]<35 for c in cases),
                control_count=sum(r['control_count'] for r in results),control_two35=sum(r['control_two35'] for r in results),
                median_nearest=np.median([c['nearest'] for c in cases],axis=0).tolist())
    a.out.write_text(json.dumps(dict(summary=report,maps=results),indent=2));print(json.dumps(report))
if __name__=='__main__':main()

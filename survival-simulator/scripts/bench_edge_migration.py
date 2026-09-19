"""Small paired full-game experiment; spectator confinement never enters policy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pathlib,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastsim.fastpolicy import PolicySimulationCore
from find_stuck_predators import episodes
BASE=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
ARMS={'baseline':{},'edge_bias':{'trap_bias':40.},'edge_protect':{'trap_bias':40.,'trap_protect':2.}}
ARMS.update(gentle={'trap_bias':10.,'trap_protect':2.,'trap_active_r':70.},
            urgent={'trap_bias':80.,'trap_protect':2.,'trap_active_r':70.})

def run(job):
    seed,arm=job;cfg={**BASE,**ARMS[arm]};start=time.monotonic()
    sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,cfg);e=sim._engine
    ts=[];tracks=[]
    while True:
        info=e.info();ts.append(info['time']);tracks.append(e.predators())
        if not e.agents() or info['time']>=2999.999:break
        sim.run_policy(3000.,len(ts)*1.)
    n=max(map(len,tracks));arr=np.full((len(ts),n,2),np.nan)
    for i,row in enumerate(tracks):
        if row:arr[i,:len(row)]=np.asarray(row)[:,:2]
    total=0.;count=0;ts=np.array(ts)
    for j in range(n):
        ok=np.isfinite(arr[:,j,0]);tt=ts[ok]
        for l,r in episodes(tt,arr[ok,j]):total+=tt[r]-tt[l];count+=1
    return dict(seed=seed,arm=arm,score=info['score'],seconds=info['time'],confined_seconds=total,
                confinement_fraction=total/(np.isfinite(arr[:,:,0]).sum()),episodes=count,wall_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--count',type=int,default=16);p.add_argument('--seed',type=int,default=21001)
    p.add_argument('--arms',default='baseline,edge_bias,edge_protect')
    p.add_argument('--workers',type=int,default=12);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    arms={name:ARMS[name] for name in a.arms.split(',')}
    if (a.out/'manifest.json').exists():raise RuntimeError('Output already exists')
    (a.out/'manifest.json').write_text(json.dumps(dict(base=BASE,arms=arms,seeds=list(range(a.seed,a.seed+a.count)),
        build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text())),indent=2))
    with (a.out/'games.jsonl').open('w',buffering=1) as f,ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(run,[(s,arm) for s in range(a.seed,a.seed+a.count) for arm in arms]):
            f.write(json.dumps(r)+'\n');print(r['seed'],r['arm'],round(r['score']),flush=True)
    (a.out/'complete').write_text('done\n')
if __name__=='__main__':main()

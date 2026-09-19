"""Spectator-only trajectory census; unchanged expanded_local_food policy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import pathlib,sys,json,time,argparse,hashlib
from multiprocessing import Pool
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastsim.fastpolicy import PolicySimulationCore
CFG=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
OUT=None

def episodes(times,positions,width=40.,seconds=60.):
    """Maximal windows bounded in both axes, merged when qualifying windows overlap."""
    from collections import deque
    lo=[deque(),deque()];hi=[deque(),deque()];left=0;spans=[]
    for right,xy in enumerate(positions):
        for a in range(2):
            while lo[a] and positions[lo[a][-1],a]>=xy[a]:lo[a].pop()
            while hi[a] and positions[hi[a][-1],a]<=xy[a]:hi[a].pop()
            lo[a].append(right);hi[a].append(right)
        while any(positions[hi[a][0],a]-positions[lo[a][0],a]>width for a in range(2)):
            left+=1
            for a in range(2):
                while lo[a][0]<left:lo[a].popleft()
                while hi[a][0]<left:hi[a].popleft()
        if times[right]-times[left]>seconds:
            # Keep individual maximal stationary boxes; overlapping drifting boxes must not merge.
            if spans and spans[-1][0]==left:spans[-1]=(left,right)
            else:spans.append((left,right))
    # One representative maximal interval per overlapping episode.
    selected=[]
    for l,r in sorted(spans,key=lambda lr:times[lr[1]]-times[lr[0]],reverse=True):
        if not any(l<=rr and r>=ll for ll,rr in selected):selected.append((l,r))
    return sorted(selected)

def one(seed):
    start=time.monotonic();sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,CFG)
    times=[];tracks=[];e=sim._engine
    while True:
        info=e.info();times.append(info['time']);tracks.append(e.predators())
        if not e.agents() or info['time']>=3000-1e-6:break
        sim.run_policy(3000.,len(times)*1.)
    n=max(map(len,tracks));arr=np.full((len(times),n,5),np.nan,dtype=np.float32)
    for j,row in enumerate(tracks):
        if row:arr[j,:len(row)]=row
    ts=np.array(times);np.savez_compressed(OUT/f'seed-{seed}.npz',times=ts,predators=arr)
    cases=[]
    for pred in range(n):
        valid=np.isfinite(arr[:,pred,0]);tt=ts[valid];xy=arr[valid,pred,:2]
        for l,r in episodes(tt,xy):
            segment=xy[l:r+1]
            cases.append(dict(predator_index=pred,start=float(tt[l]),end=float(tt[r]),duration=float(tt[r]-tt[l]),bbox=[*segment.min(axis=0).astype(float),*segment.max(axis=0).astype(float)],rest_fraction=float(np.mean(arr[valid,pred,4][l:r+1]>0))))
    return dict(seed=seed,score=info['score'],survival=info['time'],cases=cases,case_count=len(cases),trapped_predator_seconds=sum(c['duration'] for c in cases),wall_seconds=time.monotonic()-start)

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args();OUT=a.out;OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Already started')
    (OUT/'manifest.json').write_text(json.dumps(dict(seeds=list(range(19001,20001)),config=CFG,policy_seed=0,horizon=3000,sample_seconds=1,bbox_width=40,min_seconds_exclusive=60,predator_identity='append-only engine vector index',build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text())),indent=2))
    start=time.monotonic()
    with Pool(a.workers) as pool,(OUT/'games.jsonl').open('w',buffering=1) as f:
        for count,row in enumerate(pool.imap_unordered(one,range(19001,20001)),1):
            f.write(json.dumps(row)+'\n')
            if count%10==0:os.fsync(f.fileno());print(json.dumps(dict(games=count,total=1000,elapsed=time.monotonic()-start)),flush=True)
    (OUT/'complete.json').write_text(json.dumps(dict(games=1000,elapsed=time.monotonic()-start)))
if __name__=='__main__':main()

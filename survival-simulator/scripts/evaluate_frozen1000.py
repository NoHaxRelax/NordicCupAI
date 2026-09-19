"""Frozen 12-policy comparison on 1000 fresh paired maps, with tick profiling."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json,pathlib,sys,time,random
from multiprocessing import Pool
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastsim.fastpolicy import PolicySimulationCore

def configs():
    result={p.parent.name:json.loads(p.read_text())['config'] for p in sorted((ROOT/'docs/families20').glob('*/winner.json'))}
    assert len(result)==10
    result['previous_winner']=json.loads((ROOT/'docs/families20/expanded_population/baseline-config.json').read_text())
    result['original_baseline']=json.loads((ROOT/'docs/families20/population_harvest/original_baseline-config.json').read_text())
    return result

def one(job):
    name,seed,cfg=job
    sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,cfg)
    wall=time.perf_counter_ns();cpu=time.process_time_ns()
    steps,peak,interface,policy,engine=sim._engine.run_policy(3000.,3000.,True)
    cpu=time.process_time_ns()-cpu;wall=time.perf_counter_ns()-wall
    return dict(model=name,seed=seed,score=sim.env.score,survival=sim.env.time,steps=steps,peak=peak,
                ns_interface=interface,ns_policy=policy,ns_engine=engine,ns_loop_wall=wall,ns_loop_cpu=cpu)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=32);a=ap.parse_args();assert 0<=a.shard<10
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Refusing to overwrite existing run')
    cfg=configs();seeds=list(range(10001+a.shard*100,10101+a.shard*100))
    files=list((ROOT/'fastsim').glob('*.cpp'))+list((ROOT/'fastsim').glob('*.hpp'))+[pathlib.Path(__file__)]
    manifest=dict(shard=a.shard,seeds=seeds,configs=cfg,policy_seed=0,horizon=3000.,predators=True,workers=a.workers,
      python=sys.version,numpy=np.__version__,profile=True,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},
      build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    jobs=[(name,s,c)for s in seeds for name,c in cfg.items()];random.Random(1000+a.shard).shuffle(jobs)
    start=time.monotonic();count=0
    with Pool(a.workers)as pool,open(out/'games.jsonl','w',buffering=1)as f:
        it=pool.imap_unordered(one,jobs)
        for _ in jobs:
            remaining=3600-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('One-hour shard fail-safe')
            row=it.next(timeout=remaining);f.write(json.dumps(row)+'\n');count+=1
            if count%100==0:print(json.dumps(dict(shard=a.shard,games=count,total=len(jobs),elapsed=time.monotonic()-start)),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(games=count,elapsed_seconds=time.monotonic()-start))+'\n')
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

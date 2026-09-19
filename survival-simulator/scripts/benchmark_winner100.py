"""Frozen 100-map native score benchmark; no competition submission."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,pathlib,platform,sys,time
from multiprocessing import Pool
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from fastsim.fastpolicy import PolicySimulationCore

def one(job):
    seed,cfg=job;t=time.perf_counter();sim=PolicySimulationCore(seed=seed,predators=True)
    sim.policy_init(0,cfg);steps,peak,*_=sim.run_policy(3000.)
    return dict(seed=seed,policy_seed=0,score=sim.env.score,survival=sim.env.time,alive=len(sim.env.agents),predators=len(sim.env.predators),peak=peak,steps=steps,wall_seconds=time.perf_counter()-t,metrics=sim.policy_metrics())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--out',default='docs/nikolaj100');a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'games.jsonl').exists():raise RuntimeError('Refuse to overwrite a previous benchmark')
    full=json.loads((ROOT/'models/best_policies/rank1_pod-03.json').read_text())
    cfg={k:float('inf') if v is None else v for k,v in full['orchard'].items()};cfg.update(full['evasion'])
    seeds=list(range(5001,5101));sources=['fastsim/_engine.cpp','fastsim/_orchard.hpp','fastsim/_evasion.hpp','fastsim/_orchard_policy.cpp','fastsim/_policy.cpp','fastsim/policy_abi.hpp','fastsim/policy_iface.hpp','fastsim/_shared_pysem.inc','fastsim/fastpolicy.py','fastsim/__init__.py','scripts/benchmark_winner100.py','models/best_policies/rank1_pod-03.json']
    manifest=dict(upstream_commit='6e49081d845dae6f650b4e3d365f383ea4f31ee8',config=full,world_seeds=seeds,policy_seed=0,horizon=3000.,predators=True,workers=a.workers,python=sys.version,numpy=np.__version__,platform=platform.platform(),source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in sources},native_build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');t=time.perf_counter();rows=[]
    with Pool(a.workers) as pool,open(out/'games.jsonl','w',buffering=1) as f:
        for r in pool.imap_unordered(one,[(s,cfg)for s in seeds]):
            rows.append(r);f.write(json.dumps(r)+'\n')
            if len(rows)%10==0:print(f'{len(rows)}/100 complete in {time.perf_counter()-t:.1f}s',flush=True)
    scores=np.array([r['score']for r in sorted(rows,key=lambda r:r['seed'])]);rng=np.random.default_rng(64012);boot=scores[rng.integers(0,100,size=(20000,100))].mean(axis=1)
    summary=dict(n=len(rows),mean_score=float(scores.mean()),median_score=float(np.median(scores)),score_std=float(scores.std(ddof=1)),mean_score_95_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),min_score=float(scores.min()),max_score=float(scores.max()),mean_survival=float(np.mean([r['survival']for r in rows])),reached_horizon=sum(r['survival']>=3000 for r in rows),elapsed_seconds=time.perf_counter()-t,mean_game_seconds=float(np.mean([r['wall_seconds']for r in rows])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

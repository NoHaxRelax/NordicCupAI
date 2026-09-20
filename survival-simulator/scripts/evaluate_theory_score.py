"""Paired full-game evaluation of theoretically motivated policy changes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json,pathlib,random,sys,time
from multiprocessing import Pool
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastsim.fastpolicy import PolicySimulationCore

EXPANDED=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
OSCAR={
 "breed_reserve":232.7158,"cap_hard_min":1,"cap_max":33,"cap_min":2,"cap_mult":.6,"cap_tree_slack":-1,
 "cluster_radius":42.0299,"dist_pen":.3,"dump_after_t":1459.3742,"dump_mult":2.9442,"explore_energy":252.9455,
 "extra_old":False,"feed_mode":"hungry","fit_energy":.5329,"fit_hear":.571,"fit_speed":.4437,"fit_vision":1.2076,
 "fruit_min_wait":.3845,"fruit_reach":267.142,"heir_age":51.7223,"heir_at_food":False,"heir_needs_site":False,
 "heir_reserve":342.6638,"heir_select":False,"hungry_margin":1.3027,"low_pop_reserve":212.1813,
 "old_eat_last":False,"old_reach":29.1274,"ripen_wait":18.5006,"rot_margin":43.8547,"select_min_young":1,
 "spread_weight":1.0464,"sweep_rate":.0255,"tree_reach":366.1661,"tree_slots":1,"watch_patience":25.2658,
 "watch_reach":464.1112,"pred_mode":1,"pred_dodge_r":86.1536,"pred_dodge_ang":1.3247,"pred_r":70,
 "pred_face_r":64.8902,"pred_sprint_r":31.1366}

def with_(base,**updates):
    result=dict(base);result.update(updates);return result

PILOT={
 'expanded':EXPANDED,
 'expanded_bio1':with_(EXPANDED,bio_w=1),
 'expanded_bio2':with_(EXPANDED,bio_w=2),
 'expanded_bio4':with_(EXPANDED,bio_w=4),
 'expanded_bio8':with_(EXPANDED,bio_w=8),
 'expanded_bio16':with_(EXPANDED,bio_w=16),
 'expanded_bio32':with_(EXPANDED,bio_w=32),
 'expanded_bio_hard':with_(EXPANDED,bio_w=1000000),
 'expanded_bio8_late300':with_(EXPANDED,bio_w=8,bio_t=300),
 'expanded_bio8_late600':with_(EXPANDED,bio_w=8,bio_t=600),
 'oscar_ref':OSCAR,
 'oscar_bio8':with_(OSCAR,bio_w=8),
}

def one(job):
    name,seed,cfg=job;start=time.monotonic();cpu=time.process_time_ns()
    sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,cfg)
    steps,peak,interface,policy,engine=sim._engine.run_policy(3000.,3000.,True)
    events=sim.pop_events()
    return dict(model=name,seed=seed,score=sim.env.score,survival=sim.env.time,steps=steps,peak=peak,
      predation_deaths=sum(e[0]=='predator' for e in events),energy_deaths=sum(e[0]=='starvation' for e in events),
      ns_interface=interface,ns_policy=policy,ns_engine=engine,ns_loop_cpu=time.process_time_ns()-cpu,
      game_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--count',type=int,required=True);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--configs',type=pathlib.Path);a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=PILOT if not a.configs else json.loads(a.configs.read_text())
    files=[ROOT/'fastsim/_orchard.hpp',ROOT/'fastsim/_orchard_policy.cpp',pathlib.Path(__file__)]
    manifest=dict(seeds=[a.start,a.start+a.count],configs=cfg,policy_seed=0,horizon=3000,predators=True,workers=a.workers,
      sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
      build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text()))
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    done={}
    gp=out/'games.jsonl'
    if gp.exists():
        for line in gp.read_text().splitlines():
            row=json.loads(line);done[(row['model'],row['seed'])]=row
    jobs=[(name,seed,c) for seed in range(a.start,a.start+a.count) for name,c in cfg.items() if (name,seed) not in done]
    random.Random(a.start).shuffle(jobs);start=time.monotonic()
    with Pool(a.workers) as pool,gp.open('a',buffering=1) as f:
        for i,row in enumerate(pool.imap_unordered(one,jobs),1):
            f.write(json.dumps(row)+'\n')
            if i%50==0:print(json.dumps(dict(games=len(done)+i,total=len(done)+len(jobs),elapsed=time.monotonic()-start)),flush=True)
    rows=[json.loads(x) for x in gp.read_text().splitlines()]
    summary={name:{'n':len(rs),'mean_score':float(np.mean([x['score'] for x in rs])),
                   'mean_survival':float(np.mean([x['survival'] for x in rs])),
                   'mean_seconds':float(np.mean([x['game_seconds'] for x in rs])),
                   'mean_predation_deaths':float(np.mean([x['predation_deaths'] for x in rs]))}
             for name in cfg for rs in [[x for x in rows if x['model']==name]]}
    base={x['seed']:x['score'] for x in rows if x['model']=='expanded'}
    for name,s in summary.items():
        if name!='expanded':s['paired_gain']=float(np.mean([x['score']-base[x['seed']] for x in rows if x['model']==name]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

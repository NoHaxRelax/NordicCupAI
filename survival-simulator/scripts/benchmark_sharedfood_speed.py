"""Same-host full-game timing and exact per-tick action digest for speed changes."""
import argparse,pathlib,sys,json,time,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--model',required=True);ap.add_argument('--seed',type=int,default=19001);ap.add_argument('--horizon',type=float,default=3000);ap.add_argument('--trace',action='store_true');a=ap.parse_args()
r=pathlib.Path(a.root);sys.path.insert(0,str(r));from fastsim.fastpolicy import PolicySimulationCore
cfg=json.loads((r/'docs/sharedfood20/final/shard-0/manifest.json').read_text())['configs'][a.model]
s=PolicySimulationCore(seed=a.seed,predators=True);s.policy_init(0,cfg);s._engine.policy_profile(True)
t=time.monotonic();cpu=time.process_time();digest=hashlib.sha256();ticks=0
if a.trace:
 while s.env.time<a.horizon-1e-6:
  acts=s.policy_act()
  if not acts:break
  digest.update(repr(acts).encode());s.step([(aid,dict(move_distance=d,move_direction=h,turn_angle=t,spawn_agent=b))for aid,d,h,t,b in acts]);ticks+=1
else:ticks,*_=s._engine.run_policy(a.horizon,a.horizon,True)
print(json.dumps(dict(model=a.model,seed=a.seed,horizon=a.horizon,trace=a.trace,score=s.env.score,time=s.env.time,ticks=ticks,wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,action_digest=digest.hexdigest()if a.trace else None,phases=s._engine.policy_phases())),flush=True)

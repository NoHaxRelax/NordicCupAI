"""One frozen full game, optionally hash every action and observation tick."""
import argparse,sys,pathlib,json,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
sys.path.insert(0,a.root)
from fastsim.fastpolicy import PolicySimulationCore
s=PolicySimulationCore(seed=a.seed,predators=True);s.policy_init(0,json.loads(pathlib.Path(a.config).read_text()));h=hashlib.sha256();t=time.monotonic();cpu=time.process_time();n=0
if a.trace:
 while s.env.time<3000-1e-6:
  acts=s.policy_act()
  if not acts:break
  h.update(repr(acts).encode());s.step([(aid,dict(move_distance=d,move_direction=v,turn_angle=z,spawn_agent=b))for aid,d,v,z,b in acts]);n+=1
else:n,*_=s._engine.run_policy(3000.,3000.,True)
print(json.dumps(dict(seed=a.seed,score=s.env.score,survival=s.env.time,steps=n,action_digest=h.hexdigest()if a.trace else None,seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu)))

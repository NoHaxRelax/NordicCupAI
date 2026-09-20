"""Replay identical actions without modifying Python identity hashes or RNG."""
import os
os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,pathlib,json,random,time
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.core import SimulationCore as PythonCore
from src.utils.DTOs import ActionRequest
from fastsim import SimulationCore as NativeCore

def world(s):
 e=s.env
 return dict(agents=[(a.agent_id,a.x,a.y,a.direction,a.age,a.energy,a.max_energy,a.speed,a.sprint_speed)for a in e.agents],predators=[(p.x,p.y,p.direction,p.energy,p.resting)for p in e.predators],fruits=[(f.fruit_id,f.x,f.y,f.energy,f.age,f.radius)for f in e.fruits],trees=[(t.x,t.y,t.radius,t.age)for t in e.trees],score=e.score,time=e.time)
def rng(s):return s.rng.getstate()if hasattr(s,'rng')else s._engine.rng_state()
def check(seed,kind):
 A=PythonCore if kind=='python-python'else NativeCore
 B=PythonCore if kind=='python-python'else NativeCore
 a=A(seed=seed,starting_predators=30);b=B(seed=seed,starting_predators=30)
 if kind=='python-native':a=PythonCore(seed=seed,starting_predators=30)
 first_obs=None;start=time.monotonic();rr=random.Random(987)
 for step in range(1501):
  wa,wb=world(a),world(b)
  if wa!=wb or rng(a)!=rng(b):return dict(seed=seed,kind=kind,exact=False,first_world_difference=step,fields=[k for k in wa if wa[k]!=wb[k]],rng_equal=rng(a)==rng(b),first_observation_difference=first_obs,seconds=time.monotonic()-start)
  if not a.env.agents:break
  actions=[(x.agent_id,ActionRequest(agent_id=x.agent_id,move_distance=rr.choice([0.,7.,20.]),move_direction=rr.uniform(-3.14,3.14),turn_angle=rr.uniform(-.4,.4),spawn_agent=x.energy>180 and step%10==0))for x in a.env.agents]
  sa=a.step(actions);sb=b.step(actions)
  if sa!=sb and first_obs is None:first_obs=step+1
 return dict(seed=seed,kind=kind,exact=True,ticks=step,first_observation_difference=first_obs,seconds=time.monotonic()-start)
if __name__=='__main__':
 out=R/'docs/seed-shadow/replay-check.jsonl'
 with out.open('w',buffering=1)as f:
  for kind in ['python-python','python-native','native-native']:
   for seed in [1,2,3]:
    r=check(seed,kind);f.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)

"""Same action stream in unmodified Python and fastest native engine."""
import os
os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import sys,pathlib,json,time,math
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from check_seed_replay import world,rng,PythonCore,ActionRequest
from fastsim.fastpolicy import PolicySimulationCore
seed=int(sys.argv[1]);a=PythonCore(seed=seed);b=PolicySimulationCore(seed=seed);cfg=json.loads((R/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config'];b.policy_init(0,cfg);start=time.monotonic();obs_first=None
kind=sys.argv[2] if len(sys.argv)>2 else 'python-native'
c=PythonCore(seed=seed) if kind=='python-python' else b
def close(x,y):
 if isinstance(x,dict):return x.keys()==y.keys() and all(close(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(close(a,b) for a,b in zip(x,y))
 if isinstance(x,float):return math.isclose(x,y,rel_tol=0,abs_tol=1e-8)
 return x==y
first_exact=None
for step in range(30001):
 wa,wb=world(a),world(c)
 if wa!=wb and first_exact is None:
  first_exact=step
  print(json.dumps(dict(first_exact_difference=step,time=a.env.time)),flush=True)
 if not close(wa,wb) or rng(a)!=rng(c):
  print(json.dumps(dict(seed=seed,kind=kind,differences={k:[{'index':i,'python':x,'other':y} for i,(x,y) in enumerate(zip(wa[k],wb[k])) if x!=y][:5] if isinstance(wa[k],list) else [wa[k],wb[k]] for k in wa if wa[k]!=wb[k]},exact=False,first_world_difference=step,time=a.env.time,fields=[k for k in wa if wa[k]!=wb[k]],rng_equal=rng(a)==rng(c),first_observation_difference=obs_first,seconds=time.monotonic()-start)),flush=True);break
 if not a.env.agents or a.env.time>=3000:
  print(json.dumps(dict(seed=seed,within_tolerance=True,first_exact_difference=first_exact,ticks=step,time=a.env.time,score=a.env.score,first_observation_difference=obs_first,seconds=time.monotonic()-start)),flush=True);break
 acts=b.policy_act();actions=[(aid,ActionRequest(agent_id=aid,move_distance=d,move_direction=v,turn_angle=t,spawn_agent=z))for aid,d,v,t,z in acts]
 sa=a.step(actions);sb=b.step(actions)
 if c is not b:sb=c.step(actions)
 if sa!=sb and obs_first is None:obs_first=step+1
 if step%1000==0:print(json.dumps(dict(progress=step,sim_time=a.env.time)),flush=True)

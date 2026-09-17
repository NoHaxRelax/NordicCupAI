"""One legal birth to replace a stationary wall bait. No food or injected agents."""
import json,math
from pathlib import Path
from predator_control import controlled_env,add_agent,add_predator,wrap
from src.elements.obstacle import Obstacle
ROOT=Path(__file__).resolve().parents[1]

def run(seed,old_age,birth_at):
 env=controlled_env();env.rng.seed(seed)
 wall=Obstacle(784,560,32,80)
 env.obstacles=[wall,Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
 for w in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
 a=add_agent(env,823,600,150);a.max_age=old_age;env._next_agent_id=1
 p=add_predator(env,772,600,heading=0);env._update_spatial_grid()
 born=False;birth_position=None;deaths=[];held=[];attention=[];energy_spent=0
 for tick in range(1600):
  if not env.agents:break
  before=list(env.agents)
  for current in before:
   dx,dy=823-current.x,600-current.y
   move=min(current.speed,math.hypot(dx,dy))
   direction=wrap(math.atan2(dy,dx)-current.direction)
   spawn=bool(current is a and not born and birth_at is not None and env.time>=birth_at and current.energy>100)
   env.agent_step(current.agent_id,move,direction,0,spawn)
   if spawn:
    born=True;child=env.agents[-1];birth_position=[child.x,child.y];energy_spent+=100
  env.non_agent_step(.1)
  for old in before:
   if old.agent_id not in env.agents_dict:deaths.append(dict(id=old.agent_id,time=round(env.time,1),energy=round(old.energy,3),cause='depleted' if old.energy<=0 else 'capture'))
  held.append(bool(env.agents and p.x<784 and abs(p.y-600)<45 and 730<p.x))
  attention.append(bool(any(o['type']=='Agent' for o in p.observe(agents=list(env.agents),edges=list(env.edges)))))
 return dict(seed=seed,founder_old_age=old_age,birth_at=birth_at,born=born,birth_position=birth_position,birth_energy_paid=energy_spent,seconds=round(env.time,1),deaths=deaths,hold_fraction=sum(held)/len(held),attention_fraction=sum(attention)/len(attention),captures=sum(d['cause']=='capture' for d in deaths))

if __name__=='__main__':
 rows=[run(seed,age,birth) for seed in range(1,9) for age in [60,90,120] for birth in [None,40,45,48]]
 (ROOT/'results/wall-replacement.json').write_text(json.dumps(dict(scope=__doc__,runs=rows),indent=2,default=lambda v:v.item())+'\n')
 for age in [60,90,120]:
  for birth in [None,40,45,48]:
   group=[r for r in rows if r['founder_old_age']==age and r['birth_at']==birth]
   print(age,birth,'mean',round(sum(r['seconds'] for r in group)/len(group),1),'range',min(r['seconds'] for r in group),max(r['seconds'] for r in group),'captures',sum(r['captures'] for r in group),'held',round(sum(r['hold_fraction'] for r in group)/len(group),3),flush=True)

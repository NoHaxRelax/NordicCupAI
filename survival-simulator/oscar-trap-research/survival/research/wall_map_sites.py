"""Arranged wall holds on extracted generated maps, not online play.

Keeps each seed's terrain, obstacles, initial trees and fruit. Agents and predator
are deliberately placed on opposite wall faces; no new predators/trees spawn.
"""
import copy,json,math,sys,time
from pathlib import Path
from predator_control import controlled_env,add_agent,add_predator
from src.core import SimulationCore
ROOT=Path(__file__).resolve().parents[1]

def probe(source,wall,horizontal,side=1):
 env=controlled_env();env.biome_map=source.biome_map;env.obstacles=source.obstacles;env.edges=set(source.edges)
 env.trees=copy.deepcopy(source.trees);env.fruits=copy.deepcopy(source.fruits)
 env.fruits_dict={f.fruit_id:f for f in env.fruits};env._next_fruit_id=source._next_fruit_id
 env.rng.setstate(source.rng.getstate());env._update_spatial_grid()
 if not horizontal:
  ax=wall.x+wall.width+7 if side==1 else wall.x-7;ay=wall.y+wall.height/2
  px=wall.x-12 if side==1 else wall.x+wall.width+12;py=ay;heading=0 if side==1 else math.pi
 else:
  ax=wall.x+wall.width/2;ay=wall.y+wall.height+7 if side==1 else wall.y-7
  px=ax;py=wall.y-12 if side==1 else wall.y+wall.height+12;heading=math.pi/2 if side==1 else -math.pi/2
 row=dict(wall=[wall.x,wall.y,wall.width,wall.height],horizontal=horizontal,side=side,bait=[ax,ay],predator=[px,py])
 if not(35<ax<1565 and 35<ay<1165 and 40<px<1560 and 40<py<1160) or env._in_obstacle((ax,ay),5,env.obstacles) or env._in_obstacle((px,py),10,env.obstacles):return dict(**row,valid=False,reason='Adjacent wall or physical boundary blocks the arranged positions')
 a=add_agent(env,ax,ay,150);a.direction=heading+math.pi
 p=add_predator(env,px,py,heading=heading);env._update_spatial_grid()
 row.update(valid=True,bait_biome=env.biome_map[int(ax),int(ay)].type,predator_biome=env.biome_map[int(px),int(py)].type,trees_within150=sum(math.dist((ax,ay),(t.x,t.y))<=150 for t in env.trees),fruits_within100=sum(math.dist((ax,ay),(f.x,f.y))<=100 for f in env.fruits))
 held=[];att=[]
 for tick in range(600):
  if not env.agents:break
  env.agent_step(a.agent_id,0,0,0,False);env.non_agent_step(.1)
  across=((p.y-(wall.y+wall.height/2))*side<0) if horizontal else ((p.x-(wall.x+wall.width/2))*side<0)
  held.append(across and math.dist((a.x,a.y),(p.x,p.y))<65)
  att.append(any(o['type']=='Agent' for o in p.observe(agents=list(env.agents),edges=list(env.edges))))
 row.update(seconds=round(env.time,1),alive=bool(env.agents),energy=round(a.energy,2),tail_hold=sum(held[-200:])/len(held[-200:]),tail_attention=sum(att[-200:])/len(att[-200:]))
 row['success']=bool(row['alive'] and row['seconds']>=60 and row['tail_hold']>.9 and row['tail_attention']>.9)
 return row

if __name__=='__main__':
 output=[];start=time.perf_counter()
 for seed in [1,2,3,4,5,6,7,8,9,42]:
  sim=SimulationCore(seed=seed);source=sim.env
  walls=[(w,w.height<=35) for w in source.obstacles[4:] if (w.width<=35 and w.height>=70) or (w.height<=35 and w.width>=70)]
  probes=[probe(source,w,h,s) for w,h in walls for s in [-1,1]]
  row=dict(seed=seed,total_generated_walls=len(source.obstacles)-4,candidate_walls=len(walls),probes=probes)
  output.append(row)
  print(seed,'candidates',len(walls),'valid',sum(p['valid'] for p in probes),'held',sum(p.get('success',False) for p in probes),'runtime',round(time.perf_counter()-start,1),flush=True)
  (ROOT/'results/wall-map-sites.json').write_text(json.dumps(dict(scope=__doc__,seeds=output),indent=2,default=lambda v:v.item())+'\n')

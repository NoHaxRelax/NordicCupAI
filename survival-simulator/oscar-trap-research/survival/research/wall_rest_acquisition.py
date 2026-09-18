"""Safe wall acquisition while a predator is already resting beside a wall."""
import json,math
from wall_acquisition import path_to
from predator_control import controlled_env,add_agent,add_predator,wrap,ROOT
from src.elements.obstacle import Obstacle

def run(width=32,height=80,heading=0,energy=0,start_side='front',speed=10,record=None):
 env=controlled_env();wall=Obstacle(800-width/2,600-height/2,width,height)
 env.obstacles=[wall,Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
 for w in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
 ax,ay=(740,600) if start_side=='front' else (800,600-height/2-50)
 a=add_agent(env,ax,ay,150);p=add_predator(env,wall.x-12,600,heading=heading,energy=energy);p.resting=True
 env._update_spatial_grid();goal=(wall.x+wall.width+7,600);route=path_to((ax,ay),goal,wall)
 arrived=None;awake=None;held=[];attention=[]
 for tick in range(600):
  if not env.agents:break
  while route and math.dist((a.x,a.y),route[0])<1:route.pop(0)
  if route:
   tx,ty=route[0];move=min(speed,math.dist((a.x,a.y),(tx,ty)));direction=wrap(math.atan2(ty-a.y,tx-a.x)-a.direction)
  else:move=direction=0
  env.agent_step(a.agent_id,move,direction,0,False)
  if arrived is None and math.dist((a.x,a.y),goal)<1:arrived=env.time
  env.non_agent_step(.1)
  if awake is None and not p.resting:awake=env.time
  held.append(bool(env.agents and p.x<800 and math.dist((a.x,a.y),(p.x,p.y))<65))
  attention.append(any(o['type']=='Agent' for o in p.observe(agents=list(env.agents),edges=list(env.edges))))
 tail=held[-200:];att=attention[-200:]
 return dict(width=width,height=height,heading=heading,rest_start_energy=energy,start_side=start_side,speed=speed,seconds=round(env.time,1),arrived_at=None if arrived is None else round(arrived,1),woke_at=None if awake is None else round(awake,1),alive=bool(env.agents),energy_left=round(a.energy,2),success=bool(env.agents and env.time>=59.99 and sum(tail)/len(tail)>.9 and sum(att)/len(att)>.9),tail_hold=sum(tail)/len(tail),tail_attention=sum(att)/len(att))

if __name__=='__main__':
 rows=[run(heading=h,energy=e,start_side=s,speed=speed) for speed in [10,20] for h in [-math.pi,-math.pi/2,-.6,-.3,0,.3,.6,math.pi/2] for e in [0,45,75] for s in ['front','above']]
 validation=[run(width=w,height=l,heading=h,energy=0,start_side='front') for w,l in [(30,70),(35,100),(38,70)] for h in [-.45,-.15,.15,.45]]
 (ROOT/'results/wall-rest-acquisition.json').write_text(json.dumps(dict(scope=__doc__,runs=rows,validation=validation),indent=2,default=lambda v:v.item())+'\n')
 for speed in [10,20]:
  for e in [0,45,75]:
   rs=[r for r in rows if r['rest_start_energy']==e and r['speed']==speed];print('speed',speed,'initial resting energy',e,'held',sum(r['success'] for r in rs),'/',len(rs),'alive',sum(r['alive'] for r in rs),flush=True)
 print('heldout',sum(r['success'] for r in validation),'/',len(validation))

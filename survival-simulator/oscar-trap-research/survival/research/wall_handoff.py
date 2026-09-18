"""Privileged-coordinate two-agent wall acquisition. Exact local engine physics."""
import json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from predator_control import controlled_env,add_agent,add_predator,wrap
from src.elements.obstacle import Obstacle
ROOT=Path(__file__).resolve().parents[1]

def run(width=32,height=80,energy=150,gap=90,peel=40,offset=0,heading=0,side=1,track=False,seconds=60,record=None):
 env=controlled_env();wall=Obstacle(800-width/2,600-height/2,width,height)
 env.obstacles=[wall,Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
 for w in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
 bait=add_agent(env,800+width/2+7,600,energy=150)
 guide=add_agent(env,730,600,energy=energy)
 p=add_predator(env,730-gap,600+offset,heading=heading)
 guide.direction=math.atan2(p.y-guide.y,p.x-guide.x);bait.direction=math.pi
 env._update_spatial_grid();phase='lead';phase_at=None;held=[];attention=[];trace=[];death={};recorder=None
 if record:
  sys.path.insert(0,str(ROOT/'debugger'))
  from recorder import ReplayRecorder
  recorder=ReplayRecorder(env,native_render=True,title='Wall bait acquisition with guide',policy='geometry-aware handoff',seed=1729,every=2,scenario='controlled',notes='Two arranged150-energy agents, exact positions and wall geometry supplied to controller. Physical boundaries; original engine physics; no fruit or new predators.')
  recorder.capture()
 from src.utils.DTOs import ActionRequest
 for tick in range(round(seconds*10)):
  actions=[];action_t=env.time
  if guide.agent_id in env.agents_dict:
   px,py=p.x,p.y;distance=math.dist((guide.x,guide.y),(px,py));gx=wall.x-7
   if phase=='lead' and guide.x>=gx-1:phase='wait'
   if phase=='wait' and distance<peel:phase='peel';phase_at=env.time
   if phase in ('lead','wait'):target=(gx,600)
   elif phase=='released':target=(guide.x,guide.y)
   elif abs(guide.y-600)<height/2+75:target=(guide.x,600+side*(height/2+80))
   elif distance>180:phase='released';target=(guide.x,guide.y)
   else:target=(max(40,guide.x-10),guide.y)
   length=math.dist((guide.x,guide.y),target)
   request=min(length,20 if distance<100 else 10)
   relative=wrap(math.atan2(target[1]-guide.y,target[0]-guide.x)-guide.direction)
   turn=wrap(math.atan2(py-guide.y,px-guide.x)-guide.direction) if phase!='released' else 0
   action=ActionRequest(agent_id=guide.agent_id,move_distance=request,move_direction=relative,turn_angle=turn,spawn_agent=False)
   actions.append((guide.agent_id,action));env.agent_step(guide.agent_id,request,relative,turn,False)
  if bait.agent_id in env.agents_dict:
   target_y=max(600-height/2+15,min(600+height/2-15,p.y)) if track else 600
   distance=min(10,abs(target_y-bait.y));direction=wrap((math.pi/2 if target_y>bait.y else -math.pi/2)-bait.direction)
   action=ActionRequest(agent_id=bait.agent_id,move_distance=distance,move_direction=direction,turn_angle=0,spawn_agent=False)
   actions.append((bait.agent_id,action));env.agent_step(bait.agent_id,distance,direction,0,False)
  before=list(env.agents);env.non_agent_step(.1)
  for a in before:
   if a.agent_id not in env.agents_dict:death[a.agent_id]=round(env.time,1)
  alive=bait.agent_id in env.agents_dict
  held.append(bool(alive and wall.x-40<p.x<800 and abs(p.y-600)<height/2+5))
  seen=[o for o in p.observe(agents=list(env.agents),edges=list(env.edges)) if o['type']=='Agent']
  target=min(seen,key=lambda o:o['distance'])['id'] if seen else None
  attention.append(target==bait.agent_id)
  if tick%10==0:trace.append([round(env.time,1),round(p.x,1),round(p.y,1),round(guide.x,1),round(guide.y,1),round(guide.energy,1),phase,held[-1],target])
  if recorder:recorder.capture(actions,{bait.agent_id:{'rule':'Hold wall bait'},guide.agent_id:{'rule':'Lead predator' if phase=='lead' else 'Peel away'}},action_t=action_t)
  if not alive:break
 tail=held[-200:];att=attention[-200:]
 result=dict(width=width,height=height,energy=energy,gap=gap,peel=peel,offset=offset,heading=heading,side=side,track=track,
 seconds=round(env.time,1),success=bool(alive and env.time>=seconds-.01 and sum(tail)/len(tail)>.9 and sum(att)/len(att)>.9),bait_alive=alive,guide_alive=guide.agent_id in env.agents_dict,
 guide_energy=round(guide.energy,2),bait_energy=round(bait.energy,2),deaths=death,tail_hold=sum(tail)/len(tail),tail_attention=sum(att)/len(att),peel_at=phase_at,trace=trace)
 if recorder:recorder.save(record,reason='duration' if alive else 'bait died',overwrite=True)
 return result

if __name__=='__main__':
 results=[];start=time.perf_counter()
 for energy in [150,500]:
  for gap in [60,90,140]:
   for peel in [20,25,30,35,40,50]:
    for side in [-1,1]:
     for track in [False,True]:results.append(run(energy=energy,gap=gap,peel=peel,side=side,track=track))
 print('train',sum(r['success'] for r in results),'/',len(results),flush=True)
 promising=[next(r for r in results if r['energy']==150 and r['gap']==gap and r['peel']==peel and r['side']==1 and not r['track']) for gap,peel in [(60,30),(90,30),(140,20)]]
 validation=[]
 for r in promising:
  for width,height in [(30,70),(35,100),(38,70)]:
   for heading in [-.3,.3]:
    for offset in [-15,15]:
     validation.append(run(**{k:r[k] for k in ['energy','gap','peel','side','track']},width=width,height=height,heading=heading,offset=offset))
 (ROOT/'results/wall-handoff.json').write_text(json.dumps(dict(scope='Two-agent controlled acquisition; known exact coordinates, no food or new predators; physical boundaries.',train=results,validation=validation),indent=2,default=lambda v:v.item())+'\n')
 print('validation',sum(r['success'] for r in validation),'/',len(validation),'runtime',time.perf_counter()-start)
 for r in promising:print({k:v for k,v in r.items() if k!='trace'})

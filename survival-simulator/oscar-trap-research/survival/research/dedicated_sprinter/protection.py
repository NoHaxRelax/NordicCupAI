"""Fixed-horizon paired worker protection; no early exit on bait death.
Two stationary 150-energy workers are a diagnostic protection fixture, not an
optimised colony. Capture and aging/starvation deaths are separated explicitly.
"""
from experiment import build,observe,save,BaitController,DecoyController,OUT,ReplayRecorder,hashes,RunRecording,CONTROLLER_MODULE,import_module
from src.utils.DTOs import ActionRequest
import math,json,argparse

def protect(c,mode,horizon=90,record=None,*,controller_module=None,reproduction=None,native=None):
    e,a,p,workers=build({**c,'workers':2});aid=a.agent_id
    if mode=='none':e.kill_agent(a)
    states=observe(e)
    ctrl=import_module(controller_module or CONTROLLER_MODULE).BaitController(**c.get('controller',{}))
    old=DecoyController(mode='adaptive',request=20,trigger=60,initial={aid:(0,0,0)})
    deaths={};captures=0;starvations=0;workertime=0.;target_ticks=0;active_ticks=0;first_worker_target=None;firstcapture=None;baitdeath=None
    rec=RunRecording(e,policy=mode,module=controller_module or CONTROLLER_MODULE,
        seed=c.get('seed',0),scenario='arranged-worker-protection',parameters={**c,'workers':2},
        horizon=horizon,label=record,reproduction=reproduction,native=native)
    for tick in range(round(horizon*10)):
        inputs=states;action_t=e.time
        actions=[];why={}
        if a in e.agents:
            if mode=='adaptive':actions,why=old(states,e.time)
            else:
                action,label=ctrl.step(next(s for s in states if s['agent_id']==aid),e.time)
                actions=[(aid,action)];why={aid:label}
        for w in workers:
            if w in e.agents and w.agent_id not in dict(actions):actions.append((w.agent_id,ActionRequest(agent_id=w.agent_id,move_distance=0,move_direction=0,turn_angle=0,spawn_agent=False)))
        for ident,action in actions:e.agent_step(ident,action.move_distance,action.move_direction,action.turn_angle,False)
        active=not p.resting or p.energy>100
        obs=[o for o in p.observe(agents=e._get_local_agents(p),edges=e._get_local_edges(p)) if o['type']=='Agent']
        target=min(obs,key=lambda o:o['distance'])['id'] if obs else None
        if active:
            active_ticks+=1;target_ticks+=target==aid and a in e.agents
            if target in [w.agent_id for w in workers] and first_worker_target is None:first_worker_target=e.time
        before=list(e.agents);workertime+=.1*sum(w in e.agents for w in workers)
        e.non_agent_step(.1)
        for obj in before:
            if obj not in e.agents:
                cause='capture' if obj.energy>0 else 'starvation'
                deaths[obj.agent_id]=dict(t=round(e.time,2),cause=cause)
                if obj is a:baitdeath=e.time
                else:
                    captures+=cause=='capture';starvations+=cause=='starvation'
                    if cause=='capture' and firstcapture is None:firstcapture=e.time
        states=[e.get_agent_state(o.agent_id) for o in e.agents]
        rec.capture(actions,why,inputs,action_t)
    r=dict(scenario=c,mode=mode,horizon=horizon,workers_alive=sum(w in e.agents for w in workers),worker_captures=captures,worker_starvations=starvations,worker_seconds=round(workertime,2),first_worker_capture=firstcapture,first_worker_target=first_worker_target,bait_death=baitdeath,deaths=deaths,bait_target_fraction_over_whole_horizon=target_ticks/max(1,active_ticks),score=e.score)
    rec.finish(r,'fixed protection horizon reached')
    return r
if __name__=='__main__':
    rows=[]
    for seed,heading,position,rotation in [(301,0,[800,600],0),(302,.8,[600,400],.8),(303,-.8,[1000,700],-.4),(304,math.pi,[850,500],1.4),(305,0,[500,600],0),(306,-.3,[1100,700],2.3)]:
        for mode in ('none','adaptive','predictive'):
            c=dict(seed=seed,heading=heading,position=position,rotation=rotation,energy=150,food='orchard',controller=dict(center='auto'))
            r=protect(c,mode);rows.append(r);print(json.dumps(r),flush=True)
    save('protection',rows)

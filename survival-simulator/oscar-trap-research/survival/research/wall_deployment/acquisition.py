"""Observation-only acquisition from an arranged corner/resting opportunity.

No wall map, rest flag, predator energy or coordinates are given to the policy.
The observer's initial vantage/heading and the predator's rest timing are setup
privileges. Native physical walls, energy, one legal action and DTO sensing.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predator_control import controlled_env, add_predator
from src.elements.obstacle import Obstacle
from src.utils.DTOs import ObservationResponse, ActionRequest
from controller import WallPolicy

ROOT=Path(__file__).resolve().parents[2]


def run(width=30,height=80,heading=0.,energy=0.,record=None,native_render=False,reproduction=False):
    env=controlled_env()
    wall=Obstacle(784,560,width,height)
    env.obstacles=[wall,Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),
                   Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
    for w in env.obstacles:
        env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
    a=env.spawn_agent(x=755,y=545)
    a.direction=.9
    p=add_predator(env,wall.x-12,wall.y+height/2,heading=heading,energy=energy)
    p.resting=True
    env._update_spatial_grid()
    policy=WallPolicy()
    states=[ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump()]
    held=[];arrived=None;woke=None;trace=[]
    recorder=None
    if record is None:
        record=ROOT/'results/wall_deployment/replays'/f'rest-v7-w{width}-h{height}-e{energy}-heading{heading}-{uuid4().hex[:10]}.json.gz'
    if record:
        sys.path.insert(0,str(ROOT/'debugger'))
        from recorder import ReplayRecorder
        recorder=ReplayRecorder(env,native_render=native_render,every=10,scenario='arranged corner/rest window',
            title='Wall acquisition from ordinary observations',policy='WallPolicy',seed=1729,
            notes='Arranged corner vantage and resting predator. Controller only receives native DTOs; no wall map or rest flag. No food or additional spawns. '+
                  ('New reproduction of an unrecorded configuration, not recovered original footage.' if reproduction else ''))
        recorder.capture()
    for tick in range(600):
        if not env.agents: break
        action_t=env.time
        actions=policy.act(states,action_t)
        assert len(actions)==len(env.agents)==len({x['agent_id'] for x in actions})
        pairs=[]
        for action in actions:
            pairs.append((action['agent_id'],ActionRequest(**action)))
            env.agent_step(**action)
        env.non_agent_step(.1)
        if woke is None and not p.resting: woke=round(env.time,1)
        h=p.x<wall.x and abs(p.y-(wall.y+height/2))<height/2+10 and math.dist((p.x,p.y),(a.x,a.y))<65
        held.append(h)
        if arrived is None and a.x>wall.x+width and abs(a.y-(wall.y+height/2))<10: arrived=round(env.time,1)
        if tick%10==0: trace.append(dict(time=round(env.time,1),agent=[a.x,a.y],predator=[p.x,p.y],
            energy=a.energy,rule=policy.decisions.get(a.agent_id),held=h))
        if recorder: recorder.capture(pairs,policy.decisions,inputs=states,action_t=action_t)
        states=[ObservationResponse(**env.get_agent_state(c.agent_id)).model_dump() for c in env.agents]
    row=dict(width=width,height=height,heading=heading,initial_predator_energy=energy,
        seconds=round(env.time,1),alive=bool(env.agents),arrived=arrived,woke=woke,
        success=bool(env.agents and env.time>=59.99 and sum(held[-200:])/len(held[-200:])>.9),
        tail_hold=sum(held[-200:])/len(held[-200:]),energy=a.energy,policy_metrics=policy.metrics,
        policy_events=policy.events,trace=trace)
    if recorder:
        row['replay']=str(record)
        recorder.save(record)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',action='store_true')
    args=parser.parse_args()
    rows=[]
    for width,height in ((30,70),(35,100),(38,70)):
        for energy in (0,75):
            for heading in (-.3,.3):
                row=run(width,height,heading,energy,
                    native_render=args.record)
                rows.append(row)
                print({k:row[k] for k in ('width','height','heading','initial_predator_energy',
                       'seconds','success','arrived','policy_metrics')},flush=True)
    (ROOT/f'results/wall_deployment/rest-acquisition-{uuid4().hex[:10]}.json').write_text(json.dumps(dict(scope=__doc__,runs=rows),indent=2,default=lambda v:v.item())+'\n')

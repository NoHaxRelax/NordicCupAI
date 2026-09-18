"""Recorded geometry proof, observation/action contract and replay schema checks."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
from run import fixture, add_agent, add_pred, observe, RelayController, OUT
from replay_support import start_replay, finish_replay, save_batch

parser=argparse.ArgumentParser()
parser.add_argument('--backfill',action='store_true',help='Label these as new reproductions of the earlier unrecorded checks.')
args=parser.parse_args()
source=OUT/'verification.json'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def provenance(name):
    if not args.backfill:return None
    return dict(source_case='verification/'+name,source_file='verification.json',
        source_file_sha256=source_hash,limitations='New reproduction of the previously metrics-only verification fixture; original missing frames are not recoverable.')

# Privileged arranged geometry proof, not a deployable acquisition policy.
e=fixture(7)
p=add_pred(e,x=800,y=600,heading=0,energy=200)
radius=15/(2*math.sin(.15))
a=add_agent(e,800-radius*math.sin(.15),600+radius*math.cos(.15),150)
states=observe(e)
rec,path=start_replay(e,parameters=dict(policy='geometry-check',seed=7,energy=150,
    pred_energy=200,seconds=2.5,food='none'),policy_sha256=script_hash,every=1,native_render=True,
    reproduction=provenance('turning-circle'),assumptions='PRIVILEGED GEOMETRY FIXTURE: bait placed exactly at theoretical circle center; no acquisition policy. Shortened 2.5-second horizon, forest, physical boundaries, no food or new spawns. Native renderer and unmodified engine.')
gaps=[]
for _ in range(25):
    inputs=states;action_time=e.time
    action=dict(agent_id=a.agent_id,move_distance=0,move_direction=0,turn_angle=0,spawn_agent=False)
    e.agent_step(a.agent_id,0,0,0,False)
    e.non_agent_step(.1)
    gaps.append(math.hypot(a.x-p.x,a.y-p.y))
    rec.capture([(a.agent_id,action)],{a.agent_id:'Privileged stationary center fixture'},inputs,action_time)
    states=[e.get_agent_state(a.agent_id) for a in e.agents]
assert a.agent_id in e.agents_dict
assert max(abs(g-radius) for g in gaps)<1e-7
geometry=finish_replay(rec,path,dict(test='turning-circle',passed=True,ticks=25,
    radius=radius,gap_min=min(gaps),gap_max=max(gaps)))

# Controller receives copied DTO dictionaries and must not mutate them.
e=fixture(8)
add_pred(e,x=800,y=600,heading=0,energy=102)
a=add_agent(e,860,600,150)
b=add_agent(e,740,600,150)
b.direction=0
ctrl=RelayController([a.agent_id,b.agent_id],variant='orbit')
states=observe(e)
rec,path=start_replay(e,parameters=dict(policy='observation-contract-check',seed=8,energy=150,
    seconds=5,food='none'),policy_sha256=script_hash,every=1,
    reproduction=provenance('observation-contract'))
for _ in range(50):
    original=copy.deepcopy(states);action_time=e.time
    actions,why=ctrl(states,e.time)
    assert states==original
    assert len(actions)==len(states)==len({x['agent_id'] for x in actions})
    for x in actions:
        assert 0<=x['move_distance']<=e.agents_dict[x['agent_id']].sprint_speed+1e-4
        e.agent_step(x['agent_id'],x['move_distance'],x['move_direction'],x['turn_angle'],x['spawn_agent'])
    e.non_agent_step(.1)
    rec.capture([(x['agent_id'],x) for x in actions],why,original,action_time)
    states=[e.get_agent_state(a.agent_id) for a in e.agents]
assert len(e.agents)==2
contract=finish_replay(rec,path,dict(test='observation-contract',passed=True,ticks=50))

replays=[]
for path in sorted(OUT.rglob('*.json.gz')):
    with gzip.open(path,'rt') as f:data=json.load(f)
    assert data['format']=='survival-replay' and data['version']==1
    assert data['meta']['source_commit']=='acfc31a4003a5f91bf11032a02cd98c178ddbd7e'
    assert data['frames'][0]['t']==0
    assert all(a['t']<b['t'] for a,b in zip(data['frames'],data['frames'][1:]))
    assert all('action_observations' in a for frame in data['frames'][1:] for a in frame['agents'])
    replays.append(dict(file=str(path.relative_to(OUT)),frames=len(data['frames']),duration=data['summary']['duration']))
result=dict(turning_circle_radius=radius,geometry=geometry,observation_contract=contract,replays=replays)
print(save_batch('verification',result))
print(json.dumps(dict(geometry=geometry['replay'],contract=contract['replay'],replays_checked=len(replays)),indent=2))

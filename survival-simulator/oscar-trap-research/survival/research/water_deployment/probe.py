"""Public-own-biome transect certification after a candidate is nominated.

The nomination and initial approach pose are privileged. The scout learns water
bounds only from its current biome; river banks do not appear in observations.
This is a paid physical survey, not complete startup exploration/localization.
"""
from common import *
from run import initialize
from sites import world,local
from src.utils.DTOs import ActionRequest
from recording import start_recording, finish_recording
import numpy as np

def probe(seed,site,index,*,record=None,reproduction=None):
    init,skip=initialize(dict(seed=seed,site=site,native=True))
    if skip:return dict(seed=seed,site_index=index,skipped=skip)
    env,_,site=init
    env.predators=[];env.agents=env.agents[:1];a=env.agents[0]
    env.agents_dict={0:a};env.agent_observations={};env._update_spatial_grid()
    B=site['width']/2+45
    pos=np.array([B,-30.]);heading=0.
    a.x,a.y=world(site,pos);a.direction=site['angle']
    env._update_spatial_grid()
    if env._in_obstacle((a.x,a.y),5,env.obstacles):return dict(seed=seed,site_index=index,skipped='scout_start_blocked')
    goals=[[-B,-30],[-B,0],[B,0],[B,30],[-B,30]]
    transects={0:[],2:[],4:[]};leg=0;max_error=0;trace=[]
    recorder,replay_path=start_recording(env,dict(seed=seed,site=site,site_index=index,native=True),
        policy='Biome transect scout',demonstration=record,reproduction=reproduction,
        notes='Mapped candidate nomination and arranged initial scout pose. No predators. Public own biome informs paid physical crossings; true pose error is diagnostic only.')
    for tick in range(300):
        s=env.get_agent_state(0)
        if s is None:break
        if leg>=len(goals):break
        if leg in transects:transects[leg].append(dict(pos=pos.tolist(),biome=s['biome']))
        delta=np.array(goals[leg])-pos;dist=float(np.linalg.norm(delta))
        if dist<.01:leg+=1;continue
        direction=math.atan2(delta[1],delta[0]);penalty={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1)
        request=min(10,dist/penalty)
        action=ActionRequest(agent_id=0,move_distance=request,move_direction=direction-heading,turn_angle=0,spawn_agent=False)
        pos+=request*penalty*np.array([math.cos(direction),math.sin(direction)])
        t0=env.time
        env.agent_step(0,action.move_distance,action.move_direction,0,False);env.non_agent_step(.1)
        recorder.capture([(0,action)],{0:dict(rule='Cross nominated river to measure own biome',leg=leg)},[s],t0)
        true=local(site,[a.x,a.y]);max_error=max(max_error,float(np.linalg.norm(pos-true)))
        if tick%10==0:trace.append(dict(t=env.time,estimated_uv=pos.tolist(),true_uv=true.tolist(),energy=a.energy,biome=s['biome']))
    bounds=[]
    for leg,points in transects.items():
        wet=[i for i,p in enumerate(points) if p['biome']=='river']
        if not wet or min(wet)==0 or max(wet)==len(points)-1:continue
        previous=points[min(wet)-1];last=points[max(wet)+1]
        if previous['biome'] not in ['forest','grassland'] or last['biome'] not in ['forest','grassland']:continue
        low,high=sorted([previous['pos'][0],last['pos'][0]])
        innerlow,innerhigh=sorted([points[min(wet)]['pos'][0],points[max(wet)]['pos'][0]])
        bounds.append(dict(along=points[min(wet)]['pos'][1],dry_bounds=[low,high],wet_bounds=[innerlow,innerhigh]))
    common=min(b['wet_bounds'][1] for b in bounds)-max(b['wet_bounds'][0] for b in bounds) if len(bounds)==3 else None
    envelope=max(b['dry_bounds'][1] for b in bounds)-min(b['dry_bounds'][0] for b in bounds) if len(bounds)==3 else None
    return dict(seed=seed,site_index=index,elapsed_s=env.time,energy_used=150-a.energy,
        recording=finish_recording(recorder,replay_path,reason='transects completed' if leg>=len(goals) else 'survey stopped before route completion'),
        completed_transects=len(bounds),measured_common_water=common,measured_dry_envelope=envelope,
        maximum_odometry_error=max_error,bounds=bounds,trace=trace,
        qualified=bool(common is not None and common>=32 and envelope<=66 and max_error<.01),
        qualification='Candidate nomination and initial pose supplied; water boundaries reconstructed solely from public own biome along three paid crossings. Error is diagnostic truth, not available to scout.')

if __name__=='__main__':
    m={m['seed']:m['sites'] for m in json.loads((OUT/'sites-1-40.json').read_text())['data']}
    rows=[]
    for seed,i in [(23,2),(37,8),(37,10),(37,11)]:
        r=probe(seed,m[seed][i],i);rows.append(r)
        print({k:v for k,v in r.items() if k not in ['trace','bounds']},flush=True)
        save('observation-transects',rows)

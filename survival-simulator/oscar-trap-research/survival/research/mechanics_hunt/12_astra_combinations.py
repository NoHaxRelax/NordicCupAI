"""Local combination controls and observation-only generated-map comparisons."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import sys, json, math, random, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor'/'survival-simulator'))
sys.path.insert(0,str(ROOT/'research'))
from src.core import SimulationCore
from src.elements.environment import Environment
from src.elements.biome import Forest_biome
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment
from simple_policies import SimplePolicy

def act(a,spawn=False,distance=0):
    return (a.agent_id,ActionRequest(agent_id=a.agent_id,move_distance=distance,move_direction=0,turn_angle=0,spawn_agent=spawn))

def controls():
    out={}
    for label,energy,age,spawn,fruit in [
        ('young_idle',150,0,False,False),('old_idle',150,100,False,False),
        ('old_birth',150,100,True,False),('old_birth_food',150,100,True,True),
        ('food_cannot_fund_same_tick_birth',90,0,True,True),
        ('food_next_tick_birth',90,0,True,True)]:
        e=Environment(400,400,400,random.Random(112));e.biome_map.fill(Forest_biome())
        a=e.spawn_agent(x=200,y=200);a.energy=energy;a.age=age;a.max_age=60;a.direction=0
        if fruit:
            f=e.spawn_fruit(x=200,y=200)
            f.grow(40) # Explicit ripe-fruit fixture, natural reachable 60 energy.
        s=step_environment(e,[act(a,spawn)])
        if label=='food_next_tick_birth':s=step_environment(e,[act(a,True)])
        out[label]={'population':len(e.agents),'energies':[x.energy for x in e.agents],
                    'ages':[x.age for x in e.agents],'score':s['score']}
    assert out['food_cannot_fund_same_tick_birth']['population']==1
    assert out['food_next_tick_birth']['population']==2
    assert out['old_birth_food']['population']==2
    # Both consume the same fruit; birth diverts energy into a young 75-energy reserve.
    assert abs(out['old_birth_food']['energies'][1]-74.9)<1e-8
    return out

def emergency_controls():
    """Arranged open-ground encounters: normal founder traits/energy, hidden positions."""
    rows=[]
    for seed in range(4):
        for mode in ['idle','birth','sprint','sprint_birth']:
            e=Environment(400,400,400,random.Random(seed));e.biome_map.fill(Forest_biome())
            a=e.spawn_agent(x=200,y=200);a.direction=math.pi;a.energy=150
            p=e.spawn_predator(x=225,y=200);p.direction=math.pi;p.energy=100;p.resting=False
            s=step_environment(e,[act(a,'birth' in mode,20 if 'sprint' in mode else 0)])
            rows.append({'seed':seed,'mode':mode,'alive':len(e.agents),
                         'parent_alive':a.agent_id in e.agents_dict,'score':s['score'],
                         'remaining_energy':sum(x.energy for x in e.agents)})
    assert all(r['parent_alive'] for r in rows if 'sprint' in r['mode'])
    assert all(not r['parent_alive'] for r in rows if r['mode']=='idle')
    return rows

class ComboPolicy(SimplePolicy):
    def __init__(self,mode,seed):
        super().__init__('nursery',seed);self.variant=mode;self.renewed=set()
    def __call__(self,states,t):
        actions=super().__call__(states,t)
        byid={s['agent_id']:s for s in states}
        extra_slots=max(0,7-len(states)-sum(a.spawn_agent for _,a in actions))
        for aid,a in actions:
            s=byid[aid];obs=s['observations']
            danger=any(o['type']=='Predator' and o['distance']<95 for o in obs)
            food=any(o['type'] in ('Fruit','Tree') for o in obs)
            if self.variant in ('renewal','combined'):
                # One replacement per old parent, one bounded extra slot globally.
                if (extra_slots>0 and not a.spawn_agent and s['age']>80 and aid not in self.renewed
                    and s['energy']>125 and food and not danger):
                    a.spawn_agent=True
                    extra_slots-=1
                if a.spawn_agent and s['age']>80:self.renewed.add(aid)
            if self.variant in ('thrift','combined') and a.move_distance==0 and not danger:
                # Existing nursery scans every idle tick; scan once every ten ticks.
                if self.ticks%10:a.turn_angle=0
        return actions

def run(mode,seed,horizon):
    sim=SimulationCore(seed=seed);policy=ComboPolicy(mode,seed);s=sim.step([])
    turns=0.;births=0;peak=s['num_agents']
    while s['num_agents'] and s['sim_time']<horizon:
        acts=policy(s['observations'],s['sim_time'])
        assert len({i for i,a in acts})==len(acts)
        turns+=sum(min(math.pi,abs(a.turn_angle))/(2*math.pi) for _,a in acts)
        s=sim.step(acts);peak=max(peak,s['num_agents'])
    return {'mode':mode,'seed':seed,'horizon':horizon,'time':s['sim_time'],'score':s['score'],
            'alive':s['num_agents'],'created':sim.env._next_agent_id,'peak':peak,'turn_energy':turns}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--horizon',type=float,default=300)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,7,42]);args=p.parse_args()
    result={'source_commit':'acfc31a4003a5f91bf11032a02cd98c178ddbd7e','controls':controls(),
            'emergency_controls':emergency_controls(),'runs':[]}
    for seed in args.seeds:
        for mode in ['baseline','renewal','thrift','combined']:
            r=run(mode,seed,args.horizon);result['runs'].append(r);print(json.dumps(r),flush=True)
            target=ROOT/'results'/'mechanics_hunt'/'12_astra_combinations.json'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')

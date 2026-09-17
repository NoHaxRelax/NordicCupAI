"""Run unmodified upstream dynamics on generated maps. No online submissions."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import sys, pathlib, argparse, time, json, platform, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'/'survival-simulator'))
from src.core import SimulationCore
from simple_policies import SimplePolicy


def run(mode, seed, horizon):
    start=time.perf_counter()
    sim=SimulationCore(seed=seed)
    policy=SimplePolicy(mode,seed)
    state=sim.step([]) # same first empty step as official server
    peak=state['num_agents']; samples=[]
    while state['num_agents'] and state['sim_time']<horizon:
        actions=policy(state['observations'],state['sim_time'])
        state=sim.step(actions)
        peak=max(peak,state['num_agents'])
        if round(state['sim_time']*10)%500==0:
            samples.append({'time':round(state['sim_time'],2),'score':round(state['score'],4),
                'alive':state['num_agents'],'predators':len(sim.env.predators),
                'best_walk':max((min(a.speed,a.sprint_speed) for a in sim.env.agents),default=0)})
    result={'mode':mode,'seed':seed,'horizon':horizon,'survival_seconds':round(state['sim_time'],2),
        'score':round(state['score'],5),'alive':state['num_agents'],'peak_agents':peak,
        'total_agents_created':sim.env._next_agent_id,'predators':len(sim.env.predators),
        'wall_seconds':round(time.perf_counter()-start,2),'samples':samples,
        'platform':platform.platform(),'python':platform.python_version(),
        'policy_sha256':hashlib.sha256((ROOT/'research'/'simple_policies.py').read_bytes()).hexdigest(),
        'source_commit':'acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
        'limitations':'Local macOS generated maps; official evaluation uses Linux. No hidden state in policy. Read-only diagnostic metrics use engine state.'}
    path=ROOT/'results'/f'benchmark-{mode}-{seed}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','limitations')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['dummy','greedy','nursery','speed'],default='nursery')
    p.add_argument('--seeds',type=int,nargs='+',default=[1,7,42]);p.add_argument('--horizon',type=float,default=3000)
    a=p.parse_args()
    for seed in a.seeds:run(a.mode,seed,a.horizon)

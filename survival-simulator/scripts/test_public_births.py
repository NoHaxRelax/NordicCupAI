"""Check public birth constraints against a harness-only RNG trace in Python.

Instrumentation records returned random() values unchanged. The extractor receives
no RNG trace, seed, environment, or absolute positions from the harness.
"""
import os
os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.core import SimulationCore
from fastsim.fastpolicy import PolicySimulationCore
from src.utils.DTOs import ActionRequest
from models.seed_shadow.public_terrain import TerrainSamples
from models.seed_shadow.public_births import birth_block
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1904332854
real=SimulationCore(seed=seed);policy=PolicySimulationCore(seed=seed)
config=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
policy.policy_init(0,config);terrain=TerrainSamples();trace=[];original=real.rng.random
def tracked():
    value=original();trace.append(value);return value
real.rng.random=tracked
before=dict(sim_time=0.,observations=[real.env.get_agent_state(a.agent_id) for a in real.env.agents])
blocks=[];checks=0;errors=[];start=time.monotonic()
for tick in range(1800):
    actions=[(aid,ActionRequest(agent_id=aid,move_distance=d,move_direction=v,turn_angle=t,spawn_agent=z)) for aid,d,v,t,z in policy.policy_act()]
    trace.clear();after=real.step(actions);policy.step(actions);terrain.observe(after)
    block=birth_block(before,actions,after,terrain)
    if block:
        for i,draw in enumerate(block['draws']):
            lo,hi=draw['interval'];checks+=1
            if i>=len(trace) or not lo-1e-13<=trace[i]<=hi+1e-13:
                errors.append(dict(tick=tick,index=i,expected=draw,actual=trace[i] if i<len(trace) else None))
        blocks.append(block)
    before=after
    if not after['num_agents']:break
out=ROOT/'docs/seed-shadow'/f'birth-constraints-{seed}.json'
result=dict(seed=seed,blocks=len(blocks),births=sum(len(b['pairs']) for b in blocks),
    random_draw_constraints=checks,errors=errors,seconds=time.monotonic()-start,
    recovery_test=False,public_blocks=blocks)
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='public_blocks'}),flush=True)
assert not errors

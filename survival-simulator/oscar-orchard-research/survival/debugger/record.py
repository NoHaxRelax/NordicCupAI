"""Record existing simple policies, custom policies, or the controlled wall demo."""
from __future__ import annotations
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'/'survival-simulator'))
sys.path.insert(0,str(ROOT/'research'))
from recorder import ReplayRecorder
from src.core import SimulationCore
from simple_policies import SimplePolicy


def explain(states, actions, mode):
    """Readable labels for exact baseline branches; never supply engine state."""
    by_id=dict(actions);out={}
    for s in states:
        a=by_id.get(s['agent_id'])
        if a is None:continue
        obs=s['observations']
        predators=sorted((o for o in obs if o['type']=='Predator'),key=lambda o:o['distance'])
        fruits=sorted((o for o in obs if o['type']=='Fruit'),key=lambda o:o['distance'])
        trees=sorted((o for o in obs if o['type']=='Tree'),key=lambda o:o['distance'])
        if mode=='dummy':rule,detail='Random baseline','Random movement and turn, with reproduction requested.'
        elif predators and predators[0]['distance']<95:
            rule='Evade predator';detail=f"Closest observed predator: {predators[0]['distance']:.1f} units. Move away while looking toward it."
        elif fruits and (mode=='greedy' or s['energy']<s['max_energy']-55):
            rule='Collect fruit';detail=f"Nearest observed fruit: {fruits[0]['distance']:.1f} units. Aim to finish within collection range."
        elif mode!='greedy' and trees and trees[0]['distance']<75 and s['energy']>50:
            rule='Stay near food';detail=f"Tree {trees[0]['distance']:.1f} units away. Approach to 35 units, then conserve movement and scan."
        else:rule,detail='Explore','No immediate food or predator priority. Walk and gradually scan for resources.'
        if a.spawn_agent:detail+=' Reproduction requested because reserves and breeding rules allow it.'
        if any(o['type']=='Edge' for o in obs):detail+=' Nearby visible wall segments can adjust the final movement.'
        out[s['agent_id']]=dict(rule=rule,detail=detail)
    return out


def custom_policy(spec, seed):
    path,sep,name=spec.rpartition(':')
    if not sep:raise ValueError('--policy requires /path/to/policy.py:factory')
    path=Path(path).expanduser().resolve()
    module_spec=importlib.util.spec_from_file_location('survival_user_policy',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    policy=getattr(module,name)(seed=seed)
    return policy,hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.every<1 or args.seconds<=0:raise ValueError('every and seconds must be positive')
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Save under survival/results/<strategy>/replays/ so Survival Lab can discover this run.')
    if args.output.exists() and not args.overwrite:raise FileExistsError(args.output)
    if args.scenario=='wall':
        from predator_control import controlled_env,add_agent,add_predator
        from src.elements.obstacle import Obstacle
        env=controlled_env()
        wall=Obstacle(784,565,width=32,height=70)
        env.obstacles=[wall]
        env.edges.update((min(a,b),max(a,b)) for a,b in wall.edges)
        a=add_agent(env,821,600,energy=150);a.max_age=120;a.direction=math.pi
        add_predator(env,769,600,heading=0)
        env._update_spatial_grid()
        recorder=ReplayRecorder(env,title=args.title or 'Wall bait · controlled experiment',policy='stationary wall bait',
            seed=1729,every=args.every,scenario='controlled',notes='Arranged wall and creature positions; no food or new predators. Tests holding a trap, not acquiring it in a generated game.',
            native_render=args.native_render,native_width=args.native_width)
        recorder.capture(force=True)
        while env.agents and env.time<args.seconds-1e-8:
            old_time=env.time
            env.non_agent_step(.1)
            recorder.capture(decisions={0:dict(rule='Hold wall bait',detail='Remain still behind the wall. The predator senses the decoy but the obstacle blocks its route.')},action_t=old_time)
        summary=recorder.save(args.output,reason='species extinct' if not env.agents else 'requested duration reached',overwrite=args.overwrite)
    else:
        sim=SimulationCore(seed=args.seed)
        if args.policy:
            policy,digest=custom_policy(args.policy,args.seed);label=Path(args.policy.split(':')[0]).name
        else:
            policy=SimplePolicy(args.mode,args.seed);digest=hashlib.sha256((ROOT/'research/simple_policies.py').read_bytes()).hexdigest();label=args.mode
        recorder=ReplayRecorder(sim.env,title=args.title or f'{label.title()} · seed {args.seed}',policy=label,seed=args.seed,
            every=args.every,policy_sha256=digest,notes='Local generated-map run. Debugger sees engine state; policy receives ordinary observation DTOs only.',
            native_render=args.native_render,native_width=args.native_width)
        recorder.capture(force=True)
        state=sim.step([]);recorder.capture()
        while state['num_agents'] and state['sim_time']<args.seconds-1e-8:
            before=state['observations'];old_time=state['sim_time']
            actions=policy(before,old_time)
            decisions=explain(before,actions,args.mode) if not args.policy else getattr(policy,'last_decisions',{})
            state=sim.step(actions)
            recorder.capture(actions,decisions,before,old_time)
        recorder.capture(actions if 'actions' in locals() else (),decisions if 'decisions' in locals() else {},
                         before if 'before' in locals() else [],old_time if 'old_time' in locals() else None,force=True)
        summary=recorder.save(args.output,reason='species extinct' if not state['num_agents'] else 'requested duration reached',overwrite=args.overwrite)
    print(json.dumps(dict(file=str(args.output),**summary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['dummy','greedy','nursery','speed'],default='nursery')
    p.add_argument('--policy',help='Custom Python file:factory. Factory accepts seed= and returns callable(states,time).')
    p.add_argument('--scenario',choices=['generated','wall'],default='generated')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--seconds',type=float,default=3000)
    p.add_argument('--every',type=int,default=1,help='Store one world frame every N ticks; events still tracked each tick.')
    rendering=p.add_mutually_exclusive_group()
    rendering.add_argument('--native-render',dest='native_render',action='store_true',default=True,help='Capture the original Environment.draw output in each frame (default).')
    rendering.add_argument('--state-render',dest='native_render',action='store_false',help='Store state only and use the browser fallback renderer (smaller files).')
    p.add_argument('--native-width',type=int,default=960,help='Width of native rendering in pixels (160–2400; default 960).')
    p.add_argument('--title');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--register',action='store_true',help='Compatibility flag: every completed replay in survival/ is discovered automatically.')
    p.add_argument('--overwrite',action='store_true')
    run(p.parse_args())

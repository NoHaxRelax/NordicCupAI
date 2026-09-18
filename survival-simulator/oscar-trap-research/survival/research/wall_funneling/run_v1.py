"""Repeated wall acquisition, with explicit unlimited agent food and arranged arrivals.

Unchanged native movement, collision, predator AI/rest, sensing and capture.
Exact world coordinates are controller inputs. This is a local feasibility test,
not an observation-only policy or a generated-game success claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(ROOT / 'debugger'))
from predator_control import controlled_env, add_predator, wrap
from src.elements.obstacle import Obstacle
from src.utils.DTOs import ActionRequest
from recorder import ReplayRecorder

OUT = ROOT / 'results' / 'wall_funneling'


def run(width=30, length=100, mode='sacrifice', heading=0., offset=0.,
        waves=3, interval=12., seconds=60., peel=35., native=False, seed=1729):
    assert 30 <= width <= length <= 100
    assert seconds > (waves - 1) * interval + 10
    env = controlled_env()
    env.rng.seed(seed)
    wall = Obstacle(800-width/2, 600-length/2, width, length)
    env.obstacles = [wall, Obstacle(0,0,1600,30), Obstacle(0,1170,1600,30),
                     Obstacle(0,0,30,1200), Obstacle(1570,0,30,1200)]
    env.edges = {tuple(sorted((a,b))) for o in env.obstacles for a,b in o.edges}
    anchor = (wall.x+width+7, 600)
    holder = env.spawn_agent(x=anchor[0], y=anchor[1])
    holder.direction = math.pi
    env._update_spatial_grid()
    tag = f'{mode}-w{width}-l{length}-h{heading}-o{offset}-p{peel}-s{seed}-{uuid4().hex[:8]}'
    path = OUT / 'replays' / f'{tag}.json.gz'
    rec = ReplayRecorder(env, title=f'Food-free funnel: {tag}', policy='wall-funneling-v1',
        seed=seed, every=10, scenario='arranged sequential arrivals on flat forest',
        notes=__doc__+' Agent energy reset to max before every tick, including guides. '
        'A fresh guide and awake predator are introduced each wave. No food, trees, '
        'native random spawns, other obstacles or automatic guide replacement. '
        'All dimensions are within the native obstacle generator range.',
        policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_render=native, native_width=800)
    rec.capture()
    deliveries = []
    deaths = []
    trace = []
    injected_energy = 0.
    for tick in range(round(seconds*10)):
        if holder.agent_id not in env.agents_dict:
            break
        gate_open = mode != 'rest_gate' or all(
            d['predator'].resting and d['predator'].energy <= 40 for d in deliveries)
        due = not deliveries or env.time >= deliveries[-1]['arrival']+interval-.01
        if len(deliveries) < waves and due and gate_open:
            guide = env.spawn_agent(x=wall.x-125, y=600+offset)
            p = add_predator(env, guide.x-60, guide.y, heading=heading)
            guide.direction = math.pi+.02
            deliveries.append(dict(guide=guide, predator=p, phase='lead', waypoints=[],
                arrival=round(env.time,1), acquired=None, streak=0, first_loss=None,
                held_ticks=0, tail=[], switches=0))
        before = list(env.agents)
        for a in before:
            injected_energy += a.max_energy-a.energy
            a.energy = a.max_energy
        pairs, decisions = [], {}
        action_t = env.time
        for a in before:
            target = (a.x,a.y)
            speed = a.sprint_speed
            if a is holder:
                rule = 'stationary far-face holder'
            else:
                d = next(d for d in deliveries if d['guide'] is a)
                p = d['predator']
                distance = math.dist((a.x,a.y),(p.x,p.y))
                if d['phase'] == 'lead':
                    target = (wall.x-7, 600)
                    if distance > 75:
                        target = (a.x,a.y)
                    if distance > 45:
                        speed = a.speed
                    if mode == 'loop' and distance < peel and a.x > wall.x-45:
                        d['phase'] = 'loop'
                        d['waypoints'] = [(wall.x-7, wall.y-7),
                            (wall.x+width+7,wall.y-7),anchor]
                    elif mode == 'peel' and distance < peel and a.x > wall.x-45:
                        d['phase'] = 'loop'
                        d['waypoints'] = [(a.x,wall.y-130),(wall.x-200,wall.y-130)]
                if d['phase'] == 'loop':
                    while d['waypoints'] and math.dist((a.x,a.y),d['waypoints'][0]) < 1:
                        d['waypoints'].pop(0)
                    target = d['waypoints'][0] if d['waypoints'] else (a.x,a.y)
                    speed = a.sprint_speed
                rule = mode + ':' + d['phase']
            dx,dy = target[0]-a.x,target[1]-a.y
            distance = min(speed, math.hypot(dx,dy))
            direction = wrap(math.atan2(dy,dx)-a.direction) if distance else 0.
            action = ActionRequest(agent_id=a.agent_id,move_distance=distance,
                move_direction=direction,turn_angle=0.,spawn_agent=False)
            assert 0 <= distance <= a.sprint_speed
            pairs.append((a.agent_id,action))
            decisions[a.agent_id] = {'rule':rule}
            env.agent_step(a.agent_id,distance,direction,0.,False)
        env.non_agent_step(.1)
        deaths.extend({'agent_id':a.agent_id,'time':round(env.time,1),
                       'role':'holder' if a is holder else 'guide'}
                      for a in before if a.agent_id not in env.agents_dict)
        for d in deliveries:
            p = d['predator']
            seen = [o for o in p.observe(agents=list(env.agents),edges=list(env.edges))
                    if o['type']=='Agent']
            chosen = min(seen,key=lambda o:o['distance'])['id'] if seen else None
            # Require actual target attention and hearing, not just survival/proximity.
            held = (holder.agent_id in env.agents_dict and chosen == holder.agent_id
                and math.dist((p.x,p.y),anchor) <= 60
                and wall.x-55 < p.x <= wall.x-10
                and wall.y+5 < p.y < wall.y+length-5)
            d['streak'] = d['streak']+1 if held else 0
            if d['acquired'] is None and d['streak'] >= 20:
                d['acquired'] = round(env.time-1.9,1)
            if d['acquired'] is not None and not held and d['first_loss'] is None:
                d['first_loss'] = round(env.time,1)
            d['switches'] += int(d['acquired'] is not None and chosen != holder.agent_id)
            d['held_ticks'] += int(held)
            d['tail'].append(bool(held))
        if tick % 10 == 0:
            trace.append(dict(time=round(env.time,1),held=sum(d['tail'][-1] for d in deliveries),
                              total=len(deliveries)))
        rec.capture(pairs,decisions,action_t=action_t)
    rec.capture(force=True)
    rec.save(path,reason='horizon' if env.time >= seconds-.01 else 'holder captured')
    items = [dict(arrival=d['arrival'],acquired=d['acquired'],first_loss=d['first_loss'],
        guide_alive=d['guide'].agent_id in env.agents_dict,
        tail_hold=sum(d['tail'][-100:])/len(d['tail'][-100:]),
        target_loss_ticks=d['switches']) for d in deliveries]
    result = dict(width=width,length=length,mode=mode,heading=heading,offset=offset,peel=peel,
        seed=seed,waves=waves,interval=interval,requested_seconds=seconds,seconds=round(env.time,1),
        holder_alive=holder.agent_id in env.agents_dict,deaths=deaths,deliveries=items,
        continuous_success=len(items)==waves and env.time>=seconds-.01 and
            all(d['acquired'] is not None and d['first_loss'] is None for d in items),
        final_success=len(items)==waves and env.time>=seconds-.01 and
            all(d['tail_hold']==1 for d in items),
        injected_energy=round(injected_energy,2),trace=trace,replay=str(path.relative_to(ROOT)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / f'{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=float,default=30)
    parser.add_argument('--length',type=float,default=100)
    parser.add_argument('--mode',choices=['sacrifice','loop','peel','rest_gate'],default='sacrifice')
    parser.add_argument('--heading',type=float,default=0.)
    parser.add_argument('--offset',type=float,default=0.)
    parser.add_argument('--peel',type=float,default=35.)
    parser.add_argument('--waves',type=int,default=3)
    parser.add_argument('--interval',type=float,default=12.)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--native',action='store_true')
    args = vars(parser.parse_args())
    print(json.dumps(run(**args),indent=2))

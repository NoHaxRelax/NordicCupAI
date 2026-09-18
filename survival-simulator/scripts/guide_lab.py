"""Run models/entrapment/my_guide.py on native random maps; retain every tick and image.

From the repository root:
  .venv/bin/python survival-simulator/scripts/guide_lab.py --seed 10224 --serve
  .venv/bin/python survival-simulator/scripts/guide_lab.py --seed 100 --maps 5
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pygame
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest
from guide_lab_sites import enumerate_sites, _Geometry


def local(point, agent):
    dx, dy = point[0] - agent.x, point[1] - agent.y
    c, s = math.cos(agent.direction), math.sin(agent.direction)
    return (dx*c + dy*s, -dx*s + dy*c)


def observations(env, agent):
    """Native sensing at the current pose, also used for initial t=0 input."""
    return agent.observe(agents=env._get_local_agents(agent),
                         fruits=env._get_local_fruits(agent),
                         trees=env._get_local_trees(agent),
                         predators=env._get_local_predators(agent),
                         edges=env._get_local_edges(agent))


def choose_site(env, index=0):
    from models.entrapment.observed_trap_sites import our_sites
    geometry = _Geometry(env.width, env.height,
                         [(o.x, o.y, o.width, o.height) for o in env.obstacles])
    static = dict(width=env.width, height=env.height, obstacles=geometry.rects)
    usable = our_sites(static)
    if index < 0 or index >= len(usable):
        raise ValueError(f'No usable site at index {index}; map has {len(usable)} eligible sites')
    return usable[index], geometry, len(usable)


def setup(seed, encounter_seed, site_index):
    core = SimulationCore(seed=seed, starting_agents=0, starting_predators=0)
    env = core.env
    site, geometry, count = choose_site(env, site_index)
    rng = random.Random(encounter_seed)
    bait = Agent(*site['goal'], rng=env.rng, color=(60, 240, 100))
    bait.agent_id = 0
    bait.energy = bait.max_energy
    guide = Agent(0., 0., rng=env.rng, color=(60, 220, 255))
    guide.agent_id = 1
    guide.energy = guide.max_energy
    env.agents = [bait, guide]
    env.agents_dict = {a.agent_id: a for a in env.agents}
    env._next_agent_id = 2
    predator = Predator(0., 0., rng=env.rng)
    predator.energy = predator.max_energy
    predator.resting = False
    env.predators = [predator]
    for _ in range(10000):
        p = rng.uniform(45, env.width-45), rng.uniform(45, env.height-45)
        heading = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(80, 160)
        q = p[0]+distance*math.cos(heading), p[1]+distance*math.sin(heading)
        if (not geometry.free(p, 5.01) or not geometry.free(q, 10.01)
                or math.dist(p, site['goal']) < 250):
            continue
        guide.x, guide.y = p
        guide.direction = heading
        predator.x, predator.y = q
        predator.direction = heading + math.pi
        env._update_agent_grid()
        env._update_predator_grid()
        seen = observations(env, guide)
        if any(o['type'] == 'Predator' for o in seen):
            env.agent_observations[guide.agent_id] = seen
            return core, site, count, bait, guide, predator
    raise ValueError('Could not sample a visible predator encounter on this map')


def validate_action(value, agent):
    if not isinstance(value, dict):
        raise TypeError('guide() must return a dictionary')
    required = {'move_distance', 'move_direction', 'turn_angle'}
    if set(value) != required:
        raise ValueError(f'Return exactly {sorted(required)}; got {sorted(value)}')
    values = {k: float(value[k]) for k in required}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Action values must be finite numbers')
    if not 0 <= values['move_distance'] <= agent.sprint_speed:
        raise ValueError(f'move_distance must be between 0 and {agent.sprint_speed}')
    return ActionRequest(agent_id=agent.agent_id, spawn_agent=False, **values)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args, seed, encounter_seed):
    started = time.monotonic()
    bulk = getattr(args, 'bulk', False)
    tag = f'map-{seed}-encounter-{encounter_seed}-{time.time_ns()}'
    folder = args.output / tag
    folder.mkdir(parents=True)
    (folder/'frames').mkdir()
    (folder/'ticks').mkdir()
    shutil.copy2(args.policy, folder/'policy.py')
    for name in ('guide_pathfinding.py','predator_following.py','guide_steering.py'):
        helper = ROOT/'models/entrapment'/name
        if helper.exists():
            shutil.copy2(helper, folder/name)
    shutil.copy2(__file__, folder/'runner.py')
    shutil.copy2(Path(__file__).with_name('guide_lab_sites.py'), folder/'site_selector.py')
    shutil.copy2(Path(__file__).with_suffix('.html'), folder/'index.html')
    summary = dict(seed=seed, encounter_seed=encounter_seed, site_index=args.site,
                   policy=str(args.policy), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                   frames=0, seconds=0., outcome='setup_error', guide_caught=None,
                   assumptions=['native random map; full static geometry and exact local transforms',
                                'arranged random visible encounter, 80–160 units apart; predator awake/full initially',
                                'guide full only at start; native energy, aging, food and predator dynamics thereafter',
                                'bait predeployed, stationary, full energy each tick; aging disabled, can still be eaten',
                                'native ambient predator spawning remains enabled',
                                'success is a local delivery proxy, not full-game retention or proof of causation'],
                   success_rule=f'tracked predator within 40 units of bait, sensing bait, for {args.hold:g}s continuously',
                   simulator_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((ROOT/'src').rglob('*.py'))})
    try:
        spec = importlib.util.spec_from_file_location(f'guide_policy_{time.time_ns()}', args.policy)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy = module.guide
        core, site, site_count, bait, guide, tracked = setup(seed, encounter_seed, args.site)
    except Exception:
        summary['error'] = traceback.format_exc()
        write_json(folder/'summary.json', summary)
        print(f'{tag}: SETUP ERROR\n{summary["error"]}', flush=True)
        return summary, folder
    env = core.env
    summary.update(site=site, eligible_sites=site_count, dt=core.dt,
                   start=dict(guide=[guide.x,guide.y,guide.direction],
                              predator=[tracked.x,tracked.y,tracked.direction], bait=site['goal']))
    memory, held_since, death_time = {}, None, None
    # Track contact at the native kill event, before another predator moves.
    native_kill = env.kill_agent
    caught = {}
    def note_death(agent):
        if agent is guide:
            touching = [p for p in env.predators if math.hypot(p.x-guide.x,p.y-guide.y)<p.size+guide.size]
            if guide.energy > 0 and touching:
                hunter = min(touching, key=lambda p:math.hypot(p.x-guide.x,p.y-guide.y))
                sensed = hunter.observe(agents=env._get_local_agents(hunter), edges=env._get_local_edges(hunter))
                caught.update(time=round(env.time+core.dt, 6), tracked_predator=hunter is tracked,
                              predator_bait_distance=math.hypot(hunter.x-bait.x,hunter.y-bait.y),
                              bait_sensed=any(o.get('id')==bait.agent_id for o in sensed if o['type']=='Agent'))
        return native_kill(agent)
    env.kill_agent = note_death
    screen = None if bulk else pygame.Surface((args.width, args.width*3//4))
    limit = round(args.seconds/core.dt)
    alive_ticks = detectable_ticks = 0
    with gzip.open(folder/'ticks.jsonl.gz', 'wt') as trace:
        for tick in range(limit+1):
            now = tick*core.dt
            alive = guide in env.agents
            if not alive and death_time is None:
                death_time = now
            bait_alive = bait in env.agents
            sensed = tracked.observe(agents=env._get_local_agents(tracked), edges=env._get_local_edges(tracked))
            bait_sensed = any(o['type']=='Agent' and o.get('id')==bait.agent_id for o in sensed)
            guide_sensed = any(o['type']=='Agent' and o.get('id')==guide.agent_id for o in sensed)
            if alive:
                alive_ticks += 1
                detectable_ticks += int(guide_sensed)
            near = bait_alive and math.hypot(tracked.x-bait.x, tracked.y-bait.y)<=40 and bait_sensed
            held_since = (now if held_since is None else held_since) if near else None
            success = held_since is not None and now-held_since >= args.hold-1e-8
            terminal = (success or not bait_alive or tick==limit
                        or (death_time is not None and now-death_time>=args.hold+10))
            inputs, action, error = None, None, None
            if alive and not terminal:
                # Preserve the simulator's observation timing: at t>0 the DTO
                # was computed before the previous predator movement.
                inputs = dict(bait=local(site['goal'],guide),
                              edges=[[local(a,guide),local(b,guide)] for a,b in env.edges],
                              agent=copy.deepcopy(env.get_agent_state(guide.agent_id)),
                              context=dict(tick=tick,dt=core.dt,time=now,
                                           handoff=local(site['handoff'],guide),
                                           mouth=local(site['mouth'],guide)))
                try:
                    if args.break_at == tick:
                        breakpoint()
                    value = policy(**copy.deepcopy(inputs), memory=memory)
                    action = validate_action(value, guide)
                    json.dumps(memory.get('debug'), allow_nan=False)
                except Exception:
                    error = traceback.format_exc()
                    terminal = True
            evaluation = dict(guide_alive=alive, guide_energy=guide.energy if alive else None,
                              guide_sensed_by_tracked_predator=guide_sensed,
                              bait_alive=bait_alive, bait_energy=bait.energy,
                              predator_bait_distance=math.hypot(tracked.x-bait.x,tracked.y-bait.y),
                              bait_sensed_by_tracked_predator=bait_sensed,
                              continuous_near_bait_seconds=0 if held_since is None else now-held_since,
                              capture=caught or None, total_predators=len(env.predators))
            event = ('Delivery proxy passed' if success else 'Bait died' if not bait_alive else
                     'Policy error' if error else 'Guide died; observing handoff' if not alive else '')
            row = dict(tick=tick,time=now,input=inputs,action=action.model_dump() if action else None,
                       debug=memory.get('debug') if not error else repr(memory.get('debug')),
                       evaluation=evaluation,event=event,error=error)
            if not bulk:
                write_json(folder/'ticks'/f'{tick}.json', row)
            trace.write(json.dumps(row, allow_nan=False)+'\n')
            if not bulk:
                env.draw(screen)
                scale = args.width/env.width
                for point, color in ((site['handoff'],(255,220,60)),(site['goal'],(60,240,100))):
                    pygame.draw.circle(screen,color,(round(point[0]*scale),round(point[1]*scale)),6,2)
                pygame.image.save(screen,folder/'frames'/f'{tick}.png')
            summary.update(frames=tick+1,seconds=now,guide_caught=caught or None,final=evaluation)
            if terminal:
                summary['outcome'] = ('delivery_proxy_pass' if success else 'policy_error' if error else
                                      'bait_dead' if not bait_alive else 'guide_dead_without_delivery' if not alive
                                      else 'timeout')
                if error: summary['error'] = error
                break
            # Fixture bait only: the user's guide is never refilled/rejuvenated.
            bait.energy = bait.max_energy
            bait.age = 0.
            core.step([(guide.agent_id,action)] if action else [])
            bait.energy = bait.max_energy
            bait.age = 0.
            if tick and tick%100==0:
                print(f'{tag}: {now:.1f}s, guide energy {guide.energy:.1f}',flush=True)
    summary['elapsed_seconds'] = round(time.monotonic()-started,3)
    summary['recording'] = 'every_tick_gzip_no_png' if bulk else 'every_tick_json_and_png'
    summary['contact_metrics'] = dict(alive_ticks=alive_ticks, detectable_ticks=detectable_ticks,
                                   detectable_fraction=detectable_ticks/alive_ticks if alive_ticks else None)
    write_json(folder/'summary.json',summary)
    print(f'{tag}: {summary["outcome"]} at {summary["seconds"]:.1f}s; {summary["frames"]} frames',flush=True)
    if summary.get('error'):
        print(summary['error'],flush=True)
    return summary, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed',type=int,help='Map seed; omitted = random')
    parser.add_argument('--encounter-seed',type=int,help='Change starting encounter without changing map')
    parser.add_argument('--maps',type=int,default=1,help='Sequential map seeds; one worker')
    parser.add_argument('--site',type=int,default=0,help='Choose another eligible bait site on the map')
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--hold',type=float,default=10,help='Continuous seconds near bait for delivery proxy')
    parser.add_argument('--policy',type=Path,default=ROOT/'models/entrapment/my_guide.py')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/guide_lab')
    parser.add_argument('--width',type=int,default=640,help='Native replay image width')
    parser.add_argument('--break-at',type=int,help='Pause in pdb before policy call at this tick')
    parser.add_argument('--bulk',action='store_true',help='Keep every tick in gzip; skip PNG and duplicate tick files')
    parser.add_argument('--serve',action='store_true',help='Serve saved replays after running; Ctrl+C stops server')
    parser.add_argument('--serve-only',action='store_true',help='Serve existing replays without running')
    parser.add_argument('--port',type=int,default=9056)
    args = parser.parse_args()
    if args.maps<1 or not math.isfinite(args.seconds) or args.seconds<=0 or not math.isfinite(args.hold) or args.hold<=0 or args.width<160:
        parser.error('maps, seconds, hold must be positive; width must be at least 160')
    args.policy = args.policy.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.serve_only:
        pygame.init()
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
        results=[]
        for offset in range(args.maps):
            encounter = (args.encounter_seed if args.encounter_seed is not None else seed+10000)+offset
            summary,folder = run(args,seed+offset,encounter)
            results.append(summary)
            print(f'Replay: http://127.0.0.1:{args.port}/{folder.name}/index.html',flush=True)
        batch=args.output/f'batch-{time.time_ns()}.json'
        write_json(batch,results)
        pygame.quit()
        print(f'Summary: {batch}',flush=True)
    if args.serve or args.serve_only:
        print(f'Replay library: http://127.0.0.1:{args.port}/ — Ctrl+C to stop',flush=True)
        server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(args.output)))
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.server_close()
    else:
        print('To view: python survival-simulator/scripts/guide_lab.py --serve-only',flush=True)


if __name__ == '__main__':
    main()

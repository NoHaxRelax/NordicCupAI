"""Controlled probes of upstream predator mechanics, not a production policy.

Uses the unmodified upstream Environment movement, sensing and tick methods.
Synthetic maps disable new trees/predators and use assigned start positions.
Controller geometries and predator energy/rest state are diagnostic privileges.
Run from anywhere with the shared survival Python environment.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "survival-simulator"))
import numpy as np
from src.elements.environment import Environment
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.elements.biome import Forest_biome, River_biome


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def controlled_env(river_width=0):
    """Skip rendering/random world generation, preserve gameplay methods."""
    env = Environment.__new__(Environment)
    env.width, env.height, env.chunk_size = 1600, 1200, 400
    env.rng = random.Random(1729)
    env.agents, env.predators, env.fruits, env.trees, env.obstacles = [], [], [], [], []
    # Upstream sensing currently crashes on an entirely empty edge list;
    # use far-away arena boundaries, as real environments have them.
    env.edges = {((0,0),(1600,0)),((1600,0),(1600,1200)),
                 ((0,1200),(1600,1200)),((0,0),(0,1200))}
    env.agents_dict, env.fruits_dict, env.agent_observations = {}, {}, {}
    env._next_agent_id = env._next_fruit_id = 0
    env.score = env.time = 0
    env.biome_map = np.empty((env.width, env.height), dtype=object)
    env.biome_map.fill(Forest_biome())
    if river_width:
        env.biome_map[800-river_width//2:800+river_width//2, :] = River_biome()
    # Isolate the fixed creatures; no external food or new opponents.
    env.spawn_tree = lambda *args, **kwargs: None
    env.spawn_predator = lambda *args, **kwargs: None
    env._update_spatial_grid()
    return env


def add_agent(env, x, y, energy=500):
    agent = Agent(x, y, energy=energy, max_age=120, rng=env.rng)
    agent.agent_id = len(env.agents)
    env.agents.append(agent)
    env.agents_dict[agent.agent_id] = agent
    env._update_agent_grid()
    return agent


def add_predator(env, x=800, y=600, heading=0, energy=102):
    predator = Predator(x, y, energy=energy, rng=env.rng)
    predator.direction = heading
    predator.resting = False
    env.predators.append(predator)
    env._update_predator_grid()
    return predator


def action_toward(env, agent, x, y, walk=True, face=None):
    dx, dy = x-agent.x, y-agent.y
    distance = math.hypot(dx, dy)
    biome = env.biome_map[int(agent.x), int(agent.y)]
    requested = min(distance / biome.move_penalty, agent.speed if walk else agent.sprint_speed)
    relative = wrap(math.atan2(dy, dx)-agent.direction)
    turn = 0 if face is None else wrap(face-agent.direction)
    env.agent_step(agent.agent_id, requested, relative, turn, False)
    return requested


def movement_probe():
    rows = []
    for terrain in ["forest", "river"]:
        env = controlled_env(400 if terrain == "river" else 0)
        for kind, cls, request in [("agent_walk", Agent, 10), ("predator_sprint", Predator, 15)]:
            creature = cls(800, 600, energy=150, rng=env.rng)
            creature.direction = 0
            env.update_entity_position(creature, request, 0, [])
            rows.append(dict(terrain=terrain, creature=kind, distance=creature.x-800, energy_cost=150-creature.energy))
    return rows


def sensing_probe():
    rows = []
    for width in [40, 80, 120, 160, 200]:
        env = controlled_env(width)
        p = add_predator(env)
        left = add_agent(env, 800-width/2-5, 600)
        right = add_agent(env, 800+width/2+5, 600)
        left.direction, right.direction = 0, math.pi
        for offset in [0, width/4, width/2-5]:
            p.x = 800+offset
            seen = p.observe(agents=[left,right], edges=list(env.edges))
            agents = [o for o in seen if o["type"] == "Agent"]
            rows.append(dict(width=width, offset=offset, left_distance=p.x-left.x, right_distance=right.x-p.x,
                             observed_ids=[o["id"] for o in agents],
                             selected=min(agents,key=lambda a:a["distance"])["id"] if agents else None))
    return rows


def active_cycle_probe(river=False):
    env = controlled_env(800 if river else 0)
    p = add_predator(env, energy=102)
    # Feed a synthetic straight-ahead sensed target without changing predator
    # logic; move with actual environment energy/biome handling.
    active = sleep = 0
    distance = 0
    start_energy = p.energy
    while p.energy > 0:
        signals = p.step([dict(type="Agent", distance=200, angle=0, rel_dir=math.pi)])
        previous_x = p.x
        env.update_entity_position(p, signals["move"], signals["direction"], [])
        # Keep inside the homogeneous test terrain so map boundaries cannot
        # confound accumulated motion. This reset is diagnostic only.
        distance += abs(p.x-previous_x)
        p.x = 800
        active += 1
    energy_after = p.energy
    while p.energy <= 100:
        p.energy += 3
        sleep += 1
    return dict(terrain="river" if river else "forest", initial_energy=start_energy,
                active_ticks=active, sleep_ticks=sleep, active_distance=distance,
                energy_after_active=energy_after, wake_energy=p.energy)


def paired_lure(width, ticks=600, switch_x=0, retreat=50):
    """Generous idealized pair: known geometry and instant role allocation.

    Active decoy on its bank, reserve decoy retreats away from its bank.
    Swap roles whenever predator crosses switch_x from the river center.
    Agents stay out of water and move at ordinary walking speed.
    """
    env = controlled_env(width)
    p = add_predator(env)
    agents = [add_agent(env, 800-width/2-5-retreat, 600),
              add_agent(env, 800+width/2+5, 600)]
    role = 1
    switches = 0
    river_ticks = active_river_ticks = active_ticks = 0
    min_gap = float("inf")
    positions = []
    start_energy = sum(a.energy for a in agents)
    death_tick = None
    for tick in range(ticks):
        if role == 1 and p.x > 800+switch_x:
            role = 0
            switches += 1
        elif role == 0 and p.x < 800-switch_x:
            role = 1
            switches += 1
        for i,a in enumerate(agents):
            if a.agent_id not in env.agents_dict:
                continue
            side = -1 if i == 0 else 1
            goal_x = 800+side*(width/2+5+(0 if i==role else retreat))
            # Face predator; movement and looking direction are independent.
            face = math.atan2(p.y-a.y, p.x-a.x)
            action_toward(env,a,goal_x,600,face=face)
        active_ticks += not p.resting
        inside = abs(p.x-800) < width/2
        river_ticks += inside
        active_river_ticks += inside and not p.resting
        alive_before=list(env.agents)
        env.non_agent_step(0.1)
        if alive_before:
            min_gap = min(min_gap, *(math.hypot(a.x-p.x,a.y-p.y) for a in alive_before))
        if len(env.agents) < 2 and death_tick is None:
            death_tick = tick+1
        if tick%10 == 0:
            positions.append([tick/10, round(p.x,2),round(p.y,2),round(p.energy,2),p.resting,len(env.agents),
                              *[[round(a.x,2),round(a.y,2)] for a in agents]])
        if not env.agents:
            break
    elapsed = tick+1
    return dict(width=width, switch_x=switch_x, retreat=retreat, ticks=elapsed,
                agents_alive=len(env.agents), first_agent_death_s=None if death_tick is None else death_tick/10,
                predator_river_fraction=river_ticks/elapsed,
                predator_active_river_fraction=active_river_ticks/max(1,active_ticks),
                role_switches=switches, min_gap=min_gap,
                energy_remaining=sum(a.energy for a in env.agents),
                starting_energy=start_energy, trace=positions)


def wall_lure(ticks=600, width=20, height=200, offset=0, heading=0, max_age=120):
    """One stationary agent behind a thin obstacle, both within hearing."""
    from src.elements.obstacle import Obstacle
    env=controlled_env()
    wall=Obstacle(800-width/2,600-height/2,width=width,height=height)
    env.obstacles=[wall]
    env.edges.update((min(start,end),max(start,end)) for start,end in wall.edges)
    a=add_agent(env,800+width/2+5,600,energy=150)
    a.max_age=max_age
    p=add_predator(env,800-width/2-15,600+offset,heading=heading)
    env._update_spatial_grid()
    min_gap=math.inf
    positions=[]
    trapped_ticks=sensed_ticks=0
    max_center_distance=0
    for tick in range(ticks):
        sensed=p.observe(agents=[a],edges=list(env.edges))
        sensed_ticks+=any(o["type"]=="Agent" for o in sensed)
        env.non_agent_step(.1)
        min_gap=min(min_gap, math.hypot(a.x-p.x,a.y-p.y))
        trapped_ticks+=(abs(p.x-(wall.x-10))<=20 and abs(p.y-600)<=height/2)
        max_center_distance=max(max_center_distance,math.hypot(p.x-(wall.x-10),p.y-600))
        if tick%10==0:
            positions.append([tick/10,round(p.x,2),round(p.y,2),round(p.energy,2),p.resting,len(env.agents)])
        if not env.agents:
            break
    return dict(width=width,height=height,offset=offset,heading=heading,max_age=max_age,ticks=tick+1,alive=len(env.agents),
                min_gap=min_gap,energy=a.energy,predator_position=[p.x,p.y],
                trapped_fraction=trapped_ticks/(tick+1),predator_sensed_agent_fraction=sensed_ticks/(tick+1),
                max_center_distance=max_center_distance,trace=positions)


def orbit_lure(gap=40, initial_bearing=0, radius=40, mode="orbit", ticks=600):
    """Single walking lure driven only by real cached agent observations.

    Starting state is arranged; after that the action uses exactly the server
    observation fields. No energy/rest oracle and no current predator position.
    """
    env=controlled_env()
    a=add_agent(env,800+gap*math.cos(initial_bearing),600+gap*math.sin(initial_bearing),energy=150)
    a.direction=wrap(initial_bearing+math.pi)
    p=add_predator(env)
    env.agent_observations[a.agent_id]=a.observe(predators=[p],edges=list(env.edges))
    min_gap=math.inf
    pursuit_ticks=0
    observed_ticks=0
    positions=[]
    for tick in range(ticks):
        obs=env.agent_observations.get(a.agent_id,[])
        predators=[o for o in obs if o["type"]=="Predator"]
        if predators:
            observed_ticks+=1
            nearest=min(predators,key=lambda o:o["distance"])
            d,q=nearest["distance"],nearest["angle"]
            if mode=="orbit":
                inward=(d-radius)*.8
                vx=inward*math.cos(q)+10*math.cos(q+math.pi/2)
                vy=inward*math.sin(q)+10*math.sin(q+math.pi/2)
            elif mode=="rear":
                predator_heading=q+math.pi-nearest["rel_dir"]
                vx=d*math.cos(q)-radius*math.cos(predator_heading)
                vy=d*math.sin(q)-radius*math.sin(predator_heading)
            else:
                raise ValueError(mode)
            length=math.hypot(vx,vy)
            env.agent_step(a.agent_id,min(10,length),math.atan2(vy,vx),q,False)
        else:
            # Scan from current location if the lure has lost its opponent.
            env.agent_step(a.agent_id,0,0,.2,False)
        seen=p.observe(agents=[a],edges=list(env.edges))
        pursuit_ticks+=any(o["type"]=="Agent" for o in seen)
        env.non_agent_step(.1)
        min_gap=min(min_gap,math.hypot(a.x-p.x,a.y-p.y))
        if tick%10==0:
            positions.append([tick/10,round(p.x,2),round(p.y,2),round(a.x,2),round(a.y,2),p.resting])
        if not env.agents:
            break
    return dict(mode=mode,gap=gap,initial_bearing=initial_bearing,radius=radius,ticks=tick+1,alive=len(env.agents),
                min_gap=min_gap,energy=a.energy,observed_fraction=observed_ticks/(tick+1),
                predator_sensed_agent_fraction=pursuit_ticks/(tick+1),trace=positions)


def run_all():
    results = dict(
        warning="Controlled synthetic worlds and privileged geometry; not full-game scores or deployable policies.",
        source=str(ROOT / "vendor" / "survival-simulator"),
        movement=movement_probe(), sensing=sensing_probe(),
        active_cycles=[active_cycle_probe(False),active_cycle_probe(True)],
        paired_lures=[paired_lure(w,switch_x=s) for w in [40,60,80,120,160,200] for s in [0,5,10]],
        wall_lure=wall_lure(),
        realistic_walls=[wall_lure(width=w,height=h,offset=o) for w in [30,40,50] for h in [30,50,70,100] for o in [0,10,20]],
        wall_lifetime=[wall_lure(ticks=1600,width=30,height=70,max_age=age) for age in [60,90,120]],
        wall_heading_perturbations=[wall_lure(width=30,height=70,offset=o,heading=h)
                                  for o in [-20,0,20] for h in [-.3,.3,math.pi/2,math.pi]],
        wall_width_tolerance=[wall_lure(width=w,height=70,offset=o,heading=h)
                              for w in [31,32,35,37,38] for o in [0,10,20] for h in [-.3,0,.3]],
        orbit_lures=[orbit_lure(gap=g,initial_bearing=b,radius=r,mode=m)
                     for m in ["orbit","rear"] for g in [25,40,60] for b in [0,math.pi/2,math.pi] for r in [25,40]],
    )
    output = ROOT / "results" / "predator-control.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n")
    print(output)
    print(json.dumps({k:v for k,v in results.items() if k in ["movement","active_cycles"]},indent=2))
    for r in results["paired_lures"]+results["realistic_walls"]+results["wall_lifetime"]+results["wall_heading_perturbations"]+results["wall_width_tolerance"]+results["orbit_lures"]:
        print({k:v for k,v in r.items() if k!="trace"})


if __name__ == "__main__":
    run_all()

"""Privileged-coordinate wall holder/forager rotations, original local physics.

Arranged held wall, two 150-energy founders, 1-3 native age-zero trees.
No energy injections, duplicate actions, added fruit, or free replacements.
New predator spawning is disabled; new tree spawning is an explicit ablation.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from predator_control import controlled_env, add_agent, add_predator, wrap
from src.elements.environment import Environment
from src.elements.obstacle import Obstacle
from src.elements.tree import Tree

ROOT = Path(__file__).resolve().parents[1]


def run(seed=1, trees=3, spacing=90, mode="relay_birth", seconds=300,
        new_trees=False, width=32, height=80, founder_max_age=90,
        cap=3, retire_age=65, preserve_food=False, cohort=False):
    env = controlled_env()
    env.rng.seed(seed)
    wall = Obstacle(800-width/2, 600-height/2, width, height)
    env.obstacles = [wall, Obstacle(0,0,1600,30), Obstacle(0,1170,1600,30),
                     Obstacle(0,0,30,1200), Obstacle(1570,0,30,1200)]
    for w in env.obstacles:
        env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
    anchor = (wall.x+wall.width+7, 600)
    founders = [add_agent(env,*anchor,150), add_agent(env,anchor[0]+spacing,600,150)]
    for a in founders:
        a.max_age = founder_max_age
        a.direction = math.pi
    env._next_agent_id = 2
    pred = add_predator(env,wall.x-12,600,heading=0)
    offsets = [0] if trees == 1 else [-35,35] if trees == 2 else [-55,0,55]
    env.trees = [Tree(anchor[0]+spacing,600+y) for y in offsets]
    if new_trees:
        env.spawn_tree = Environment.spawn_tree.__get__(env, Environment)
    env._update_spatial_grid()
    holder = 0
    incoming = None
    retired = set()
    generation = {0:0, 1:0}
    births, handoffs, deaths, trace, meals = [], [], [], [], []
    attention = []
    held = []
    allocated = {}
    last_handoff = -100
    first_lost = None
    total_food = 0
    fruit_count = 0
    last_tree_death = None
    # Instrument removal as passive accounting only. Removed fruits with age
    # <=100 are picked up; rot removes only older fruit in the native method.
    native_remove = env.remove_fruit
    def remove_fruit(fruit):
        nonlocal total_food, fruit_count
        if fruit.age <= 100:
            total_food += fruit.energy
            fruit_count += 1
            eater = next((a.agent_id for a in env.agents
                          if math.dist((a.x,a.y),(fruit.x,fruit.y)) < a.size+fruit.radius), None)
            meals.append(dict(time=round(env.time,1),energy=round(fruit.energy,2),agent=eater))
        native_remove(fruit)
    env.remove_fruit = remove_fruit
    for tick in range(round(seconds*10)):
        if not env.agents:
            break
        alive = env.agents_dict
        if holder not in alive:
            holder = None
        if incoming not in alive:
            incoming = None
        if incoming is not None and math.dist((alive[incoming].x,alive[incoming].y),anchor) < 1:
            old = holder
            holder, incoming = incoming, None
            if old is not None and alive[old].age >= retire_age:
                retired.add(old)
            handoffs.append(dict(time=round(env.time,1),outgoing=old,incoming=holder))
            last_handoff = env.time
        if mode != "static" and incoming is None:
            candidates = [a for a in env.agents if a.agent_id != holder and a.agent_id not in retired]
            need = holder is None or (env.time-last_handoff>8 and
                   (alive[holder].energy<75 or alive[holder].age>=retire_age))
            if candidates and need:
                best = max(candidates, key=lambda a:a.energy-(8 if cohort else 2)*max(0,a.age-55))
                if holder is None or best.energy>alive[holder].energy+25 or (best.age<retire_age-10 and alive[holder].age>=retire_age):
                    incoming = best.agent_id
                    allocated.pop(incoming,None)
        claimed = set()
        before = list(env.agents)
        score_before = env.score
        for a in before:
            aid = a.agent_id
            spawn = False
            target = (a.x,a.y)
            if aid == holder or aid == incoming:
                target = anchor
            elif aid in retired:
                # Move old workers out of the orchard once, then allow natural
                # energy depletion; no deletion or agent-count exemption.
                target = (anchor[0]+250,600+150)
            elif mode != "static":
                candidates = [f for f in env.fruits
                              if f.x>=anchor[0]+15 and abs(f.y-600)<180
                              and f.x<anchor[0]+260 and f.fruit_id not in claimed
                              and (f.energy>=58 or a.energy<65)]
                if preserve_food and a.energy > a.max_energy-65:
                    candidates = []
                previous = allocated.get(aid)
                target_fruit = next((f for f in candidates if f.fruit_id==previous),None)
                if target_fruit is None and candidates:
                    target_fruit = min(candidates,key=lambda f:math.dist((a.x,a.y),(f.x,f.y)))
                if target_fruit is not None:
                    target = (target_fruit.x,target_fruit.y)
                    allocated[aid] = target_fruit.fruit_id
                    claimed.add(target_fruit.fruit_id)
                # Birth while protected and away from the wall. Birth threshold
                # funds both the legal 100-energy cost and continued foraging.
                spawn = mode == "relay_birth" and len(env.agents)<cap and a.age>=40 and a.energy>170
            if cohort and mode == "relay_birth" and aid not in retired and aid != incoming:
                # Retired elders still exist and pay native age/energy costs.
                # Keep a working reserve instead of treating them as replacements.
                active_count = sum(c.agent_id not in retired for c in env.agents)
                spawn = len(env.agents)<6 and active_count<3 and a.age>=25 and a.energy>170
            dx,dy = target[0]-a.x,target[1]-a.y
            distance = min(a.speed,a.sprint_speed,math.hypot(dx,dy))
            direction = wrap(math.atan2(dy,dx)-a.direction) if distance else 0
            prior_ids = set(env.agents_dict)
            env.agent_step(aid,distance,direction,0,spawn)
            if spawn:
                for child in env.agents:
                    if child.agent_id not in prior_ids:
                        generation[child.agent_id] = generation[aid]+1
                        births.append(dict(time=round(env.time,1),parent=aid,child=child.agent_id,
                                           generation=generation[child.agent_id],energy_cost=100,
                                           child_energy=child.energy))
        # Predators can only choose among entities they can actually sense.
        seen = [o for o in pred.observe(agents=list(env.agents),edges=list(env.edges)) if o['type']=='Agent']
        attention.append(bool(seen and min(seen,key=lambda o:o['distance'])['id'] in (holder,incoming)))
        env.non_agent_step(.1)
        for old in before:
            if old.agent_id not in env.agents_dict:
                deaths.append(dict(id=old.agent_id,time=round(env.time,1),age=round(old.age,1),
                                   cause='depleted' if old.energy<=0 else 'capture',
                                   generation=generation[old.agent_id]))
        holds = wall.x-45<pred.x<wall.x and abs(pred.y-600)<height/2+10
        held.append(holds)
        if not holds and first_lost is None:
            first_lost = round(env.time,1)
        if not env.trees and last_tree_death is None:
            last_tree_death = round(env.time,1)
        if tick%50 == 0:
            trace.append(dict(time=round(env.time,1),holder=holder,incoming=incoming,
                              agents=[dict(id=a.agent_id,energy=round(a.energy,1),age=round(a.age,1),
                                           x=round(a.x,1),y=round(a.y,1),retired=a.agent_id in retired)
                                      for a in env.agents],trees=len(env.trees),fruits=len(env.fruits),
                              predator=[round(pred.x,1),round(pred.y,1)],held=holds))
    return dict(seed=seed,initial_trees=trees,spacing=spacing,mode=mode,new_tree_spawning=new_trees,
                width=width,height=height,founder_max_age=founder_max_age,
                cap=cap,retire_age=retire_age,preserve_food=preserve_food,cohort=cohort,
                horizon=seconds,seconds=round(env.time,1),alive=len(env.agents),
                generations=max(generation.values()),births=births,handoffs=handoffs,deaths=deaths,
                captures=sum(d['cause']=='capture' for d in deaths),food_count=fruit_count,
                food_energy=round(total_food,2),meals=meals,tree_patch_empty_at=last_tree_death,
                hold_fraction=sum(held)/len(held),attention_fraction=sum(attention)/len(attention),
                first_hold_loss=first_lost,trace=trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='wall-orchard.json')
    parser.add_argument('--suite',choices=['initial','refinement','cohort','heldout'],default='initial')
    args=parser.parse_args()
    cases = [dict(seed=seed,trees=trees,mode='relay_birth') for seed in (1,2) for trees in (1,2,3)]
    cases += [dict(seed=1,trees=3,mode=mode) for mode in ('static','relay')]
    cases += [dict(seed=seed,trees=3,mode='relay_birth',new_trees=True) for seed in (1,2)]
    if args.suite == 'refinement':
        cases = [dict(seed=1,trees=3,new_trees=True,cap=cap,retire_age=age,preserve_food=True)
                 for cap in (3,4,5) for age in (65,85)]
    if args.suite == 'heldout':
        cases = [dict(seed=seed,trees=trees,spacing=spacing,new_trees=new_trees,
                      width=width,height=height,cap=6,retire_age=65,preserve_food=True,cohort=True)
                 for seed,trees,spacing,width,height,new_trees in
                 [(3,2,60,30,70,True),(4,3,120,35,100,True),
                  (3,2,60,30,70,False),(4,3,120,35,100,False)]]
    if args.suite == 'cohort':
        cases = [dict(seed=seed,trees=3,new_trees=new_trees,cap=6,retire_age=65,
                      preserve_food=True,cohort=True)
                 for seed in (1,2) for new_trees in (False,True)]
    results=[]
    start=time.perf_counter()
    for case in cases:
        row=run(**case)
        results.append(row)
        print({k:row[k] for k in ('seed','initial_trees','mode','new_tree_spawning','seconds','alive',
                                 'generations','captures','food_count','hold_fraction','first_hold_loss')},
              'births',len(row['births']),'handoffs',len(row['handoffs']),flush=True)
        (ROOT/'results'/args.output).write_text(json.dumps(dict(scope=__doc__,runs=results),indent=2,
                                                         default=lambda v:v.item())+'\n')
    print('runtime',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    main()

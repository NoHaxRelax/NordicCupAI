"""Independent local source-to-claim checks; no vendor edits or network access."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor/survival-simulator'))
from src.elements.environment import Environment
from src.elements.biome import Forest_biome
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment, create_environment


def fixture(seed=11):
    env = Environment(400, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    a = env.spawn_agent(x=100, y=200)
    a.direction = 0
    return env, a


def act(a, move=0, direction=0, turn=0, spawn=False):
    return (a.agent_id, ActionRequest(agent_id=a.agent_id, move_distance=move,
            move_direction=direction, turn_angle=turn, spawn_agent=spawn))


def run():
    out = {}
    rows = []
    for n in (1, 10):
        e, a = fixture()
        step_environment(e, [act(a, 10)] * n)
        rows.append(dict(actions=n, x=float(a.x), energy=a.energy, age=a.age))
    assert rows[0]['x'] == 110 and rows[1]['x'] == 200
    assert rows[0]['age'] == rows[1]['age'] == .1
    out['known_duplicate_actions'] = rows

    rows=[]
    for child_first in (True,False):
        e,a=fixture()
        child_action=(e._next_agent_id,ActionRequest(agent_id=e._next_agent_id,
            move_distance=0,move_direction=0,turn_angle=1,spawn_agent=False))
        parent_action=act(a,spawn=True)
        step_environment(e,[child_action,parent_action] if child_first else [parent_action,child_action])
        child=e.agents[-1]
        rows.append(dict(child_action_first=child_first,energy=child.energy,heading=child.direction))
    assert math.isclose(rows[0]['energy']-rows[1]['energy'],1/(2*math.pi))
    assert math.isclose(rows[1]['heading']-rows[0]['heading'],1)
    out['known_unobserved_newborn_action_requires_prior_birth'] = rows

    e, a = fixture()
    step_environment(e, [act(a, 10, turn=math.pi/2)])
    assert a.x == 110 and a.y == 200 and a.direction == math.pi/2
    out['move_before_turn'] = dict(x=float(a.x), y=float(a.y), heading=a.direction)

    rows = []
    for energy, move, turn in ((100.5, 0, 0), (100.5, 10, 0),
                               (100.5, 0, math.pi), (100.7, 10, 0)):
        e, a = fixture()
        a.energy = energy
        step_environment(e, [act(a, move, turn=turn, spawn=True)])
        rows.append(dict(initial_energy=energy, move=move, turn=turn,
                         population=len(e.agents), final_energy=a.energy))
    assert [r['population'] for r in rows] == [2, 1, 1, 2]
    out['reproduction_after_movement_and_turn_cost'] = rows

    rows = []
    for width, sprint in ((30, 40), (30.001, 40), (30, 20)):
        e, a = fixture()
        a.x = 195
        a.speed = 20 if sprint == 40 else 10
        a.sprint_speed = sprint
        e.spawn_obstacle(x=200, y=100, width=width, height=200)
        e._update_agent_grid()
        step_environment(e, [act(a, 40)])
        rows.append(dict(width=width, sprint=sprint, x=float(a.x), y=float(a.y),
                         crossed=bool(a.x >= 200+width+5), energy=a.energy))
    assert [r['crossed'] for r in rows] == [True, False, False]
    out['wall_tunneling_fragility'] = rows

    rows = []
    for energy in (150, 99):
        e, a = fixture()
        a.x = 35
        a.speed, a.sprint_speed, a.energy = 20, 40, energy
        e._update_agent_grid()
        step_environment(e, [act(a, 40, math.pi)])
        inside = e._in_obstacle((a.x, a.y), a.size, e.obstacles)
        rows.append(dict(initial_energy=energy, x=float(a.x), y=float(a.y), inside=inside))
    assert rows[0]['inside'] and not rows[1]['inside']
    out['boundary_clamp_requires_available_sprint'] = rows

    rows = []
    for reverse in (False, True):
        e, a = fixture()
        b = e.spawn_agent(x=100, y=200)
        b.direction = 0
        f = e.spawn_fruit(x=100, y=200)
        actions = [act(a), act(b)]
        step_environment(e, list(reversed(actions)) if reverse else actions)
        obs = [e.get_agent_state(z.agent_id)['observations'] for z in (a,b)]
        rows.append(dict(reverse_actions=reverse, energies=[a.energy,b.energy],
                         observed_fruits=[sum(o['type']=='Fruit' for o in row) for row in obs],
                         fruit_present=f in e.fruits))
    assert rows[0]['energies'] == rows[1]['energies'] == [169.9,149.9]
    assert rows[0]['observed_fruits'] == rows[1]['observed_fruits'] == [1,0]
    out['fruit_claim_is_birth_order_not_request_order'] = rows

    rows = []
    for resting in (False, True):
        e, a = fixture()
        p = e.spawn_predator(x=140,y=200)
        p.direction, p.energy, p.resting = math.pi, 99, resting
        step_environment(e,[act(a)])
        seen = next(o for o in e.get_agent_state(a.agent_id)['observations'] if o['type']=='Predator')
        rows.append(dict(resting=resting, observed=seen['distance'],
                         actual=math.hypot(p.x-a.x,p.y-a.y)))
    assert rows[0]['observed'] == 40 and rows[0]['actual'] == 25
    assert rows[1]['observed'] == rows[1]['actual'] == 40
    out['predator_lag_is_motion_dependent'] = rows

    e, a = fixture()
    p = e.spawn_predator(x=140,y=200)
    p.direction,p.energy,p.resting = math.pi,100,False
    step_environment(e,[act(a)])
    e.kill_agent(a)  # diagnostic: cache retention does not mean ghost in returned state
    out['dead_cache_not_returned'] = dict(cache_retained=a.agent_id in e.agent_observations,
                                         returned=e.get_agent_state(a.agent_id))
    assert out['dead_cache_not_returned'] == dict(cache_retained=True,returned=None)

    # Deterministic geometry fixture: exact equidistance can rely on set iteration.
    e, a = fixture()
    b = e.spawn_agent(x=200,y=200)
    p = e.spawn_predator(x=150,y=200)
    a.direction=b.direction=0
    observed = p.observe(agents=list(e._get_local_agents(p)), edges=e._get_local_edges(p))
    tied = [o['id'] for o in observed if o['type']=='Agent' and o['distance']==50]
    assert set(tied)=={a.agent_id,b.agent_id}
    out['equal_target_distance_order_not_promised'] = dict(observation_ids=tied,
        selected_by_min=min([o for o in observed if o['type']=='Agent'],key=lambda o:o['distance'])['id'],
        warning='Order originates from a set of object identities; do not infer stable ID priority.')
    rows=[]
    for seed in (11,42):
        e=create_environment(1600,1200,400,5,0,32,50,random.Random(seed))
        widths=[min(o.width,o.height) for o in e.obstacles[4:]]
        candidates=[]
        for a in e.agents:
            modifier=e.biome_map[int(a.x),int(a.y)].move_penalty
            endpoint=(a.x+10*modifier*math.cos(a.direction),
                      a.y+10*modifier*math.sin(a.direction))
            if not e._in_obstacle(endpoint,a.size,e.obstacles):
                candidates.append((a,endpoint))
        assert candidates
        a,endpoint=candidates[0]
        start_heading=a.direction
        step_environment(e,[act(a,10,turn=math.pi/2)])
        error=math.hypot(a.x-endpoint[0],a.y-endpoint[1])
        assert error<1e-8 and math.isclose(a.direction-start_heading,math.pi/2)
        assert min(widths)>30
        rows.append(dict(seed=seed,obstacle_count=len(widths),minimum_thickness=min(widths),
                         widths_at_most_30=sum(w<=30 for w in widths),
                         movement_before_turn_endpoint_error=error,energy=a.energy))
    out['unmodified_generated_maps'] = rows
    return out


if __name__ == '__main__':
    result = dict(source_commit='acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
                  scope='Isolated fixtures; original engine methods; forest override; standard founder energy unless labelled; normal spawning retained.',
                  results=run())
    path=ROOT/'results/mechanics_hunt/11_astra_evidence.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

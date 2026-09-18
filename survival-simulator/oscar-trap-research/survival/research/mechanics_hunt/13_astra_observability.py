"""Observability audit; original physics, no network, diagnostic state separated."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor' / 'survival-simulator'))
from src.core import SimulationCore
from src.elements.environment import Environment
from src.elements.biome import Forest_biome
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


def action(aid, distance=0, turn=0, spawn=False):
    return (aid, ActionRequest(agent_id=aid, move_distance=distance,
                              move_direction=0, turn_angle=turn, spawn_agent=spawn))


def fixture():
    e = Environment(400, 400, 400, random.Random(1313))
    e.biome_map.fill(Forest_biome())
    a = e.spawn_agent(x=180, y=200)
    a.direction = 0
    return e, a


def payload(e, a):
    agents, fruits, trees, obstacles, predators, edges = e._get_local_objects(a)
    e.agent_observations[a.agent_id] = a.observe(agents, fruits, trees, obstacles, edges, predators)
    return e.get_agent_state(a.agent_id)


def ambiguous_states():
    fruits = []
    for growth in [0, 40]:
        e, a = fixture()
        f = e.spawn_fruit(x=200, y=200)
        f.grow(growth)
        before = payload(e, a)
        out = step_environment(e, [action(a.agent_id, distance=20)])
        fruits.append({'growth_fixture': growth, 'before': before,
                       'energy_after': a.energy, 'score': out['score']})
    assert fruits[0]['before'] == fruits[1]['before']
    assert math.isclose(fruits[1]['energy_after']-fruits[0]['energy_after'], 40)
    predators = []
    for energy in [0, 102]:
        e, a = fixture()
        p = e.spawn_predator(x=200, y=200)
        p.direction = math.pi
        p.energy, p.resting = energy, True
        before = payload(e, a)
        out = step_environment(e, [action(a.agent_id)])
        predators.append({'hidden_predator_energy': energy, 'before': before,
                           'alive_after': out['num_agents'], 'score': out['score']})
    assert predators[0]['before'] == predators[1]['before']
    assert [p['alive_after'] for p in predators] == [1, 0]
    aging = []
    for threshold in [60, 120]:
        e, a = fixture()
        a.age, a.max_age = 70, threshold
        before = payload(e, a)
        step_environment(e, [action(a.agent_id)])
        aging.append({'hidden_max_age': threshold, 'before': before, 'energy_after': a.energy})
    assert aging[0]['before'] == aging[1]['before']
    assert math.isclose(aging[1]['energy_after']-aging[0]['energy_after'], .701)
    return {'fruit_age': fruits, 'predator_sleep': predators, 'aging_threshold': aging}


def reproduction_controls():
    cases = []
    for energy, distance, turn in [(100.5, 0, 0), (100.5, 10, 0), (101, 10, math.pi), (101.01, 10, math.pi)]:
        e, a = fixture()
        a.energy = energy
        s = payload(e, a)
        # All input fields here are ordinary policy data.
        move = min(distance, s['sprint_speed'])
        if s['energy'] < s['max_energy']/5 and move > s['speed']:
            move = s['speed']
        cost = min(move, s['speed'])*.05 + max(0, move-s['speed'])*.5
        predicted = s['energy']-cost-min(math.pi, abs(turn))/(2*math.pi)>100
        step_environment(e, [action(a.agent_id, distance, turn, True)])
        born = e._next_agent_id == 2
        assert born == predicted
        cases.append({'energy':energy, 'move':distance, 'turn':turn,
                      'predicted_birth':predicted, 'actual_birth':born,
                      'parent_survives':a.agent_id in e.agents_dict,
                      'population_after':len(e.agents)})
    return cases


def detect_aging(previous, current):
    """Uses only two ordinary payloads after an idle action, with no true state."""
    if not math.isclose(current['age']-previous['age'], .1, abs_tol=1e-7):
        return 'skipped_or_unknown'
    loss = previous['energy']-current['energy']
    if math.isclose(loss, .1+.01*current['age'], abs_tol=1e-7):
        return 'old'
    if math.isclose(loss, .1, abs_tol=1e-7):
        return 'young'
    return 'confounded'


def generated_maps():
    results = []
    for seed in [13, 29, 47]:
        sim = SimulationCore(seed=seed)
        state = sim.step([])
        counts = {'old':0, 'young':0, 'confounded':0, 'skipped_or_unknown':0}
        errors = []
        first = {}
        checks = 0
        while state['num_agents'] and state['sim_time'] < 110:
            previous = {s['agent_id']:s for s in state['observations']}
            state = sim.step([action(aid) for aid in previous])
            for current in state['observations']:
                aid = current['agent_id']
                classification = detect_aging(previous[aid], current)
                counts[classification] += 1
                # Diagnostic ground truth only, never enters the detector/actions.
                agent = sim.env.agents_dict[aid]
                truth = agent.age > agent.max_age
                if classification in ['young', 'old']:
                    checks += 1
                    if (classification == 'old') != truth:
                        errors.append({'id':aid, 'time':state['sim_time'], 'classification':classification})
                if classification == 'old' and aid not in first:
                    first[aid] = {'observed_onset_age':current['age'], 'hidden_max_age_for_validation':agent.max_age}
        assert not errors, errors
        results.append({'seed':seed, 'seconds':state['sim_time'], 'score':state['score'],
                        'counts':counts, 'checked':checks, 'errors':errors, 'first_onsets':first,
                        'population_remaining':state['num_agents']})
        print(json.dumps(results[-1]), flush=True)
    assert sum(r['counts']['old'] for r in results)>0
    return results


def detector_controls():
    cases = []
    for energy in [150, 500]:
        e, a = fixture()
        a.age, a.max_age, a.energy = 70, 60, energy
        e.spawn_fruit(x=a.x, y=a.y)
        before = payload(e, a)
        step_environment(e, [action(a.agent_id)])
        classification = detect_aging(before, e.get_agent_state(a.agent_id))
        assert classification == 'confounded'
        cases.append({'starting_energy':energy, 'hidden_old':True,
                      'fruit_collected':True, 'classification':classification})
    return cases


def main():
    vendor = ROOT/'vendor'/'survival-simulator'
    hashes = {str(p.relative_to(vendor)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(vendor.rglob('*.py'))}
    result = {'source_commit':'acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
              'source_hashes':hashes,
              'scope':'Fixtures set position/terrain/age/hidden state. Generated runs use complete default maps and energy, one idle action per observed living agent, unmodified physics and spawning.',
              'ambiguous_states':ambiguous_states(),
              'reproduction_controls':reproduction_controls(),
              'detector_controls':detector_controls(),
              'generated_maps':generated_maps()}
    assert hashes == {str(p.relative_to(vendor)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(vendor.rglob('*.py'))}
    out = ROOT/'results'/'mechanics_hunt'/'13_astra_observability.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()

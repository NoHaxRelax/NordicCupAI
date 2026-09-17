"""Check models.trapper.predator_model against the engine, tick by tick, on generated maps.

Wraps Predator.step and Environment.update_entity_position / update_entity_direction
(read-only: the wrappers call the originals) and compares the model's decision and
resulting position/heading/energy with the engine's. Also checks target selection
and site detection counts.

Run from survival-simulator:
    .venv/bin/python scripts/trapper/validate_predator_model.py --seeds 1 2 3 --ticks 1500
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import SimulationCore                      # noqa: E402
from src.elements.predator import Predator               # noqa: E402
from src.utils.DTOs import ActionRequest                 # noqa: E402
from models.trapper import predator_model as pm          # noqa: E402
from models.trapper.oracle import OracleWorld            # noqa: E402
from models.trapper.sites import find_sites              # noqa: E402


def flee_policy(states, rng):
    """Keep agents alive a while so predators have something to chase."""
    out = []
    for s in states:
        preds = [o for o in s['observations'] if o['type'] == 'Predator']
        fruits = [o for o in s['observations'] if o['type'] == 'Fruit']
        if preds:
            p = min(preds, key=lambda o: o['distance'])
            d = p['angle'] + math.pi
            out.append((s['agent_id'], ActionRequest(agent_id=s['agent_id'], move_distance=s['sprint_speed'] if p['distance'] < 80 else s['speed'],
                                                     move_direction=d, turn_angle=max(-.3, min(.3, p['angle'])), spawn_agent=False)))
        elif fruits:
            f = min(fruits, key=lambda o: o['distance'])
            out.append((s['agent_id'], ActionRequest(agent_id=s['agent_id'], move_distance=min(s['speed'], f['distance']),
                                                     move_direction=f['angle'], turn_angle=max(-.3, min(.3, f['angle'])), spawn_agent=s['energy'] > 220)))
        else:
            out.append((s['agent_id'], ActionRequest(agent_id=s['agent_id'], move_distance=s['speed'], move_direction=0.,
                                                     turn_angle=rng.uniform(-.2, .2), spawn_agent=s['energy'] > 220)))
    return out


def validate(seed, ticks, starting_predators):
    sim = SimulationCore(seed=seed, starting_predators=starting_predators)
    env = sim.env
    oracle = OracleWorld(env)
    stats = dict(steps=0, decision_mismatch=0, move_mismatch=0, target_checks=0, target_mismatch=0, modes={}, kills_pred=0, kills_seen=0)
    world = oracle.update()
    rects = world.rects

    orig_step = Predator.step
    orig_move = env.update_entity_position
    orig_turn = env.update_entity_direction
    pending = {}

    def wrapped_step(self, observation=None):
        # engine decision
        state = pm.PredState(self.x, self.y, self.direction, self.energy, self.resting, self.speed, self.sprint_speed,
                             self.size, self.max_energy, self.hearing_radius, self.vision_radius, self.cone_angle)
        rng_state = self.rng.getstate()
        mine, mode = pm.decide(state, observation, self.rng)
        self.rng.setstate(rng_state)
        theirs = orig_step(self, observation)
        stats['steps'] += 1
        stats['modes'][mode] = stats['modes'].get(mode, 0) + 1
        same = set(mine) == set(theirs) and all(
            (mine[k] is None and theirs[k] is None) or (mine[k] is not None and theirs[k] is not None and abs(float(mine[k]) - float(theirs[k])) < 1e-9)
            for k in mine)
        if not same:
            stats['decision_mismatch'] += 1
            if stats['decision_mismatch'] <= 3:
                print('DECISION MISMATCH', mode, mine, theirs)
        # independent observation model: does the model observe the same agents?
        agents = [(a.agent_id, a.x, a.y, a.direction) for a in env.agents]
        my_obs = pm.observed_agents(state, agents, rects)
        their_obs = sorted((o for o in observation if o.get('type') == 'Agent'), key=lambda o: o['distance'])
        stats['target_checks'] += 1
        if [o['id'] for o in their_obs] != my_obs:
            stats['target_mismatch'] += 1
            if stats['target_mismatch'] <= 3:
                print('OBSERVATION MISMATCH', [o['id'] for o in their_obs], my_obs)
        pending[id(self)] = (state, theirs, mode)
        return theirs

    def wrapped_move(entity, distance, direction=None, local_obstacles=None):
        key = id(entity)
        if isinstance(entity, Predator) and key in pending:
            state, signals, mode = pending[key]
            lrects = pm.local_rects(rects, state.x, state.y)
            predicted = pm.move(state, distance, direction, world.biome_at((state.x, state.y)), lrects, env.width, env.height)
            orig_move(entity, distance, direction, local_obstacles)
            err = math.hypot(predicted.x - entity.x, predicted.y - entity.y) + abs(predicted.energy - entity.energy)
            if err > 1e-6:
                stats['move_mismatch'] += 1
                if stats['move_mismatch'] <= 3:
                    print('MOVE MISMATCH', mode, (predicted.x, predicted.y, predicted.energy), (entity.x, entity.y, entity.energy))
            pending[key] = (predicted, signals, mode)
            return
        return orig_move(entity, distance, direction, local_obstacles)

    def wrapped_turn(entity, angle):
        key = id(entity)
        if isinstance(entity, Predator) and key in pending:
            state, signals, mode = pending.pop(key)
            predicted = pm.turn(state, angle)
            orig_turn(entity, angle)
            if abs(predicted.heading - entity.direction) > 1e-9 or abs(predicted.energy - entity.energy) > 1e-9:
                stats['move_mismatch'] += 1
            return
        return orig_turn(entity, angle)

    Predator.step = wrapped_step
    env.update_entity_position = wrapped_move
    env.update_entity_direction = wrapped_turn
    rng = random.Random(seed)
    state = sim.step([])
    try:
        for _ in range(ticks):
            if not state['num_agents']:
                break
            state = sim.step(flee_policy(state['observations'], rng))
            oracle.update()
    finally:
        Predator.step = orig_step
    sites = find_sites(rects, env.width, env.height)
    stats['sites'] = dict(wall=sum(s.kind == 'wall' for s in sites), gap=sum(s.kind == 'gap' for s in sites))
    stats['time'] = round(state['sim_time'], 1)
    stats['predators'] = len(env.predators)
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--ticks', type=int, default=1500)
    ap.add_argument('--predators', type=int, default=4)
    a = ap.parse_args()
    ok = True
    for seed in a.seeds:
        st = validate(seed, a.ticks, a.predators)
        print(f'seed {seed}:', st)
        ok &= st['decision_mismatch'] == 0 and st['move_mismatch'] == 0 and st['target_mismatch'] == 0
    print('PASS' if ok else 'FAIL')

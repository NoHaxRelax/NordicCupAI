"""Unmodified Python game -> public DTOs and executed actions for V4 replay.

The supplied seed is evaluator-only. This is a replay-parity fixture, not seed
recovery. No hash monkeypatches, hidden state serialization, or RNG interception.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
import numpy

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--ticks', type=int, default=1000)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
core = SimulationCore(seed=a.seed)
state = core.step([])  # Same initial empty step as the public API and V4 replay.
rr = random.Random(2026092009)
a.output.parent.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
frames = 0
with a.output.open('w') as f:
    for tick in range(a.ticks):
        agents = state['observations']
        if not agents:
            break
        body = dict(game_status='ok', score=state['score'], sim_time=state['sim_time'],
                    n_agents=state['num_agents'], agent_status=agents)
        actions = [dict(agent_id=agent['agent_id'], move_distance=rr.choice([0., 5., 15.]),
                        move_direction=rr.uniform(-3.14, 3.14), turn_angle=rr.uniform(-.5, .5),
                        spawn_agent=agent['energy'] > 170 and tick % 20 == 0)
                   for agent in agents]
        f.write(json.dumps(dict(seq=tick, before=body, actions=actions)) + '\n')
        state = core.step([(act['agent_id'], ActionRequest(**act)) for act in actions])
        frames += 1
result = dict(scope='Known-seed Python public replay fixture, not blind recovery', seed=a.seed,
              frames=frames, python=platform.python_version(), numpy=numpy.__version__,
              seconds=time.perf_counter()-start, sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
a.output.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))

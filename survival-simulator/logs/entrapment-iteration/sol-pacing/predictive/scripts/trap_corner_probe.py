"""Native geometry probe: an ordinary inside corner versus a narrow channel.

Fixture bait and predators stay energized; no guide or colony policy is tested.
Approaches begin within hearing range. All ticks are retained for inspection.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elements.environment import Environment
from src.elements.obstacle import Obstacle
from src.elements.agent import Agent
from src.elements.predator import Predator


def run(folder, trials):
    folder.mkdir(parents=True, exist_ok=False)
    layouts = {
        'inside_corner': ([(100, 100, 100, 200), (200, 100, 200, 100)], (205.05, 205.05)),
        'channel_15': ([(100, 100, 100, 200), (215, 100, 100, 200)], (207.5, 105.)),
    }
    results = []
    for name, (rects, goal) in layouts.items():
        env = Environment(600, 400, 400, random.Random(1))
        env.obstacles.extend(Obstacle(*r) for r in rects)
        env.edges = [edge for obstacle in env.obstacles for edge in obstacle.edges]
        env.spawn_predator = lambda *args, **kwargs: None
        env._update_spatial_grid()
        rng = random.Random(9182026)
        for trial in range(trials):
            bait = Agent(*goal, rng=env.rng)
            bait.agent_id = 0
            for _ in range(10000):
                angle = rng.uniform(-math.pi, math.pi)
                distance = rng.uniform(50, 59)
                p = (goal[0]+distance*math.cos(angle), goal[1]+distance*math.sin(angle))
                if not env._in_obstacle(p, 10.01, env.obstacles):
                    break
            else:
                raise RuntimeError('No free hearing-range approach')
            predator = Predator(*p, rng=env.rng)
            predator.direction = math.atan2(goal[1]-p[1], goal[0]-p[0])
            env.agents = [bait]
            env.agents_dict = {0: bait}
            env.predators = [predator]
            env.time = 0.
            env.agent_observations.clear()
            env._update_spatial_grid()
            with gzip.open(folder/f'{name}-{trial}.jsonl.gz', 'wt') as trace:
                for tick in range(301):
                    alive = bait in env.agents
                    trace.write(json.dumps(dict(tick=tick, bait_alive=alive,
                        bait=goal, predator=[predator.x, predator.y, predator.direction]))+'\n')
                    if not alive or tick == 300:
                        break
                    bait.energy = bait.max_energy
                    bait.age = 0.
                    predator.energy = predator.max_energy
                    predator.resting = False
                    env.non_agent_step(.1)
            results.append(dict(layout=name, trial=trial, caught=not alive,
                                ticks=tick, initial_predator=p))
    summary = dict(assumptions='Native steps; full-energy stationary bait and predator; 30-second geometry probe',
                   layouts=layouts, cases=results)
    (folder/'results.json').write_text(json.dumps(summary, indent=2))
    for name in layouts:
        rows = [r for r in results if r['layout'] == name]
        print(name, sum(r['caught'] for r in rows), '/', len(rows), 'bait caught', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=16)
    args = parser.parse_args()
    run(args.output, args.trials)

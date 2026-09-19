"""Exact event accounting and unchanged native dynamics under telemetry."""
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import numpy as np
from src.elements.environment import Environment
from src.elements.agent import Agent
from src.elements.fruit import Fruit
from src.elements.predator import Predator
from scripts.research_diagnostics import Diagnostics
from scripts.research_support import research_case


def environment():
    env = Environment.__new__(Environment)
    env.width, env.height, env.chunk_size = 200, 200, 400
    env.rng = random.Random(17)
    env.time, env.score = 0., 0.
    env.agents, env.fruits, env.trees, env.predators, env.obstacles, env.edges = [], [], [], [], [], []
    env.edges = [((0, 0), (200, 0)), ((200, 0), (200, 200)), ((200, 200), (0, 200)), ((0, 200), (0, 0))]
    env.agents_dict, env.fruits_dict, env.agent_observations = {}, {}, {}
    env._next_agent_id, env._next_fruit_id = 0, 0
    biome = SimpleNamespace(type='grassland', energy_drain_rate=1., move_penalty=1., tree_spawn_rate=0., fruit_spawn_rate=0.)
    env.biome_map = np.full((200, 200), biome)
    env._update_spatial_grid()
    return env


def agent(env, x=50, y=50, energy=490):
    a = Agent(x, y, energy=energy, rng=env.rng, max_age=100)
    a.agent_id = env._next_agent_id
    env._next_agent_id += 1
    env.agents.append(a)
    env.agents_dict[a.agent_id] = a
    env._update_agent_grid()
    return a


def fruit(env, x=50, y=50, energy=60, age=0):
    f = Fruit(x, y)
    f.energy, f.age = energy, age
    f.fruit_id = env._next_fruit_id
    env._next_fruit_id += 1
    env.fruits.append(f)
    env.fruits_dict[f.fruit_id] = f
    env._update_fruit_grid()
    return f


def record(env, folder, **overrides):
    return Diagnostics(env, folder, dict(case_id='test', seed=17, policy_seed=0,
                       diagnostics=dict(screenshots=False, **overrides)), .1)


class DiagnosticsTests(unittest.TestCase):
    def test_absorption_rot_and_multiple_meals_accounted_separately(self):
        env = environment()
        a = agent(env)
        fruit(env)
        fruit(env, energy=20)
        fruit(env, x=170, y=170, age=101)
        with tempfile.TemporaryDirectory() as folder:
            d = record(env, folder)
            d.before(0, [])
            d.decision([], {'policy_debug': {'roles': {str(a.agent_id): 'gatherer'}}})
            env.non_agent_step(.1)
            d.after()
            result = d.finish('horizon')
            self.assertAlmostEqual(result['totals']['fruit_absorbed_energy'], 10.1)
            self.assertEqual(result['totals']['fruit_gross_energy'], 80)
            self.assertAlmostEqual(result['totals']['fruit_cap_waste'], 69.9)
            self.assertEqual(result['totals']['fruits_rotted'], 1)
            self.assertEqual(result['mean_gross_energy_per_fruit'], 40)
            self.assertAlmostEqual(result['mean_absorbed_energy_per_fruit'], 5.05)

    def test_starvation_and_predation_are_native_call_site_causes(self):
        for predation in (False, True):
            env = environment()
            a = agent(env, energy=100 if predation else .05)
            if predation:
                p = Predator(a.x, a.y, rng=env.rng, speed=0, sprint_speed=0)
                p.resting, p.energy = False, 100
                env.predators.append(p)
                env._update_predator_grid()
            with tempfile.TemporaryDirectory() as folder:
                d = record(env, folder)
                d.before(0, [])
                d.decision([], {})
                env.non_agent_step(.1)
                d.after()
                result = d.finish('extinct')
                self.assertEqual(result['deaths'], {'predation' if predation else 'energy_depletion': 1})

    def test_recording_preserves_fixed_action_engine_state_and_rng(self):
        outcomes = []
        for traced in (False, True):
            env = environment()
            a = agent(env, energy=300)
            fruit(env, x=60, energy=40)
            with tempfile.TemporaryDirectory() as folder:
                d = record(env, folder) if traced else None
                for tick in range(20):
                    if d:
                        d.before(tick, [])
                        d.decision([], {})
                    env.agent_step(a.agent_id, 1., 0., .02, spawn_agent=tick == 0)
                    env.non_agent_step(.1)
                    if d:
                        d.after()
                outcomes.append(dict(time=env.time, score=env.score, rng=env.rng.getstate(),
                    agents=[(x.agent_id, x.x, x.y, x.energy, x.age) for x in env.agents],
                    fruits=[(x.fruit_id, x.energy, x.age) for x in env.fruits]))
                if d:
                    result = d.finish('horizon')
                    self.assertAlmostEqual(result['totals']['reproduction_energy'], 100)
                    self.assertEqual(result['totals']['births'], 1)
                    self.assertGreater(result['totals']['movement_energy'], 0)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_durable_checkpoint_bounded_tail_and_error_frame(self):
        env = environment()
        agent(env)
        with tempfile.TemporaryDirectory() as folder:
            d = record(env, folder, tail_seconds=1., post_event_seconds=.1, checkpoint_seconds=.2)
            for tick in range(30):
                d.before(tick, [])
                d.decision([], {})
                env.non_agent_step(.1)
                d.after()
            self.assertTrue((Path(folder)/'checkpoint.json').exists())
            d.before(30, [{'agent_id': 0}])
            result = d.finish('error')
            rows = [json.loads(x) for x in gzip.decompress((Path(folder)/'terminal.jsonl.gz').read_bytes()).splitlines()]
            self.assertLessEqual(len(rows), 12)
            self.assertTrue(rows[-1]['incomplete_step'])
            self.assertFalse(result['complete'])

    def test_telemetry_configuration_changes_cache_identity(self):
        a = research_case('s', {}, {}, False, 1, 0)
        b = research_case('s', {}, {}, False, 1, 0, {'enabled': False})
        self.assertNotEqual(a['case_id'], b['case_id'])


if __name__ == '__main__':
    unittest.main()

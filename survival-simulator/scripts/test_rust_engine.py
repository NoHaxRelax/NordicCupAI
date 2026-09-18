"""Exact differential tests against the retained Python engine.

Run from the repository root:
  .venv/bin/python -m unittest discover -s survival-simulator/scripts -p test_rust_engine.py -v

The oracle and native engine run on the SAME object identities. Python's sets of
entities use address-based hashes; independently constructed seeded worlds do
not guarantee identical observation or interaction order even in Python alone.
Constructor replay removes that confounder without sorting/changing the game.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import random
import struct
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.backend import native, resolve_backend
from src.core import SimulationCore
from src.elements import environment as environment_module
from src.elements.agent import Agent
from src.elements.fruit import Fruit
from src.elements.predator import Predator
from src.elements.tree import Tree
from src.utils.DTOs import ActionRequest
from src.utils.sensing import compute_visibility

COLLECTIONS = ("agents", "fruits", "trees", "predators", "obstacles")


class Snapshot:
    def __init__(self, env):
        self.env = env
        self.attrs = env.__dict__.copy()
        for key in (*COLLECTIONS, "agents_dict", "fruits_dict", "agent_observations"):
            self.attrs[key] = self.attrs[key].copy()
        self.objects = [(obj, obj.__dict__.copy()) for key in COLLECTIONS
                        for obj in getattr(env, key)]
        self.rng = env.rng.getstate()

    def restore(self, backend):
        self.env.__dict__.clear()
        self.env.__dict__.update(self.attrs)
        for key in (*COLLECTIONS, "agents_dict", "fruits_dict", "agent_observations"):
            self.env.__dict__[key] = self.attrs[key].copy()
        for obj, attrs in self.objects:
            obj.__dict__.clear()
            obj.__dict__.update(attrs)
        self.env.rng.setstate(self.rng)
        self.env._backend = backend


class ConstructorReplay:
    def __init__(self):
        self.recorded = []
        self.index = 0
        self.replaying = False

    def wrap(self, cls):
        def create(*args, **kwargs):
            if self.replaying:
                expected, obj = self.recorded[self.index]
                self.index += 1
                if expected is not cls:
                    raise AssertionError(f"constructor order: {expected} != {cls}")
                obj.__dict__.clear()
                cls.__init__(obj, *args, **kwargs)
            else:
                obj = cls(*args, **kwargs)
                self.recorded.append((cls, obj))
            return obj
        return create

    @contextmanager
    def installed(self):
        with patch.multiple(environment_module, **{
            cls.__name__: self.wrap(cls) for cls in (Agent, Predator, Fruit, Tree)
        }):
            yield self


def exact(value):
    """Compare floats bit for bit, including signed zero; normalize NumPy scalars."""
    if isinstance(value, (float, np.floating)):
        return ("number", struct.pack("!d", float(value)).hex())
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return ("number", struct.pack("!d", float(value)).hex())
    if isinstance(value, dict):
        return {key: exact(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(v) for v in value]
    return value


def state(env):
    return exact({
        "score": env.score, "time": env.time,
        "next_agent_id": env._next_agent_id, "next_fruit_id": env._next_fruit_id,
        "rng": env.rng.getstate(),
        "observations": env.agent_observations,
        "agent_dict": list(env.agents_dict), "fruit_dict": list(env.fruits_dict),
        **{key: [(id(obj), {k: v for k, v in vars(obj).items() if k != "rng"})
                 for obj in getattr(env, key)] for key in COLLECTIONS},
        **{key: {cell: sorted(id(obj) for obj in objects) for cell, objects in value.items()}
           for key, value in vars(env).items() if key.startswith("grid_") and key != "grid_edges"},
    })


def assert_same(a, b, path="state"):
    """Find the first mismatch without printing megabytes of observations/RNG."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: keys differ: {a.keys()} vs {b.keys()}")
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def differential(env, operation):
    before = Snapshot(env)
    with ConstructorReplay().installed() as replay:
        env._backend = "python"
        reference = exact(operation())
        reference_state = state(env)
        before.restore("rust")
        replay.replaying = True
        result = exact(operation())
        assert_same(reference, result, "return")
        assert_same(reference_state, state(env))
        if replay.index != len(replay.recorded):
            raise AssertionError("Rust did not consume all reference constructors")


def actions_for(env, rng, tick):
    actions = [(a.agent_id, ActionRequest(
        agent_id=a.agent_id, move_distance=rng.uniform(-5, 45),
        move_direction=rng.uniform(-math.pi, math.pi),
        turn_angle=rng.uniform(-0.8, 0.8), spawn_agent=(tick % 67 == 0),
    )) for a in env.agents]
    # Duplicates, dead IDs, and children acted upon during the same tick.
    if tick % 31 == 0:
        actions += actions[:1]
        actions += [(999999, ActionRequest(agent_id=999999, move_distance=0,
                                          move_direction=0, turn_angle=0, spawn_agent=False))]
    return actions


@unittest.skipIf(native is None, "build the Rust extension first")
class RustParityTests(unittest.TestCase):
    def make_core(self, seed=4, **kwargs):
        return SimulationCore(env_width=600, env_height=450, chunk_size=400,
                              starting_agents=12, starting_predators=4,
                              starting_fruits=40, starting_trees=25,
                              seed=seed, backend="python", **kwargs)

    def test_visibility_exact(self):
        for implementation in (compute_visibility, native.compute_visibility):
            with self.assertRaises(IndexError):
                implementation(0, 0, 0, math.pi / 3, 200, [])
        rng = random.Random(901)
        for case in range(150):
            edges = [((rng.uniform(-300, 300), rng.uniform(-300, 300)),
                      (rng.uniform(-300, 300), rng.uniform(-300, 300)))
                     for _ in range(rng.randrange(1, 100))]
            args = (rng.uniform(-20, 20), rng.uniform(-20, 20),
                    rng.uniform(-30, 30), rng.uniform(0.1, math.pi), 200, edges)
            with self.subTest(case=case):
                assert_same(exact(compute_visibility(*args)), exact(native.compute_visibility(*args)))

    def test_observation_exact(self):
        core = self.make_core()
        env = core.env
        for creature in env.agents + env.predators:
            a, f, t, o, p, e = env._get_local_objects(creature)
            expected = creature.observe(agents=a, fruits=f, trees=t, obstacles=o, predators=p, edges=e)
            actual = native.observe(creature, a, f, t, p, e)
            assert_same(exact(expected), exact(actual))

    def test_full_trajectories(self):
        for seed in (0, 1, 17, 9059):
            core = self.make_core(seed)
            rng = random.Random(seed + 1000)
            for tick in range(250):
                actions = actions_for(core.env, rng, tick)
                with self.subTest(seed=seed, tick=tick):
                    differential(core.env, lambda: core.step(actions))

    def test_death_rot_old_age_and_reproduction(self):
        core = self.make_core(42, dt=0.37)
        env = core.env
        for index, agent in enumerate(env.agents):
            agent.energy = 0.01 if index < 3 else 499.0
            agent.age = 130.0
        for index, fruit in enumerate(env.fruits):
            fruit.age = 101 if index < 3 else 90
            fruit.energy = 65
        for tree in env.trees:
            tree.age = 101
        for predator in env.predators:
            predator.resting = False
            predator.energy = 0.01
        rng = random.Random(123)
        for tick in range(50):
            actions = actions_for(env, rng, tick)
            differential(env, lambda: core.step(actions))

    def test_movement_and_spawning(self):
        env = self.make_core(19).env
        agent = env.agents[0]
        for energy in (0.1, 99.9, 100.0, 499.0):
            for distance in (-10.0, 0.0, 10.0, 20.0, 100.0):
                for direction in (None, -math.pi, 0.0, math.pi):
                    agent.energy = energy
                    differential(env, lambda: env.update_entity_position(
                        agent, distance, direction, env._get_local_obstacles(agent)))
        for _ in range(60):
            differential(env, lambda: (env.spawn_agent(parent=agent) is None,
                                      env.spawn_fruit(max_attempts=0) is None,
                                      env.spawn_tree(max_attempts=0) is None,
                                      env.spawn_predator() is None))

    def test_backend_and_reset(self):
        self.assertEqual(resolve_backend("rust"), "rust")
        self.assertEqual(resolve_backend("python"), "python")
        with self.assertRaises(ValueError):
            resolve_backend("typo")
        core = self.make_core(3)
        reference = [{k: v for k, v in vars(a).items() if k != "rng"} for a in core.env.agents]
        rng = core.rng.getstate()
        core.backend = "rust"
        core.reset()
        assert_same(exact(reference), exact([{k: v for k, v in vars(a).items() if k != "rng"} for a in core.env.agents]))
        self.assertEqual(rng, core.rng.getstate())
        self.assertEqual(core.env._backend, "rust")

    def test_long_lifecycle_and_late_game(self):
        core = self.make_core(91, dt=1.0)
        env = core.env
        # Explicitly exercise predator spawning at the end of a 3000s game.
        env.time = 2500.0
        for tick in range(600):
            with self.subTest(tick=tick):
                differential(env, lambda: core.step([]))

    def test_contact_ties_thresholds_and_mutation_caps(self):
        core = self.make_core(55)
        env = core.env
        a, b = env.agents[:2]
        b.x, b.y = a.x, a.y
        a.energy, b.energy = 450, 123
        for fruit in env.fruits[:3]:
            fruit.x, fruit.y = a.x, a.y
        pred = env.predators[0]
        pred.x, pred.y = a.x, a.y
        pred.resting = False
        pred.energy = 150
        env._update_spatial_grid()
        differential(env, lambda: core.step([]))
        # Mutation clamping and repeated parent spawning, including failed placement.
        parent = env.agents[-1]
        for attr in ("speed", "sprint_speed", "max_energy", "hearing_radius", "vision_radius", "cone_angle"):
            setattr(parent, attr, 10000.0)
        for _ in range(30):
            differential(env, lambda: env.spawn_agent(parent=parent) is None)

    def test_collision_sliding_and_biomes(self):
        env = self.make_core(7).env
        agent = env.agents[0]
        from src.elements.biome import Forest_biome, Grassland_biome, Swamp_biome, Desert_biome, River_biome
        # Every biome multiplier, world clamping, zero movement and overlapping walls.
        for biome_type in (Forest_biome, Grassland_biome, Swamp_biome, Desert_biome, River_biome):
            for position in ((35., 35.), (31., 200.), (590., 440.), (300., 225.)):
                for heading in (-math.pi, -0.0, 0.0, math.pi / 2, 10000.):
                    agent.x, agent.y = position
                    agent.direction = heading
                    agent.energy = 499.
                    env.biome_map[int(agent.x), int(agent.y)] = biome_type()
                    differential(env, lambda: env.update_entity_position(
                        agent, 20., None, env._get_local_obstacles(agent)))


if __name__ == "__main__":
    unittest.main()

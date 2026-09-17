"""Arranged arenas for controlled tests: flat terrain, chosen obstacles, placed creatures.

Uses the unmodified engine classes; only the initial state is arranged.
"""
from __future__ import annotations

import os
import random

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from src.elements.environment import Environment      # noqa: E402
from src.elements.agent import Agent                  # noqa: E402
from src.elements.predator import Predator            # noqa: E402
from src.elements.biome import Forest_biome           # noqa: E402
from src.utils.simulation import step_environment     # noqa: E402


class Arena:
    def __init__(self, seed=0, width=1600, height=1200, flat=True):
        self.rng = random.Random(seed)
        self.env = Environment(width, height, 400, self.rng)
        env = self.env
        if flat:
            env.biome_map[:, :] = Forest_biome()
        env.obstacles = env.obstacles[:4]          # keep the arena boundaries only
        self._rebuild()
        env.trees = []
        env.fruits = []
        env.agents_dict = {}

    def _rebuild(self):
        env = self.env
        env.edges = set()
        for obs in env.obstacles:
            env.edges.update([((obs.x, obs.y), (obs.x + obs.width, obs.y)),
                              ((obs.x + obs.width, obs.y), (obs.x + obs.width, obs.y + obs.height)),
                              ((obs.x, obs.y + obs.height), (obs.x + obs.width, obs.y + obs.height)),
                              ((obs.x, obs.y), (obs.x, obs.y + obs.height))])
        env._update_spatial_grid()

    def add_rect(self, x, y, w, h):
        return self.env.spawn_obstacle(x=x, y=y, width=w, height=h)

    def add_agent(self, x, y, heading=0.0, energy=150.0, **traits):
        env = self.env
        a = Agent(x=x, y=y, rng=env.rng, energy=energy, **traits)
        a.direction = heading
        a.agent_id = env._next_agent_id
        env._next_agent_id += 1
        env.agents.append(a)
        env.agents_dict[a.agent_id] = a
        env._update_agent_grid()
        return a

    def add_predator(self, x, y, heading=0.0, energy=102.0, resting=False):
        env = self.env
        p = Predator(x, y, rng=env.rng)
        p.direction = heading
        p.energy = energy
        p.resting = resting
        env.predators.append(p)
        env._update_predator_grid()
        return p

    def step(self, actions):
        return step_environment(self.env, actions, 0.1)

    def observations(self):
        """Observation DTOs as the server would return them (after a no-op step)."""
        return [self.env.get_agent_state(a.agent_id) for a in self.env.agents]

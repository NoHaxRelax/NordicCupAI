"""Native survival simulator engine with the vendor `SimulationCore` interface.

    from fastsim import SimulationCore
    sim = SimulationCore(seed=1)                 # same defaults as src.core
    sim = SimulationCore(seed=1, predators=False)  # like harness_np's no-op spawn_predator
    state = sim.step(actions)                    # same dict as step_environment

`sim.env` is a read-only view (agents, fruits, trees, predators, time, score,
_next_agent_id, agents_dict) built on demand; it is a snapshot, not live objects.
`sim.pop_events()` returns deaths and eaten fruit since the last call, which is
what the research harness otherwise gets by wrapping kill_agent/remove_fruit.
"""
import random as _random
from types import SimpleNamespace

try:
    from . import _engine
except ImportError as exc:  # pragma: no cover
    raise ImportError('fastsim._engine is not built; run: python fastsim/build.py') from exc

import numpy as _np

# Use numpy's own float64 loops (they differ from libm on some CPUs, e.g. arctan2 on AVX-512).
_engine.set_numpy_loops(_np.sin, _np.cos, _np.arctan2, _np.hypot)

BIOME_NAMES = ('forest', 'swamp', 'desert', 'grassland', 'river')


def seed_key(seed):
    """32-bit words CPython's random.seed(int) feeds to init_by_array."""
    if not isinstance(seed, int):
        raise TypeError('fastsim only supports integer seeds (like the official server)')
    n = abs(seed)
    words = []
    while n:
        words.append(n & 0xFFFFFFFF)
        n >>= 32
    return words or [0]


class FastEnv:
    """Snapshot accessors mirroring the attributes the harnesses read."""

    def __init__(self, engine, width, height):
        self._e = engine
        self.width, self.height = width, height

    @property
    def time(self):
        return self._e.info()['time']

    @property
    def score(self):
        return self._e.info()['score']

    @property
    def _next_agent_id(self):
        return self._e.info()['next_agent_id']

    @property
    def agents(self):
        out = []
        for (aid, x, y, d, age, en, maxe, sp, spr, hear, vis, vis_int, cone, max_age) in self._e.agents():
            out.append(SimpleNamespace(agent_id=aid, x=x, y=y, direction=d, age=age, energy=en, max_energy=maxe,
                                       speed=sp, sprint_speed=spr, hearing_radius=hear,
                                       vision_radius=int(vis) if vis_int else vis, cone_angle=cone,
                                       max_age=max_age, size=5))
        return out

    @property
    def agents_dict(self):
        return {a.agent_id: a for a in self.agents}

    @property
    def predators(self):
        return [SimpleNamespace(x=x, y=y, direction=d, energy=e, resting=r, size=10, predator_id=pid)
                for pid, (x, y, d, e, r) in zip(self._e.predator_ids(), self._e.predators())]

    def enable_diagnostics(self, enabled):
        self._e.diagnostics(enabled)

    def pop_diagnostics(self):
        self._e.pop_events()  # The research collector consumes the richer stream below.
        return self._e.pop_diagnostics()

    @property
    def fruits(self):
        return [SimpleNamespace(fruit_id=i, x=x, y=y, energy=e, age=a, radius=r)
                for (i, x, y, e, a, r, _) in self._e.fruits()]

    @property
    def trees(self):
        return [SimpleNamespace(x=x, y=y, radius=r, age=a) for (x, y, r, a) in self._e.trees()]

    @property
    def obstacles(self):
        return [SimpleNamespace(x=x, y=y, width=w, height=h) for (x, y, w, h) in self._e.obstacles()]

    def biome_at(self, x, y):
        bm = self._e.biome_map()
        ix = min(max(int(x), 0), self.width - 1)
        iy = min(max(int(y), 0), self.height - 1)
        return BIOME_NAMES[bm[ix * self.height + iy]]


class SimulationCore:
    def __init__(self, env_width=1600, env_height=1200, chunk_size=400, starting_agents=5, starting_predators=0,
                 starting_fruits=None, starting_trees=50, seed=None, dt=1 / 10, predators=True):
        self.env_width = env_width
        self.env_height = env_height
        self.chunk_size = chunk_size
        self.starting_agents = starting_agents
        self.starting_predators = starting_predators
        self.starting_fruits = starting_fruits or env_width // 50
        self.starting_trees = starting_trees
        self.dt = dt
        self.predators = predators
        if seed is None:
            seed = _random.randint(0, 2 ** 32 - 1)
        self.seed = seed
        self._create()

    def _create(self):
        self._engine = _engine.Engine(seed_key(self.seed), self.env_width, self.env_height, self.chunk_size,
                                      self.starting_agents, self.starting_predators, self.starting_fruits,
                                      self.starting_trees, self.dt, self.predators)
        self.env = FastEnv(self._engine, self.env_width, self.env_height)

    def step(self, actions):
        return self._engine.step(actions)

    def state(self):
        return self._engine.state()

    def pop_events(self):
        return self._engine.pop_events()

    def reset(self):
        self._create()

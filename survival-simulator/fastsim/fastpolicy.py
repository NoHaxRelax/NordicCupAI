"""Engine + native policy in one process, with the tick loop inside C++.

    from fastsim.fastpolicy import PolicySimulationCore
    sim = PolicySimulationCore(seed=1, predators=True)
    sim.policy_init(1, config)          # config as OrchardEvasionPolicy keywords
    steps, peak = sim.run_policy(3000)  # ONE Python call for a whole 3000 s game

`PolicySimulationCore` is `fastsim.SimulationCore` backed by the `_policy` extension
instead of `_engine`: the same engine code, plus the native orchard + evasion policy
linked in beside it. `run_policy` decides and steps entirely in C++ with the GIL
released, so a full game costs O(1) Python calls rather than O(30000) and nothing is
serialized between the engine and the policy.

`policy_act()` is the lockstep-verification entry point (decide without stepping, and
hand the actions back as Python tuples); it is not the fast path.

The policy itself is compiled against fastsim/policy_abi.hpp only and cannot see the
engine - see that header and fastsim/check_boundary.py.
"""
import numpy as _np

from . import FastEnv, SimulationCore, seed_key  # noqa: F401  (seed_key is re-exported)

try:
    from . import _policy as _mod
except ImportError as exc:  # pragma: no cover
    raise ImportError('fastsim._policy is not built; run: python fastsim/build_policy.py') from exc

# The _policy module holds its own copy of the engine, so it needs its own loop
# pointers; fastsim/__init__.py does the same for _engine.
_mod.set_numpy_loops(_np.sin, _np.cos, _np.arctan2, _np.hypot)


class PolicySimulationCore(SimulationCore):
    def _create(self):
        self._engine = _mod.Engine(seed_key(self.seed), self.env_width, self.env_height, self.chunk_size,
                                   self.starting_agents, self.starting_predators, self.starting_fruits,
                                   self.starting_trees, self.dt, self.predators)
        self.env = FastEnv(self._engine, self.env_width, self.env_height)

    def policy_init(self, seed, config=None):
        """Create the native policy. `config` takes OrchardEvasionPolicy keywords."""
        self._engine.policy_init(seed_key(seed), dict(config or {}))

    def policy_act(self):
        """Decisions for the current state, without stepping: [(aid, dist, dir, turn, spawn)]."""
        return self._engine.policy_act()

    def run_policy(self, horizon, stop_at=None):
        """Advance to `horizon` (or until extinction) inside C++. -> (steps, peak_agents)."""
        return self._engine.run_policy(float(horizon), float(horizon if stop_at is None else stop_at))

    def policy_metrics(self):
        return self._engine.policy_metrics()

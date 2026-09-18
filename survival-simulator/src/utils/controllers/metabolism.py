"""Estimate passive energy use from public observations and issued actions.

Call ``update(states, sim_time)`` before choosing actions, then
``remember_actions(actions)`` with the actions actually submitted. Returned
rates are energy per simulated second, including a small planning buffer.
Fruit can conceal an energy loss, so an observed meal never clears confirmed
senescence. No simulator object or private lifespan is needed.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass
class _History:
    state: dict
    healthy_at_age: float | None = None
    positive_samples: int = 0
    aging: bool = False


class MetabolismTracker:
    def __init__(self, walking_energy_per_unit=.05, sprinting_energy_per_unit=.5,
                 spawn_energy_cost=100., sprint_min_energy_fraction=.2,
                 baseline_rate=1., reserve_buffer=.2):
        self.walking_cost = walking_energy_per_unit
        self.sprinting_cost = sprinting_energy_per_unit
        self.spawn_cost = spawn_energy_cost
        self.sprint_min_energy_fraction = sprint_min_energy_fraction
        self.baseline_rate = baseline_rate
        self.reserve_buffer = reserve_buffer
        self.reset()

    def reset(self):
        self.history = {}
        self.pending_costs = {}
        self.rates = {}
        self.last_time = None

    @staticmethod
    def _field(action, name):
        return action[name] if isinstance(action, Mapping) else getattr(action, name)

    def _action_cost(self, state, action):
        distance = max(0., min(float(self._field(action, "move_distance")), state["sprint_speed"]))
        if (state["energy"] < state["max_energy"] * self.sprint_min_energy_fraction
                and distance > state["speed"]):
            distance = state["speed"]
        movement = (min(distance, state["speed"]) * self.walking_cost
                    + max(0., distance - state["speed"]) * self.sprinting_cost)
        turning = min(math.pi, abs(float(self._field(action, "turn_angle")))) / math.tau
        cost = movement + turning
        # Movement and turning are charged before the engine checks whether a
        # requested birth has enough energy to proceed.
        if self._field(action, "spawn_agent") and state["energy"] - cost > self.spawn_cost:
            cost += self.spawn_cost
        return cost

    def remember_actions(self, actions):
        """Remember one issued action per agent, replacing same-step retries."""
        for action in actions:
            agent_id = self._field(action, "agent_id")
            if agent_id in self.history:
                self.pending_costs[agent_id] = self._action_cost(self.history[agent_id].state, action)

    def estimate(self, state, dt=.1):
        """Predict passive drain using public onset bounds and observed losses.

        A clean sample showing normal upkeep narrows the otherwise unknown
        onset interval. Missing or food-masked samples only increase that
        interval's uncertainty; they cannot demonstrate rejuvenation.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        age = float(state["age"])
        record = self.history.get(state["agent_id"])
        age_rate = max(0., age) * .01 / dt
        if record is not None and (record.aging or record.positive_samples):
            return self.baseline_rate + self.reserve_buffer + age_rate
        lower = 60.
        if record is not None and record.healthy_at_age is not None:
            lower = max(lower, record.healthy_at_age)
        if age <= lower:
            probability = 0.
        elif lower >= 120.:
            probability = 1.
        else:
            probability = min(1., (age - lower) / (120. - lower))
        return self.baseline_rate + self.reserve_buffer + probability * age_rate

    def update(self, states, sim_time=None, dt=.1):
        """Consume public states and return rates for the living population.

        Inference uses each agent's age increment, not elapsed world time:
        some engine steps expose unchanged age after another agent dies.
        Unchanged-age samples update energy bookkeeping without inventing
        passive drain. Missing actions or skipped observation frames are not
        evidence of aging.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not states:
            self.reset()
            return {}
        rewind = (sim_time is not None and self.last_time is not None and sim_time < self.last_time)
        rewind = rewind or any(state["agent_id"] in self.history
            and state["age"] < self.history[state["agent_id"]].state["age"] for state in states)
        if rewind:
            self.reset()
        if sim_time is not None and sim_time == self.last_time:
            return self.rates.copy()
        living = {state["agent_id"] for state in states}
        self.history = {key: value for key, value in self.history.items() if key in living}
        self.pending_costs = {key: value for key, value in self.pending_costs.items() if key in living}
        for state in states:
            agent_id = state["agent_id"]
            current = {key: float(state[key]) for key in
                       ("age", "energy", "speed", "sprint_speed", "max_energy")}
            record = self.history.get(agent_id)
            cost = self.pending_costs.pop(agent_id, None)
            if record is None:
                self.history[agent_id] = _History(current)
                continue
            elapsed = current["age"] - record.state["age"]
            if cost is not None and math.isclose(elapsed, dt, rel_tol=1e-6, abs_tol=1e-8):
                passive = (record.state["energy"] - current["energy"] - cost) / elapsed
                expected_extra = current["age"] * .01 / dt
                # An unmodeled/missing charge should not be mistaken for an
                # aging signature. Real onset adds approximately age*.01 each
                # step; fruit can reduce, never increase, the observed loss.
                plausible_aging = (current["age"] > 60.
                    and self.baseline_rate + .5 * expected_extra <= passive
                    <= self.baseline_rate + expected_extra + .3)
                if plausible_aging:
                    record.positive_samples += 1
                    record.aging |= record.positive_samples >= 2
                elif abs(passive - self.baseline_rate) <= .2 and not record.aging:
                    record.healthy_at_age = current["age"]
                    record.positive_samples = 0
                # Negative residuals, including capped meals, are deliberately
                # ignored. Once onset is confirmed it cannot be cleared.
            record.state = current
        self.rates = {state["agent_id"]: self.estimate(state, dt) for state in states}
        self.last_time = sim_time
        return self.rates.copy()

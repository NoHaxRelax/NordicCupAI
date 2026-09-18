"""Founder-normalized inherited traits and energy-conscious breeding priorities."""

from dataclasses import dataclass
import math

from pydantic import BaseModel, ConfigDict, Field

from models.exploration.policy_inputs import ReproductionHint


TRAITS = ("speed", "sprint_speed", "hearing_radius", "vision_angle", "vision_range", "max_energy")


class GrowthPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    normal_energy_threshold: float = Field(default=260.0, ge=0)
    elite_energy_threshold: float = Field(default=220.0, ge=0)
    low_rank_energy_threshold: float = Field(default=300.0, ge=0)
    normal_cooldown_seconds: float = Field(default=8.0, ge=0)
    elite_cooldown_seconds: float = Field(default=5.0, ge=0)
    growth_population_target: int = Field(default=60, ge=1, strict=True)


class PopulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    elite_fraction: float = Field(default=0.2, ge=0, le=0.5)
    dispersal_fraction: float = Field(default=0.2, ge=0, le=0.5)
    normal_energy_threshold: float = Field(default=260.0, ge=0)
    elite_energy_threshold: float = Field(default=220.0, ge=0)
    low_rank_energy_threshold: float = Field(default=300.0, ge=0)
    normal_cooldown_seconds: float = Field(default=12.0, ge=0)
    elite_cooldown_seconds: float = Field(default=8.0, ge=0)
    growth_population_target: int = Field(default=40, ge=1, strict=True)
    capacity_threshold_fraction: float = Field(default=0.85, gt=0, le=1)
    post_alignment: GrowthPhaseConfig = Field(default_factory=GrowthPhaseConfig)


@dataclass(frozen=True)
class TraitRating:
    score: float
    elite: bool
    low_rank: bool
    percentile: float
    ratios: dict[str, float]


class PopulationTracker:
    def __init__(self, config: PopulationConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.baseline: dict[str, float] = {}
        self.ratings: dict[int, TraitRating] = {}
        self.last_birth_request: dict[int, float] = {}
        self.ages: dict[int, float] = {}
        self.capacities: dict[int, float] = {}
        self.last_time = None
        self.population_phase = False

    def update(self, states, now):
        if not states:
            self.reset()
            return
        if (self.last_time is not None and now < self.last_time) or any(
            state["agent_id"] in self.ages and state["age"] < self.ages[state["agent_id"]]
            for state in states
        ):
            self.reset()
        if self.last_time == now:
            return
        ordered = sorted(states, key=lambda state: state["agent_id"])
        if not self.baseline:
            # Freeze the first complete population; later children and deaths
            # must never change the meaning of a score of 1.0.
            for trait in TRAITS:
                values = [float(state[trait]) for state in ordered]
                if not all(math.isfinite(value) and value > 0 for value in values):
                    raise ValueError(f"Initial {trait} values must be finite and positive")
                self.baseline[trait] = math.fsum(values) / len(values)
        scores, ratios = {}, {}
        for state in ordered:
            agent_id = state["agent_id"]
            ratios[agent_id] = {trait: float(state[trait]) / self.baseline[trait] for trait in TRAITS}
            scores[agent_id] = math.fsum(ratios[agent_id].values()) / len(TRAITS)
        ranked = sorted(scores, key=lambda agent_id: (-scores[agent_id], agent_id))
        count = len(ranked)
        varied = scores[ranked[0]] - scores[ranked[-1]] > 1e-9
        elite_count = math.ceil(count * self.config.elite_fraction) if varied else 0
        low_count = min(count - elite_count, math.ceil(count * self.config.dispersal_fraction)) if varied else 0
        elites = set(ranked[:elite_count])
        lower = set(ranked[count - low_count:]) if low_count else set()
        self.ratings = {}
        start = 0
        while start < count:
            end = start + 1
            while end < count and abs(scores[ranked[end]] - scores[ranked[start]]) <= 1e-9:
                end += 1
            # Ties share the display percentile; IDs break only the bounded
            # breeding/scouting selection at the 20-percent cutoffs.
            percentile = (count - end + (end - start - 1) / 2) / (count - 1) if count > 1 else 0.5
            for agent_id in ranked[start:end]:
                self.ratings[agent_id] = TraitRating(scores[agent_id], agent_id in elites,
                                                    agent_id in lower, percentile, ratios[agent_id])
            start = end
        self.ages = {state["agent_id"]: state["age"] for state in ordered}
        self.capacities = {state["agent_id"]: state["max_energy"] for state in ordered}
        self.last_birth_request = {key: value for key, value in self.last_birth_request.items() if key in self.ages}
        self.last_time = now

    def reproduction_hint(self, agent_id, now, base_threshold):
        if not self.config.enabled or agent_id not in self.ratings:
            return None
        rating = self.ratings[agent_id]
        config = (self.config.post_alignment if self.population_phase and self.config.post_alignment.enabled
                  else self.config)
        threshold = (config.elite_energy_threshold if rating.elite else
                     config.low_rank_energy_threshold if rating.low_rank else config.normal_energy_threshold)
        if len(self.ratings) >= config.growth_population_target:
            # More bodies eventually consume the local food budget. Fall back
            # to the previous conservative threshold after the growth phase.
            threshold = max(threshold, base_threshold)
        # Low-capacity mutations must still be able to reproduce when the
        # independent post-action energy reserve can be met.
        threshold = min(threshold, self.capacities[agent_id] * self.config.capacity_threshold_fraction)
        cooldown = config.elite_cooldown_seconds if rating.elite else config.normal_cooldown_seconds
        previous = self.last_birth_request.get(agent_id)
        allowed = previous is None or now == previous or now - previous >= cooldown - 1e-9
        return ReproductionHint(threshold, allowed)

    def remember_actions(self, actions, now):
        for action in actions:
            if action.spawn_agent:
                self.last_birth_request[action.agent_id] = now

    def snapshot(self):
        config = (self.config.post_alignment if self.population_phase and self.config.post_alignment.enabled
                  else self.config)
        return {
            "baseline": dict(self.baseline), "trait_names": list(TRAITS),
            "living": len(self.ratings), "selection_enabled": self.config.enabled,
            "elite_ids": [key for key, value in sorted(self.ratings.items()) if value.elite],
            "disperser_ids": [key for key, value in sorted(self.ratings.items()) if value.low_rank],
            "growth_population_target": config.growth_population_target,
            "phase": "population" if self.population_phase else "alignment",
            "score_definition": "mean(trait / initial_population_mean[trait])",
        }

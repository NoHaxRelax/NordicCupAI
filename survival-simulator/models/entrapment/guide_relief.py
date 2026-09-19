"""Observation-only forecasts for replacing a tiring predator guide.

The caller supplies route estimates made from the shared observed-wall map.  No
predator state, simulator coordinates, or hidden terrain enters this module.
All times returned by this module are seconds relative to the current tick.
"""

from dataclasses import dataclass, field, replace
import math
from typing import Hashable, Iterable


SPRINT_CUTOFF_FRACTION = 0.2
TICK_SECONDS = 0.1
WALK_COST_PER_UNIT = 0.05
SPRINT_COST_PER_UNIT = 0.5


@dataclass(frozen=True)
class AgentForecastInput:
    agent_id: Hashable
    energy: float
    max_energy: float
    age: float
    speed: float
    sprint_speed: float
    route_distance: float
    route_time: float | None = None
    terrain_progress: float = 1.0
    biome_drain_per_second: float = 1.0
    available: bool = True


@dataclass(frozen=True)
class TravelForecast:
    arrival_time: float
    sprint_cutoff_time: float
    death_time: float
    energy_at_arrival: float
    arrives: bool
    arrives_with_sprint: bool


@dataclass(frozen=True)
class ReliefPlan:
    reserve_id: Hashable | None
    rendezvous: object
    dispatch_in: float | None
    deadline: float
    release_old_guide: bool
    reason: str
    diagnostics: dict = field(default_factory=dict)


def _validate(agent: AgentForecastInput) -> None:
    values = (agent.energy, agent.max_energy, agent.age, agent.speed,
              agent.sprint_speed, agent.route_distance, agent.terrain_progress,
              agent.biome_drain_per_second)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("agent forecast values must be finite")
    if (agent.max_energy <= 0 or agent.energy < 0 or agent.age < 0
            or agent.speed <= 0 or agent.sprint_speed <= 0
            or agent.route_distance < 0 or agent.terrain_progress <= 0
            or agent.biome_drain_per_second < 0):
        raise ValueError("agent forecast values are outside their native ranges")
    if agent.route_time is not None and (not math.isfinite(agent.route_time)
                                         or agent.route_time < 0):
        raise ValueError("route_time must be finite and nonnegative")


def forecast_travel(agent: AgentForecastInput, *, wants_sprint: bool = True,
                    horizon: float = 600.0) -> TravelForecast:
    """Conservatively replay native movement/age costs over an observed route.

    ``terrain_progress`` is physical progress per requested movement unit (for
    example .3 in river).  ``route_time``, when supplied by a navigator, is a
    lower bound on arrival time and can represent turns, congestion, or a paced
    route.  The replay still charges movement over the full route distance.
    """
    _validate(agent)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    remaining, energy, age = agent.route_distance, agent.energy, agent.age
    elapsed = 0.0
    cutoff = 0.0 if energy < agent.max_energy * SPRINT_CUTOFF_FRACTION else math.inf
    death = math.inf
    energy_at_route_end = energy
    route_end = 0.0 if remaining == 0 else math.inf

    while elapsed < horizon and energy > 0 and remaining > 1e-9:
        can_sprint = wants_sprint and energy >= agent.max_energy * SPRINT_CUTOFF_FRACTION
        requested = (agent.sprint_speed if can_sprint
                     else min(agent.speed, agent.sprint_speed))
        requested = min(requested, remaining / agent.terrain_progress)
        walk = min(requested, agent.speed)
        movement_cost = walk * WALK_COST_PER_UNIT + max(0.0, requested-walk) * SPRINT_COST_PER_UNIT
        energy -= movement_cost + TICK_SECONDS * agent.biome_drain_per_second
        age += TICK_SECONDS
        if age > 60.0:
            energy -= 0.01 * age
        elapsed += TICK_SECONDS
        remaining -= requested * agent.terrain_progress
        if cutoff == math.inf and energy < agent.max_energy * SPRINT_CUTOFF_FRACTION:
            cutoff = elapsed
        if energy <= 0:
            death = elapsed
            break
    if remaining <= 1e-9:
        route_end, energy_at_route_end = elapsed, max(0.0, energy)

    arrival = max(route_end, agent.route_time or 0.0)
    energy_at_arrival = energy_at_route_end if arrival <= elapsed else 0.0

    # Continue idling to estimate the death deadline; no future food is assumed.
    while elapsed < horizon and energy > 0:
        energy -= TICK_SECONDS * agent.biome_drain_per_second
        age += TICK_SECONDS
        if age > 60.0:
            energy -= 0.01 * age
        elapsed += TICK_SECONDS
        if cutoff == math.inf and energy < agent.max_energy * SPRINT_CUTOFF_FRACTION:
            cutoff = elapsed
        if energy_at_arrival == 0.0 and elapsed >= arrival:
            energy_at_arrival = max(0.0, energy)
    if energy <= 0 and death == math.inf:
        death = elapsed

    arrives = math.isfinite(route_end) and arrival < death
    return TravelForecast(arrival, cutoff, death, energy_at_arrival,
                          arrives, arrives and arrival < cutoff)


def _after_wait(agent: AgentForecastInput, delay: float) -> AgentForecastInput | None:
    """Project a stationary, unfed donor to dispatch using native idle/age loss."""
    energy, age, elapsed = agent.energy, agent.age, 0.0
    while elapsed + 1e-9 < delay and energy > 0:
        step = min(TICK_SECONDS, delay-elapsed)
        energy -= step * agent.biome_drain_per_second
        age += step
        # Native old-age loss is per simulation tick, not scaled by dt.
        if age > 60.0:
            energy -= 0.01 * age
        elapsed += step
    if energy <= 0:
        return None
    return replace(agent, energy=energy, age=age)


def _sprint_reserve(agent: AgentForecastInput, seconds: float) -> float:
    """Energy above cutoff needed for a short full-speed handover sprint."""
    ticks = math.ceil(seconds / TICK_SECONDS)
    requested = agent.sprint_speed
    walk = min(requested, agent.speed)
    move = walk * WALK_COST_PER_UNIT + max(0.0, requested-walk) * SPRINT_COST_PER_UNIT
    # Age loss is evaluated at dispatch age. This is a small reserve, not a
    # future-life proof; plan_relief is expected to run again on every DTO.
    old = 0.01 * (agent.age + seconds) if agent.age + seconds > 60.0 else 0.0
    return ticks * (move + TICK_SECONDS * agent.biome_drain_per_second + old)


def plan_relief(current: AgentForecastInput,
                candidates: Iterable[AgentForecastInput], *, rendezvous: object,
                handover_margin: float = 2.0, preparation_time: float = 0.5,
                handover_sprint_seconds: float = 0.5,
                sprint_required_to_bait: bool = True,
                replacement_at_rendezvous: bool = False,
                replacement_sees_predator: bool = False,
                old_guide_sees_predator_following_replacement: bool = False) -> ReliefPlan:
    """Choose a feasible reserve and say when it must leave ordinary gathering.

    A reserve remains a gatherer until ``dispatch_in`` reaches zero.  Releasing
    the old guide requires a physical overlap plus reciprocal evidence from
    ordinary predator sightings; a planned or merely arrived replacement is
    insufficient.
    """
    if handover_margin < 0 or preparation_time < 0 or handover_sprint_seconds < 0:
        raise ValueError("relief timing values must be nonnegative")
    guide = forecast_travel(current, wants_sprint=True)
    failure_deadline = min(guide.death_time,
                           guide.sprint_cutoff_time if sprint_required_to_bait else math.inf)
    needs_relief = not guide.arrives or guide.arrival_time + handover_margin >= failure_deadline
    rows = []
    for candidate in candidates:
        if not candidate.available or candidate.agent_id == current.agent_id:
            continue
        immediate = forecast_travel(candidate, wants_sprint=False)
        reserve = _sprint_reserve(candidate, handover_sprint_seconds)

        def at(delay):
            delayed = _after_wait(candidate, delay)
            if delayed is None:
                return None, False
            forecast = forecast_travel(delayed, wants_sprint=False)
            absolute_arrival = delay + forecast.arrival_time
            enough_energy = (forecast.energy_at_arrival >=
                             candidate.max_energy * SPRINT_CUTOFF_FRACTION + reserve)
            feasible = (forecast.arrives and enough_energy
                        and absolute_arrival + handover_margin < failure_deadline)
            return forecast, feasible

        _, immediate_feasible = at(0.0)
        latest_dispatch = -math.inf
        delayed_forecast = immediate
        if immediate_feasible:
            high = min(600.0, max(0.0, failure_deadline-handover_margin))
            low = 0.0
            # Feasibility only worsens in this stationary/no-food projection.
            for _ in range(32):
                middle = (low+high)/2
                probe, okay = at(middle)
                if okay:
                    low, delayed_forecast = middle, probe
                else:
                    high = middle
            latest_dispatch = math.floor((low+1e-9)/TICK_SECONDS) * TICK_SECONDS
            delayed_forecast, quantized_feasible = at(latest_dispatch)
            if not quantized_feasible:
                latest_dispatch = -math.inf
        feasible = immediate_feasible and latest_dispatch >= preparation_time
        rows.append((candidate, delayed_forecast, latest_dispatch, feasible,
                     immediate, reserve))

    feasible_rows = [row for row in rows if row[3]]
    # Prefer the donor that can stay productive longest, then the safer arrival.
    chosen = max(feasible_rows,
                 key=lambda row: (row[2], row[1].death_time-row[1].arrival_time,
                                  row[1].energy_at_arrival), default=None)
    confirmed = (replacement_at_rendezvous and replacement_sees_predator
                 and old_guide_sees_predator_following_replacement)
    diagnostics = {
        "guide": guide,
        "failure_deadline": failure_deadline,
        "needs_relief": needs_relief,
        "candidates": {row[0].agent_id: {"forecast": row[1],
                         "latest_dispatch": row[2], "feasible": row[3],
                         "immediate_forecast": row[4],
                         "handover_sprint_energy_reserve": row[5]} for row in rows},
        "handover_confirmed": confirmed,
    }
    if not needs_relief:
        return ReliefPlan(None, rendezvous, None, failure_deadline, False,
                          "guide forecast reaches bait before sprint/death deadline", diagnostics)
    if chosen is None:
        return ReliefPlan(None, rendezvous, None, failure_deadline, False,
                          "no observed-map donor can meet the handover deadline", diagnostics)
    candidate, _, latest_dispatch, _, _, _ = chosen
    return ReliefPlan(candidate.agent_id, rendezvous, max(0.0, latest_dispatch),
                      failure_deadline, confirmed,
                      "handover confirmed" if confirmed else "reserve selected; await dispatch/observed handover",
                      diagnostics)

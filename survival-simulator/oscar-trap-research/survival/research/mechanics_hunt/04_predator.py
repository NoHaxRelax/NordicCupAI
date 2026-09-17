"""Predator state-machine probes against the unmodified vendored simulator.

This is local mechanics research, not a competition submission.  Synthetic
fixtures arrange positions and sometimes set predator energy/rest state to
isolate transitions.  Generated-map policy actions use only public cached
agent observations and issue exactly one action per living agent per tick.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

import numpy as np

from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.elements.obstacle import Obstacle
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def controlled_env(seed: int = 404) -> Environment:
    """Build a flat fixture while preserving upstream sensing/tick physics."""
    env = Environment.__new__(Environment)
    env.width, env.height, env.chunk_size = 1600, 1200, 400
    env.rng = random.Random(seed)
    env.agents, env.predators, env.fruits, env.trees, env.obstacles = [], [], [], [], []
    # Real maps always have boundary edges.  Keeping distant boundary edges also
    # avoids the upstream empty-edge visibility failure without affecting probes.
    env.edges = {
        ((0, 0), (1600, 0)),
        ((1600, 0), (1600, 1200)),
        ((0, 1200), (1600, 1200)),
        ((0, 0), (0, 1200)),
    }
    env.agents_dict, env.fruits_dict, env.agent_observations = {}, {}, {}
    env._next_agent_id = env._next_fruit_id = 0
    env.score = env.time = 0.0
    env.biome_map = np.empty((env.width, env.height), dtype=object)
    env.biome_map.fill(Forest_biome())
    # Fixture-only removal of unrelated random spawning; creature mechanics are
    # the original engine methods.
    env.spawn_tree = lambda *args, **kwargs: None
    env.spawn_predator = lambda *args, **kwargs: None
    env._update_spatial_grid()
    return env


def add_agent(env: Environment, x: float, y: float, energy: float = 150.0) -> Agent:
    agent = Agent(x, y, energy=energy, max_age=1e9, rng=env.rng)
    agent.agent_id = env._next_agent_id
    env._next_agent_id += 1
    env.agents.append(agent)
    env.agents_dict[agent.agent_id] = agent
    env._update_agent_grid()
    return agent


def add_predator(
    env: Environment,
    x: float,
    y: float,
    *,
    energy: float = 0.0,
    resting: bool = True,
    direction: float = 0.0,
) -> Predator:
    predator = Predator(x, y, energy=energy, rng=env.rng)
    predator.resting = resting
    predator.direction = direction
    env.predators.append(predator)
    env._update_predator_grid()
    return predator


def request(agent: Agent, *, turn: float = 0.0) -> tuple[int, ActionRequest]:
    action = ActionRequest(
        agent_id=agent.agent_id,
        move_distance=0.0,
        move_direction=0.0,
        turn_angle=turn,
        spawn_agent=False,
    )
    return agent.agent_id, action


def idle_actions(env: Environment) -> list[tuple[int, ActionRequest]]:
    return [request(agent) for agent in list(env.agents)]


def rest_threshold_probe() -> dict:
    """Measure strict wake comparison and the exact overlap-safe window."""
    one_tick = []
    for start_energy in (99.0, 100.0, 100.000001):
        env = controlled_env()
        agent = add_agent(env, 1000, 600)
        agent.direction = math.pi
        predator = add_predator(env, 800, 600, energy=start_energy, resting=True)
        before = (predator.x, predator.y)
        step_environment(env, [request(agent)])
        one_tick.append(
            {
                "start_energy": start_energy,
                "end_energy": predator.energy,
                "resting": predator.resting,
                "distance_moved": math.hypot(predator.x - before[0], predator.y - before[1]),
            }
        )

    assert one_tick[0]["resting"] and math.isclose(one_tick[0]["end_energy"], 102.0)
    assert one_tick[1]["resting"] and math.isclose(one_tick[1]["end_energy"], 103.0)
    assert not one_tick[2]["resting"] and one_tick[2]["distance_moved"] > 0

    env = controlled_env()
    agent = add_agent(env, 800, 600)
    predator = add_predator(env, 800, 600, energy=0.0, resting=True)
    trace = []
    death_tick = None
    for tick in range(1, 50):
        step_environment(env, idle_actions(env))
        trace.append(
            {
                "tick": tick,
                "energy": predator.energy,
                "resting": predator.resting,
                "agent_alive": agent.agent_id in env.agents_dict,
            }
        )
        if agent.agent_id not in env.agents_dict:
            death_tick = tick
            break
    assert death_tick == 35
    assert trace[33]["resting"] and math.isclose(trace[33]["energy"], 102.0)
    assert not trace[34]["resting"]
    return {
        "one_tick_boundary": one_tick,
        "overlap_from_zero": {
            "death_tick": death_tick,
            "safe_completed_ticks": death_tick - 1,
            "safe_seconds": (death_tick - 1) / 10,
            "selected_trace": [trace[i] for i in (0, 32, 33, 34)],
        },
    }


def last_lunge_probe() -> dict:
    """Show that contact and feeding happen before the post-action sleep test."""
    rows = []
    for label, agent_x in (("contact_after_low_energy_move", 814.0), ("no_contact_control", 840.0)):
        env = controlled_env()
        agent = add_agent(env, agent_x, 600, energy=150.0)
        agent.direction = math.pi
        predator = add_predator(env, 800, 600, energy=0.1, resting=False, direction=0.0)
        step_environment(env, [request(agent)])
        rows.append(
            {
                "case": label,
                "agent_start_x": agent_x,
                "predator_end_x": predator.x,
                "predator_end_energy": predator.energy,
                "predator_resting": predator.resting,
                "agent_alive": agent.agent_id in env.agents_dict,
                "score": env.score,
            }
        )
    positive, negative = rows
    assert not positive["agent_alive"]
    assert positive["predator_end_energy"] > 100 and not positive["predator_resting"]
    assert negative["agent_alive"]
    assert negative["predator_end_energy"] < 0 and negative["predator_resting"]
    return {
        "cases": rows,
        "interpretation": (
            "An already-active predator can spend below zero, complete its movement/contact check, "
            "and remain active if the victim replenishes it. Failed contact puts the same predator to sleep."
        ),
    }


def pileup_probe() -> dict:
    """Measure active multi-kill contact and the resting negative control."""
    rows = []
    for resting, energy in ((False, 50.0), (True, 0.0)):
        env = controlled_env()
        agents = [add_agent(env, 800, 600) for _ in range(3)]
        predator = add_predator(env, 800, 600, energy=energy, resting=resting)
        step_environment(env, [request(agent) for agent in agents])
        rows.append(
            {
                "start_resting": resting,
                "survivors": len(env.agents),
                "predator_energy": predator.energy,
                "predator_resting": predator.resting,
                "score": env.score,
            }
        )
    assert rows[0]["survivors"] == 0 and math.isclose(rows[0]["predator_energy"], 200.0)
    assert rows[1]["survivors"] == 3 and rows[1]["predator_resting"]
    return {
        "cases": rows,
        "interpretation": "One active contact pass can kill every overlapping agent; resting skips the whole pass.",
    }


def target_selection_probe() -> dict:
    """Isolate no-commitment nearest selection and input-order tie behavior."""
    predator = Predator(0, 0, energy=200, rng=random.Random(4))
    predator.direction = 0.0

    def observed(distance: float, angle: float) -> dict:
        return {"type": "Agent", "distance": distance, "angle": angle, "rel_dir": math.pi}

    above = observed(40.0, 0.5)
    below = observed(41.0, -0.5)
    first = predator.step([above, below])
    above["distance"], below["distance"] = 41.0, 40.0
    second = predator.step([above, below])

    tie_above = observed(40.0, 0.5)
    tie_below = observed(40.0, -0.5)
    tie_first = predator.step([tie_above, tie_below])
    tie_reversed = predator.step([tie_below, tie_above])
    epsilon_first = predator.step([observed(40.000001, 0.5), observed(40.0, -0.5)])
    epsilon_reversed = predator.step([observed(40.0, -0.5), observed(40.000001, 0.5)])

    assert first["direction"] > 0 and second["direction"] < 0
    assert tie_first["direction"] > 0 and tie_reversed["direction"] < 0
    assert epsilon_first["direction"] < 0 and epsilon_reversed["direction"] < 0
    return {
        "nearest_switch": {
            "first_direction": first["direction"],
            "after_one_unit_distance_swap": second["direction"],
        },
        "equal_distance_tie": {
            "above_first_direction": tie_first["direction"],
            "below_first_direction": tie_reversed["direction"],
            "engine_note": "Predator.step uses the first minimum; Environment supplies a set-derived order, not agent ID order.",
        },
        "epsilon_control": {
            "above_first_direction": epsilon_first["direction"],
            "below_first_direction": epsilon_reversed["direction"],
        },
    }


def gaze_branch_probe() -> dict:
    """Measure discontinuities in the far-target gaze-dependent branch."""
    predator = Predator(0, 0, energy=200, rng=random.Random(5))
    predator.direction = 0.0
    rel_directions = (
        0.0,
        -1e-9,
        1e-9,
        -math.pi / 2,
        math.pi / 2,
        -(math.pi / 2 + 1e-9),
        math.pi / 2 + 1e-9,
        math.pi,
    )
    rows = []
    for rel_dir in rel_directions:
        signals = predator.step(
            [{"type": "Agent", "distance": 100.0, "angle": 0.0, "rel_dir": rel_dir}]
        )
        rows.append(
            {
                "relative_agent_gaze": rel_dir,
                "requested_move": signals.get("move"),
                "relative_move_direction": signals.get("direction"),
                "turn": signals.get("turn", 0.0),
            }
        )
    assert math.isclose(rows[0]["relative_move_direction"], 0.0)
    assert math.isclose(abs(rows[1]["relative_move_direction"]), math.pi / 4)
    assert math.isclose(abs(rows[2]["relative_move_direction"]), math.pi / 4)
    assert rows[1]["relative_move_direction"] == -rows[2]["relative_move_direction"]
    assert math.isclose(rows[3]["relative_move_direction"], math.pi / 4)
    assert math.isclose(rows[4]["relative_move_direction"], -math.pi / 4)
    assert math.isclose(rows[5]["relative_move_direction"], 0.0)
    assert math.isclose(rows[6]["relative_move_direction"], 0.0)
    return {"distance": 100.0, "rows": rows}


def predator_spawn_probe() -> dict:
    """Confirm that an obstructed predator spawn is a silent one-shot failure."""
    env = controlled_env(seed=407)
    obstacle = Obstacle(700, 500, width=200, height=200)
    env.obstacles = [obstacle]
    env._update_obstacle_grid()
    original_spawn = Environment.spawn_predator.__get__(env, Environment)
    blocked = original_spawn(x=800, y=600)
    free = original_spawn(x=1000, y=600)
    assert blocked is None and len(env.predators) == 1 and free is env.predators[0]
    return {
        "blocked_returned_none": blocked is None,
        "free_spawn_succeeded": free is not None,
        "predator_count_after_both_calls": len(env.predators),
        "practical_scope": (
            "Generated starting and timed spawn calls can be wasted when their single random point overlaps an obstacle. "
            "This lowers realized predator pressure but is not player-controlled."
        ),
    }


def cached_gaze_fixture(mode: str, max_ticks: int = 300) -> dict:
    """Stationary lure controlled only from its public cached observations."""
    env = controlled_env(seed=405)
    agent = add_agent(env, 1000, 600, energy=150.0)
    agent.direction = math.pi
    predator = add_predator(env, 800, 600, energy=200.0, resting=False, direction=0.0)
    env.agent_observations[agent.agent_id] = agent.observe(predators=[predator], edges=list(env.edges))
    observed_ticks = rest_ticks = 0
    min_gap = math.inf
    trace = []
    for tick in range(max_ticks):
        observations = env.agent_observations.get(agent.agent_id, [])
        seen = [item for item in observations if item["type"] == "Predator"]
        if seen:
            observed_ticks += 1
            nearest = min(seen, key=lambda item: item["distance"])
            if mode == "exact":
                offset = 0.0
            elif mode == "fixed_small":
                offset = 0.02
            elif mode == "alternate_small":
                offset = 0.02 if tick % 2 == 0 else -0.02
            elif mode == "alternate_medium":
                offset = 0.2 if tick % 2 == 0 else -0.2
            else:
                raise ValueError(mode)
            turn = wrap(nearest["angle"] + offset)
        else:
            # Public-observation recovery: scan in place rather than consulting
            # hidden coordinates.
            turn = 0.2
        before_resting = predator.resting
        step_environment(env, [request(agent, turn=turn)])
        rest_ticks += int(before_resting)
        min_gap = min(min_gap, math.hypot(agent.x - predator.x, agent.y - predator.y))
        if tick < 8 or tick % 10 == 9:
            trace.append(
                {
                    "tick": tick + 1,
                    "gap": min_gap,
                    "predator_energy": predator.energy,
                    "predator_resting": predator.resting,
                    "agent_energy": agent.energy,
                    "seen": bool(seen),
                    "turn": turn,
                }
            )
        if agent.agent_id not in env.agents_dict:
            break
    return {
        "mode": mode,
        "ticks": tick + 1,
        "survived": agent.agent_id in env.agents_dict,
        "contact_or_horizon_s": (tick + 1) / 10,
        "observed_fraction": observed_ticks / (tick + 1),
        "predator_rest_fraction": rest_ticks / (tick + 1),
        "minimum_center_gap": min_gap,
        "agent_energy": agent.energy,
        "trace": trace,
    }


def cached_gaze_retreat_fixture(mode: str, max_ticks: int = 90) -> dict:
    """Ordinary walking retreat while looking near the observed predator.

    This is a bounded steering-mechanics probe, not the excluded dedicated
    sprint-decoy experiment: the founder requests only its cheap walk speed.
    """
    env = controlled_env(seed=406)
    agent = add_agent(env, 600, 600, energy=150.0)
    agent.direction = math.pi
    predator = add_predator(env, 400, 600, energy=200.0, resting=False, direction=0.0)
    env.agent_observations[agent.agent_id] = agent.observe(predators=[predator], edges=list(env.edges))
    observed_ticks = 0
    initial_gap = math.hypot(agent.x - predator.x, agent.y - predator.y)
    maximum_gap = initial_gap
    min_gap = initial_gap
    trace = []
    for tick in range(max_ticks):
        observations = env.agent_observations.get(agent.agent_id, [])
        seen = [item for item in observations if item["type"] == "Predator"]
        if seen:
            observed_ticks += 1
            nearest = min(seen, key=lambda item: item["distance"])
            if mode == "exact":
                offset = 0.0
            elif mode == "fixed_small":
                offset = 0.02
            elif mode == "alternate_small":
                offset = 0.02 if tick % 2 == 0 else -0.02
            elif mode == "alternate_medium":
                offset = 0.2 if tick % 2 == 0 else -0.2
            else:
                raise ValueError(mode)
            # Move directly away using the cached bearing, while independently
            # turning to look almost at the predator.
            move_direction = wrap(nearest["angle"] + math.pi)
            turn = wrap(nearest["angle"] + offset)
        else:
            # Continue backing away along the last-facing relationship.  This
            # uses no map coordinates or hidden predator state.
            move_direction = math.pi
            turn = 0.0
        action = ActionRequest(
            agent_id=agent.agent_id,
            move_distance=agent.speed,
            move_direction=move_direction,
            turn_angle=turn,
            spawn_agent=False,
        )
        step_environment(env, [(agent.agent_id, action)])
        gap = math.hypot(agent.x - predator.x, agent.y - predator.y)
        min_gap = min(min_gap, gap)
        maximum_gap = max(maximum_gap, gap)
        if tick < 10 or tick % 10 == 9:
            trace.append(
                {
                    "tick": tick + 1,
                    "gap": gap,
                    "agent": [agent.x, agent.y, agent.energy],
                    "predator": [predator.x, predator.y, predator.energy, predator.resting],
                    "seen": bool(seen),
                }
            )
        if agent.agent_id not in env.agents_dict:
            break
    return {
        "mode": mode,
        "ticks": tick + 1,
        "survived": agent.agent_id in env.agents_dict,
        "contact_or_horizon_s": (tick + 1) / 10,
        "initial_gap": initial_gap,
        "minimum_gap": min_gap,
        "maximum_gap": maximum_gap,
        "observed_fraction": observed_ticks / (tick + 1),
        "agent_energy": agent.energy,
        "trace": trace,
    }


def policy_actions(state: dict, tick: int, mode: str) -> list[tuple[int, ActionRequest]]:
    """One public-observation action for every returned living agent."""
    actions = []
    for status in state["observations"]:
        predators = [item for item in status["observations"] if item["type"] == "Predator"]
        if predators:
            nearest = min(predators, key=lambda item: item["distance"])
            if mode == "exact":
                offset = 0.0
            elif mode == "fixed_small":
                offset = 0.02
            elif mode == "alternate_small":
                offset = 0.02 if tick % 2 == 0 else -0.02
            else:
                raise ValueError(mode)
            turn = wrap(nearest["angle"] + offset)
            move_distance = status["speed"]
            move_direction = wrap(nearest["angle"] + math.pi)
        else:
            turn = 0.2
            move_distance = 0.0
            move_direction = 0.0
        action = ActionRequest(
            agent_id=status["agent_id"],
            move_distance=move_distance,
            move_direction=move_direction,
            turn_angle=turn,
            spawn_agent=False,
        )
        actions.append((status["agent_id"], action))
    return actions


def generated_map_run(seed: int, mode: str, max_ticks: int = 600) -> dict:
    """Generated default-size map; actions use only public response fields."""
    core = SimulationCore(starting_predators=1, seed=seed)
    env = core.env
    initial_agents = len(env.agents)
    if not env.predators:
        return {
            "seed": seed,
            "mode": mode,
            "initial_predator_spawned": False,
            "initial_agents": initial_agents,
            "skipped_reason": "The upstream one-shot starting predator placement landed on an obstacle.",
        }
    initial_predator = env.predators[0]
    # The first response fills the observation cache; still one idle action per
    # living agent.  A default-spawn predator is resting at zero energy.
    state = core.step(idle_actions(env))
    first_wake_tick = None
    visible_predator_observations = 0
    total_statuses = 0
    for tick in range(1, max_ticks):
        total_statuses += len(state["observations"])
        visible_predator_observations += sum(
            any(item["type"] == "Predator" for item in status["observations"])
            for status in state["observations"]
        )
        was_resting = initial_predator.resting
        state = core.step(policy_actions(state, tick, mode))
        if was_resting and not initial_predator.resting and first_wake_tick is None:
            # Tick numbers include the initial cache-filling step.
            first_wake_tick = tick + 1
        if not env.agents:
            break
    return {
        "seed": seed,
        "mode": mode,
        "initial_predator_spawned": True,
        "ticks": tick + 1,
        "sim_time": env.time,
        "initial_agents": initial_agents,
        "agents_alive": len(env.agents),
        "agents_lost": initial_agents - len(env.agents),
        "score": env.score,
        "initial_predator_first_wake_tick": first_wake_tick,
        "visible_predator_observation_fraction": (
            visible_predator_observations / total_statuses if total_statuses else 0.0
        ),
        "action_information": "public cached observations and tick parity only; walk away when a predator is observed",
    }


def run_all(include_generated: bool = True) -> dict:
    results = {
        "source_commit": SOURCE_COMMIT,
        "scope": {
            "engine": "unmodified vendored simulator methods",
            "synthetic": "arranged flat fixtures; random tree/predator spawning removed and labeled",
            "generated": "default-size generated maps with one explicit default-energy starting predator",
            "remote_actions": "none",
            "action_contract": "all environment steps issue at most one action per living returned agent",
        },
        "known_reproductions_and_extensions": {
            "rest_threshold": rest_threshold_probe(),
            "last_lunge_before_sleep": last_lunge_probe(),
            "overlap_pileup": pileup_probe(),
        },
        "target_selection": target_selection_probe(),
        "gaze_branch": gaze_branch_probe(),
        "predator_spawn_one_shot": predator_spawn_probe(),
        "cached_observation_gaze_fixtures": [
            cached_gaze_fixture(mode)
            for mode in ("exact", "fixed_small", "alternate_small", "alternate_medium")
        ],
        "cached_observation_walking_retreat": [
            cached_gaze_retreat_fixture(mode)
            for mode in ("exact", "fixed_small", "alternate_small", "alternate_medium")
        ],
        "rejected_or_bounded_hypotheses": [
            "Sleeping contact is not a permanent refuge: the zero-energy overlap dies on wake tick 35.",
            "A predator does not commit to a target; a one-unit nearest-distance swap changes its choice immediately.",
            "Equal-distance target ties are not a controllable agent-ID rule; they inherit set-derived observation order.",
            "Resting status and predator energy are absent from public observations, so exact wake countdown requires inferred history.",
        ],
    }
    if include_generated:
        generated = [
            generated_map_run(seed, mode)
            for seed in (*range(1, 10), 42)
            for mode in ("exact", "fixed_small", "alternate_small")
        ]
        results["generated_map_observation_only"] = generated
        successful = [row for row in generated if row.get("initial_predator_spawned")]
        results["generated_map_summary"] = {
            mode: {
                "successful_seed_runs": sum(row["mode"] == mode for row in successful),
                "total_agents_lost": sum(
                    row["agents_lost"] for row in successful if row["mode"] == mode
                ),
                "mean_score": sum(row["score"] for row in successful if row["mode"] == mode)
                / max(1, sum(row["mode"] == mode for row in successful)),
            }
            for mode in ("exact", "fixed_small", "alternate_small")
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "mechanics_hunt" / "04_predator.json",
    )
    parser.add_argument("--skip-generated", action="store_true")
    args = parser.parse_args()
    results = run_all(include_generated=not args.skip_generated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)
    print(
        json.dumps(
            {
                "rest": results["known_reproductions_and_extensions"]["rest_threshold"],
                "last_lunge": results["known_reproductions_and_extensions"]["last_lunge_before_sleep"],
                "gaze_fixtures": [
                    {key: row[key] for key in ("mode", "contact_or_horizon_s", "survived", "agent_energy")}
                    for row in results["cached_observation_gaze_fixtures"]
                ],
                "generated": results.get("generated_map_observation_only", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

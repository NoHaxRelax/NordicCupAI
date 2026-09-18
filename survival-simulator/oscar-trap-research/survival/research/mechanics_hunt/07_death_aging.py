"""Death, aging, and agent-list iteration probes for the pinned simulator.

Runs only the local vendored engine. Synthetic fixtures are explicitly labelled in
the JSON; generated-map runs do not modify engine state and issue at most one idle
action per agent returned by the preceding response.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.core import SimulationCore
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"
DT = 0.1


def fixture(seed: int = 707) -> Environment:
    """Small controlled world; positions/energies/ages are diagnostic fixtures."""
    env = Environment(400, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    return env


def add_agent(env: Environment, x: float, y: float, energy: float = 150.0):
    agent = env.spawn_agent(x=x, y=y)
    agent.direction = 0.0
    agent.energy = energy
    agent.max_age = 1e9
    return agent


def idle(agent_id: int) -> ActionRequest:
    return ActionRequest(
        agent_id=agent_id,
        move_distance=0.0,
        move_direction=0.0,
        turn_angle=0.0,
        spawn_agent=False,
    )


def walk(agent_id: int, distance: float) -> ActionRequest:
    return ActionRequest(
        agent_id=agent_id,
        move_distance=distance,
        move_direction=0.0,
        turn_angle=0.0,
        spawn_agent=False,
    )


def act(env: Environment, actions: list[ActionRequest]):
    return step_environment(env, [(a.agent_id, a) for a in actions], dt=DT)


def agent_snapshot(env: Environment, agent) -> dict:
    return {
        "id": agent.agent_id,
        "age": float(agent.age),
        "energy": float(agent.energy),
        "alive": agent.agent_id in env.agents_dict,
        "has_cached_observation": agent.agent_id in env.agent_observations,
    }


def world_summary(result: dict) -> dict:
    return {
        "score": float(result["score"]),
        "sim_time": float(result["sim_time"]),
        "num_agents": int(result["num_agents"]),
        "returned_ids": [s["agent_id"] for s in result["observations"]],
        "returned_energy": {str(s["agent_id"]): float(s["energy"]) for s in result["observations"]},
        "returned_age": {str(s["agent_id"]): float(s["age"]) for s in result["observations"]},
    }


def probe_single_removal_skip() -> dict:
    env = fixture()
    dying = add_agent(env, 100, 100, energy=0.05)
    follower = add_agent(env, 200, 100, energy=150.0)
    act(env, [idle(dying.agent_id), idle(follower.agent_id)])
    assert dying.agent_id not in env.agents_dict
    assert follower.age == 0.0 and follower.energy == 150.0
    return {
        "classification": "synthetic fixture; known bug reproduced",
        "dying": agent_snapshot(env, dying),
        "following_agent": agent_snapshot(env, follower),
        "free_tick_energy_saved": DT,
        "free_tick_age_saved": DT,
    }


def probe_alternating_skip_pattern() -> dict:
    env = fixture(708)
    agents = [
        add_agent(env, 80, 100, energy=0.05),
        add_agent(env, 140, 100, energy=150.0),
        add_agent(env, 200, 100, energy=0.05),
        add_agent(env, 260, 100, energy=150.0),
    ]
    act(env, [idle(a.agent_id) for a in agents])
    survivors = [agent_snapshot(env, a) for a in agents]
    assert [a["alive"] for a in survivors] == [False, True, False, True]
    assert agents[1].age == 0.0 and agents[3].age == 0.0
    return {
        "classification": "synthetic fixture; extension",
        "initial_order": [a.agent_id for a in agents],
        "after_one_tick": survivors,
        "finding": "each in-loop removal can skip the element shifted into its index",
    }


def probe_skipped_lifecycle_still_executes_action() -> dict:
    env = fixture(716)
    dying = add_agent(env, 100, 100, energy=0.05)
    follower = add_agent(env, 200, 100, energy=150.0)
    act(env, [idle(dying.agent_id), walk(follower.agent_id, 10.0)])
    assert math.isclose(follower.x, 210.0)
    assert math.isclose(follower.energy, 149.5)
    assert follower.age == 0.0
    return {
        "classification": "synthetic negative control",
        "follower_x": float(follower.x),
        "follower_energy": float(follower.energy),
        "follower_age": float(follower.age),
        "has_fresh_observation": follower.agent_id in env.agent_observations,
        "finding": "the action loop runs first; only lifecycle, observation, and fruit handling are skipped",
    }


def probe_death_wave_cascade() -> dict:
    env = fixture(709)
    first = add_agent(env, 80, 100, energy=0.05)
    already_negative = add_agent(env, 140, 100, energy=-0.5)
    third = add_agent(env, 200, 100, energy=150.0)
    env.agent_observations[already_negative.agent_id] = [{"diagnostic": "stale"}]

    tick1 = act(env, [idle(first.agent_id), idle(already_negative.agent_id), idle(third.agent_id)])
    snap1 = [agent_snapshot(env, a) for a in (first, already_negative, third)]
    assert not snap1[0]["alive"] and snap1[1]["alive"]
    assert already_negative.energy == -0.5 and third.age == DT

    tick2 = act(env, [idle(already_negative.agent_id), idle(third.agent_id)])
    snap2 = [agent_snapshot(env, a) for a in (already_negative, third)]
    assert not snap2[0]["alive"] and third.age == DT
    return {
        "classification": "synthetic fixture; extension",
        "after_tick_1": {"world": world_summary(tick1), "agents": snap1},
        "after_tick_2": {"world": world_summary(tick2), "agents": snap2},
        "finding": "a skipped non-positive agent survives in the response, then its next-tick removal skips its follower",
        "dead_observation_cache_retained": already_negative.agent_id in env.agent_observations,
    }


def probe_passive_death_before_fruit() -> dict:
    cases = {}
    for label, energy in (("exact_passive_cost", 0.1), ("epsilon_above", 0.100001)):
        env = fixture(710)
        agent = add_agent(env, 100, 100, energy=energy)
        fruit = env.spawn_fruit(x=100, y=100, radius=5)
        result = act(env, [idle(agent.agent_id)])
        cases[label] = {
            "start_energy": energy,
            "agent": agent_snapshot(env, agent),
            "fruit_remaining": fruit in env.fruits,
            "world": world_summary(result),
        }
    assert not cases["exact_passive_cost"]["agent"]["alive"]
    assert cases["exact_passive_cost"]["fruit_remaining"]
    assert cases["epsilon_above"]["agent"]["alive"]
    assert not cases["epsilon_above"]["fruit_remaining"]
    return {
        "classification": "synthetic fixture; extension",
        "cases": cases,
        "threshold": "on ordinary terrain, pre-step energy must be strictly greater than 0.1 to reach fruit collection",
    }


def probe_old_age_negative_and_fruit_rescue() -> dict:
    def run(with_fruit: bool):
        env = fixture(711 if with_fruit else 712)
        agent = add_agent(env, 100, 100, energy=0.5)
        agent.age = 100.0
        agent.max_age = 60.0
        fruit = env.spawn_fruit(x=100, y=100, radius=5) if with_fruit else None
        result = act(env, [idle(agent.agent_id)])
        return env, agent, fruit, result

    dry_env, dry, _, dry_result = run(False)
    wet_env, rescued, fruit, rescue_result = run(True)
    assert math.isclose(dry.energy, -0.601, abs_tol=1e-9) and dry.agent_id in dry_env.agents_dict
    assert rescued.energy > 19.0 and rescued.agent_id in wet_env.agents_dict and fruit not in wet_env.fruits

    dry_next = act(dry_env, [idle(dry.agent_id)])
    assert dry.agent_id not in dry_env.agents_dict
    return {
        "classification": "synthetic fixture; extension",
        "no_fruit_after_first_tick": {
            "agent": {"age": 100.1, "energy": -0.601, "alive": True},
            "world": world_summary(dry_result),
        },
        "no_fruit_after_second_tick": {
            "agent": agent_snapshot(dry_env, dry),
            "world": world_summary(dry_next),
        },
        "overlapping_fruit": {
            "agent": agent_snapshot(wet_env, rescued),
            "fruit_remaining": fruit in wet_env.fruits,
            "world": world_summary(rescue_result),
        },
        "finding": "old-age damage can make energy negative after the death gate; fruit collection later in the same loop can restore it",
    }


def probe_negative_energy_predation() -> dict:
    cases = {}
    for label, age, max_age in (("old_age_negative", 100.0, 60.0), ("young_positive_control", 0.0, 1e9)):
        env = fixture(713)
        agent = add_agent(env, 100, 100, energy=0.5)
        agent.age = age
        agent.max_age = max_age
        predator = env.spawn_predator(x=100, y=100)
        predator.resting = False
        predator.energy = 100.0
        result = act(env, [idle(agent.agent_id)])
        cases[label] = {
            "agent_post_lifecycle_energy": float(agent.energy),
            "agent_alive": agent.agent_id in env.agents_dict,
            "predator_energy": float(predator.energy),
            "score": float(env.score),
            "survival_time_component": float(env.time),
            "predation_score_component": float(env.score - env.time),
            "world": world_summary(result),
        }
    assert cases["old_age_negative"]["predation_score_component"] > 0
    assert cases["young_positive_control"]["predation_score_component"] < 0
    return {
        "classification": "synthetic overlap fixture; known tiny score-sign bug reproduced with control",
        "cases": cases,
        "practical_magnitude": cases["old_age_negative"]["predation_score_component"],
    }


def probe_seeded_map_old_age_rescue(seed: int = 42) -> dict:
    """Default generated map/entities, with explicit diagnostic state placement."""
    core = SimulationCore(seed=seed)
    env = core.env
    agent = env.agents[0]
    fruit = env.fruits[0]
    original = {
        "agent_position": [float(agent.x), float(agent.y)],
        "agent_max_age": float(agent.max_age),
        "fruit_position": [float(fruit.x), float(fruit.y)],
        "fruit_energy": float(fruit.energy),
    }
    # Privileged fixture assignments: realistic reachable values, artificial timing/overlap.
    agent.x, agent.y = fruit.x, fruit.y
    agent.age = agent.max_age + 1.0
    agent.energy = 0.5
    env._update_agent_grid()
    actions = [idle(a.agent_id) for a in env.agents]
    result = core.step([(a.agent_id, a) for a in actions])
    assert agent.agent_id in env.agents_dict
    assert fruit not in env.fruits and agent.energy > 0
    return {
        "classification": "default seed-42 map and real spawned fruit; artificial energy, age, and overlap",
        "seed": seed,
        "original_generated_state": original,
        "assigned_start_energy": 0.5,
        "assigned_age": float(original["agent_max_age"] + 1.0),
        "post_tick_agent": agent_snapshot(env, agent),
        "fruit_collected": fruit not in env.fruits,
        "world": world_summary(result),
    }


def probe_predation_does_not_skip_prior_lifecycle() -> dict:
    env = fixture(714)
    victim = add_agent(env, 100, 100, energy=10.0)
    follower = add_agent(env, 200, 100, energy=150.0)
    predator = env.spawn_predator(x=100, y=100)
    predator.resting = False
    predator.energy = 100.0
    act(env, [idle(victim.agent_id), idle(follower.agent_id)])
    assert victim.agent_id not in env.agents_dict
    assert math.isclose(follower.age, DT) and math.isclose(follower.energy, 149.9)
    return {
        "classification": "synthetic negative control",
        "victim": agent_snapshot(env, victim),
        "follower": agent_snapshot(env, follower),
        "finding": "predation happens after the entire agent lifecycle loop, so predator removals do not trigger the in-loop skip",
    }


def probe_terminal_tick_accounting() -> dict:
    env = fixture(715)
    only = add_agent(env, 100, 100, energy=0.05)
    result = act(env, [idle(only.agent_id)])
    assert result["num_agents"] == 0
    assert math.isclose(result["sim_time"], DT) and math.isclose(result["score"], DT)
    return {
        "classification": "synthetic fixture; neutral accounting rule",
        "world": world_summary(result),
        "finding": "the extinction step still adds dt to time and score after the final death",
    }


def natural_idle_run(seed: int, horizon: float = 200.0) -> dict:
    """Untouched default generated map; observations drive subsequent actions."""
    core = SimulationCore(seed=seed)
    response = core.step([])  # bootstrap: no agent has yet been returned to the controller
    previous = {s["agent_id"]: s for s in response["observations"]}
    skip_events = []
    old_age_onsets = {}
    death_batches = []
    action_count_mismatches = 0

    while response["num_agents"] and response["sim_time"] < horizon:
        statuses = response["observations"]
        actions = [idle(s["agent_id"]) for s in statuses]
        if len(actions) != len(statuses) or len({a.agent_id for a in actions}) != len(actions):
            action_count_mismatches += 1
        before_ids = {s["agent_id"] for s in statuses}
        response = core.step([(a.agent_id, a) for a in actions])
        current = {s["agent_id"]: s for s in response["observations"]}
        gone = sorted(before_ids - set(current))
        if gone:
            death_batches.append({"time": response["sim_time"], "ids": gone})
        for agent_id, cur in current.items():
            prev = previous.get(agent_id)
            if prev is None:
                continue
            age_delta = cur["age"] - prev["age"]
            energy_delta = cur["energy"] - prev["energy"]
            if abs(age_delta) < 1e-12:
                skip_events.append({
                    "time": response["sim_time"],
                    "id": agent_id,
                    "age": cur["age"],
                    "energy": cur["energy"],
                })
            if energy_delta < -0.2 and agent_id not in old_age_onsets:
                old_age_onsets[agent_id] = {
                    "time": response["sim_time"],
                    "reported_age": cur["age"],
                    "observed_energy_delta": energy_delta,
                }
        previous = current

    return {
        "seed": seed,
        "classification": "untouched default generated map; ordinary returned status; one idle action per returned agent",
        "end_time": response["sim_time"],
        "end_score": response["score"],
        "survivors": response["num_agents"],
        "skip_event_count": len(skip_events),
        "skip_events": skip_events,
        "old_age_onsets_inferred_from_energy_delta": old_age_onsets,
        "death_batches": death_batches,
        "action_count_mismatches": action_count_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "mechanics_hunt" / "07_death_aging.json",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 42])
    parser.add_argument("--horizon", type=float, default=200.0)
    args = parser.parse_args()

    generated_runs = [natural_idle_run(seed, args.horizon) for seed in args.seeds]
    result = {
        "source_commit": SOURCE_COMMIT,
        "scope": "local vendored engine only; no remote calls; original physics unmodified",
        "contract": "all probes issue zero or one action per listed agent; no duplicate-ID action exploit",
        "controlled_probes": {
            "single_removal_skip": probe_single_removal_skip(),
            "alternating_skip_pattern": probe_alternating_skip_pattern(),
            "skipped_lifecycle_still_executes_action": probe_skipped_lifecycle_still_executes_action(),
            "death_wave_cascade": probe_death_wave_cascade(),
            "passive_death_before_fruit": probe_passive_death_before_fruit(),
            "old_age_negative_and_fruit_rescue": probe_old_age_negative_and_fruit_rescue(),
            "negative_energy_predation": probe_negative_energy_predation(),
            "seeded_map_old_age_rescue": probe_seeded_map_old_age_rescue(),
            "predation_no_prior_lifecycle_skip": probe_predation_does_not_skip_prior_lifecycle(),
            "terminal_tick_accounting": probe_terminal_tick_accounting(),
        },
        "generated_map_idle_runs": generated_runs,
        "generated_map_summary": {
            "runs": len(generated_runs),
            "total_initial_agents": 5 * len(generated_runs),
            "total_deaths": sum(len(batch["ids"]) for run in generated_runs for batch in run["death_batches"]),
            "total_observed_skipped_updates": sum(run["skip_event_count"] for run in generated_runs),
            "runs_with_at_least_one_skip": sum(run["skip_event_count"] > 0 for run in generated_runs),
            "action_count_mismatches": sum(run["action_count_mismatches"] for run in generated_runs),
        },
        "candidate_assessment": {
            "defend_against": "budget action costs so post-action energy stays strictly above the next passive drain before relying on fruit",
            "compliant_tactical_advantage": "an old agent that passes the passive death gate can be restored by fruit after old-age damage makes its energy negative",
            "do_not_optimize": "death-list skips and negative-energy predation are real but small, situational, and not controllable enough to justify sacrifices",
        },
        "rejected_hypotheses": [
            "predator removals skip a later agent's lifecycle update",
            "a fruit can save an agent whose passive drain reaches exactly zero before collection",
            "the extinction tick omits its normal dt survival score",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

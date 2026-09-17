"""Local, bounded mechanics probes against the unmodified upstream simulator.

No network calls or competition submissions. Scenario state is configured explicitly
to isolate each rule; all submitted actions use upstream ActionRequest DTOs.
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "survival-simulator"))

from src.elements.environment import Environment
from src.elements.biome import Forest_biome
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


def fixture(seed=103):
    env = Environment(400, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    # Explicit forest fixture, no incidental initial fruit/predators/trees.
    agent = env.spawn_agent(x=100, y=200)
    agent.direction = 0.0
    agent.max_age = 1e9
    return env, agent


def action(agent_id=0, distance=0.0, direction=0.0, turn=0.0, spawn=False):
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=turn,
                         spawn_agent=spawn)


def advance(env, actions):
    return step_environment(env, [(a.agent_id, a) for a in actions])


def snapshot(env, agent):
    return {"x": float(agent.x), "y": float(agent.y), "energy": float(agent.energy),
            "age": float(agent.age), "alive": agent.agent_id in env.agents_dict,
            "score": float(env.score), "time": float(env.time),
            "population": len(env.agents)}


def probes():
    results = {}
    for label, actions in (
            ("negative_distance", [action(distance=-1000)]),
            ("idle", [action()]), ("omitted", []),
            ("single_walk", [action(distance=10)]),
            ("single_oversized_move", [action(distance=100)]),
            ("ten_duplicate_walks", [action(distance=10)] * 10)):
        env, agent = fixture()
        advance(env, actions)
        results[label] = snapshot(env, agent)
    assert results["negative_distance"] == results["idle"] == results["omitted"]
    assert results["ten_duplicate_walks"]["x"] == 200
    assert results["single_oversized_move"]["x"] == 120

    env, agent = fixture()
    agent.direction = math.pi / 2
    advance(env, [action(distance=10)])
    results["movement_is_relative_to_facing"] = snapshot(env, agent)
    assert math.isclose(agent.y, 210)
    assert math.isclose(agent.x, 100)

    env, agent = fixture()
    advance(env, [action(distance=10, direction=math.pi)])
    results["backwards_without_turning"] = {**snapshot(env, agent),
                                            "facing": float(agent.direction)}
    assert agent.x == 90 and agent.direction == 0

    env, agent = fixture()
    agent.energy = 500
    advance(env, [action(spawn=True)] * 5)
    results["five_duplicate_reproductions"] = {
        **snapshot(env, agent),
        "offspring_energies": [float(a.energy) for a in env.agents[1:]],
        "offspring_ages": [float(a.age) for a in env.agents[1:]],
    }
    assert len(env.agents) == 5  # Strictly >100 blocks the fifth request.

    env, agent = fixture()
    predicted_child_id = env._next_agent_id
    advance(env, [action(spawn=True), action(agent_id=predicted_child_id, turn=1.0)])
    child = env.agents_dict[predicted_child_id]
    results["newborn_id_can_act_same_tick"] = snapshot(env, child)
    assert math.isclose(child.energy, 75 - 1 / (2 * math.pi) - 0.1)

    env, agent = fixture()
    second = env.spawn_agent(x=200, y=200)
    second.max_age = 1e9
    agent.energy = 0.05
    advance(env, [action(), action(agent_id=second.agent_id)])
    results["death_skips_next_agent_update"] = {
        "first": snapshot(env, agent), "second": snapshot(env, second),
        "second_observation_cached": second.agent_id in env.agent_observations,
    }
    assert second.age == 0 and second.energy == 150

    env, agent = fixture()
    agent.energy = 0.5
    agent.age = 100
    agent.max_age = 60
    advance(env, [action()])
    results["old_age_negative_energy_survives_until_next_tick"] = snapshot(env, agent)
    assert agent.energy < 0 and agent.agent_id in env.agents_dict

    env, agent = fixture()
    agent.energy = 0.5
    agent.age = 100
    agent.max_age = 60
    predator = env.spawn_predator(x=100, y=200)
    predator.resting = False
    predator.energy = 100
    advance(env, [action()])
    results["negative_energy_predation_rewards_score"] = {
        **snapshot(env, agent), "predator_energy": float(predator.energy),
        "extra_score_over_time": float(env.score - env.time),
    }
    assert not env.agents and env.score > env.time

    env, agent = fixture()
    agent.x, agent.y = 35.0, 200.0
    agent.speed, agent.sprint_speed, agent.energy = 20.0, 40.0, 500.0
    env._update_agent_grid()
    advance(env, [action(distance=40, direction=math.pi)])
    results["max_sprint_boundary_tunnel"] = {
        **snapshot(env, agent),
        "inside_boundary_after_clamp": bool(env._in_obstacle(
            (agent.x, agent.y), agent.size, env.obstacles)),
    }
    assert agent.x == 5 and results["max_sprint_boundary_tunnel"]["inside_boundary_after_clamp"]

    env, agent = fixture()
    agent.x, agent.y = 195.0, 200.0
    agent.speed, agent.sprint_speed, agent.energy = 20.0, 40.0, 500.0
    env.spawn_obstacle(x=200, y=100, width=30, height=200)
    env._update_agent_grid()
    advance(env, [action(distance=40)])
    results["exact_minimum_width_wall_tunnel"] = snapshot(env, agent)
    assert agent.x == 235

    env, agent = fixture()
    fruit = env.spawn_fruit(x=110, y=200)
    advance(env, [action(distance=20)])
    results["fruit_path_crossing_not_collected"] = {
        **snapshot(env, agent), "fruit_remaining": fruit in env.fruits,
    }
    assert fruit in env.fruits  # Touch requires strict <10; path is not tested.

    env, agent = fixture()
    fruit = env.spawn_fruit(x=100, y=200)
    for _ in range(200):
        fruit.grow(0.2)
    agent.energy = 500
    advance(env, [action()])
    results["full_energy_fruit_still_scores"] = snapshot(env, agent)
    assert math.isclose(env.score, 0.16) and agent.energy == 500
    env, agent = fixture()
    predator = env.spawn_predator(x=100, y=200)
    predator.resting = True
    predator.energy = 0
    advance(env, [action()])
    results["sleeping_predator_contact"] = {
        **snapshot(env, agent), "predator_energy": predator.energy,
        "distance": math.hypot(agent.x-predator.x, agent.y-predator.y),
    }
    assert agent.agent_id in env.agents_dict and predator.energy == 3

    env, agent = fixture()
    predator = env.spawn_predator(x=140, y=200)
    predator.direction = math.pi
    predator.resting = False
    predator.energy = 100
    advance(env, [action()])
    observed = next(o for o in env.agent_observations[agent.agent_id] if o["type"] == "Predator")
    actual = math.hypot(predator.x-agent.x, predator.y-agent.y)
    results["predator_observation_is_before_its_move"] = {
        "observed_distance": observed["distance"], "actual_distance": actual,
    }
    assert observed["distance"] == 40 and math.isclose(actual, 25)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "edge_cases.json")
    args = parser.parse_args()
    result = {"source_commit": "acfc31a4003a5f91bf11032a02cd98c178ddbd7e",
              "scope": "local isolated fixtures; upstream mechanics unmodified",
              "results": probes()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

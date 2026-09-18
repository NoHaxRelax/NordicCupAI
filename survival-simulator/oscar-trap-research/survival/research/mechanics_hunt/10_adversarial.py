"""Independent adversarial mechanics verification for the survival simulator.

All gameplay calls use the unmodified vendored engine.  Synthetic fixtures isolate
ordering rules; real-map checks use the upstream SimulationCore map generator but
still arrange energy and predator proximity and are labelled accordingly.
No network or competition APIs are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.elements.fruit import Fruit
from src.elements.obstacle import Obstacle
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment

SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def controlled_env(seed: int, width: int = 400, height: int = 400) -> Environment:
    """Small deterministic world using original methods but no random spawns.

    This bypasses map generation and disables tree/predator spawning. It is an
    explicit diagnostic fixture, not a full-game result.
    """
    env = Environment.__new__(Environment)
    env.rng = random.Random(seed)
    env.width, env.height, env.chunk_size = width, height, 400
    env.agents, env.fruits, env.trees = [], [], []
    env.obstacles, env.predators = [], []
    env.edges = {
        ((0.0, 0.0), (float(width), 0.0)),
        ((float(width), 0.0), (float(width), float(height))),
        ((float(width), float(height)), (0.0, float(height))),
        ((0.0, float(height)), (0.0, 0.0)),
    }
    env.agents_dict, env.fruits_dict, env.agent_observations = {}, {}, {}
    env._next_agent_id = env._next_fruit_id = 0
    env.score = env.time = 0.0
    env.biome_map = np.empty((width, height), dtype=object)
    env.biome_map.fill(Forest_biome())
    env.grid_agents = defaultdict(set)
    env.grid_fruits = defaultdict(set)
    env.grid_trees = defaultdict(set)
    env.grid_obstacles = defaultdict(set)
    env.grid_edges = defaultdict(set)
    env.grid_predators = defaultdict(set)
    # Preserve non_agent_step while removing unrelated stochastic arrivals.
    env.spawn_tree = lambda *args, **kwargs: None
    env.spawn_predator = lambda *args, **kwargs: None
    env._update_spatial_grid()
    return env


def add_agent(env: Environment, x: float, y: float, energy: float = 150.0) -> Agent:
    agent = Agent(x, y, energy=energy, rng=env.rng, max_age=1e9)
    agent.agent_id = env._next_agent_id
    env._next_agent_id += 1
    agent.direction = 0.0
    env.agents.append(agent)
    env.agents_dict[agent.agent_id] = agent
    env._update_agent_grid()
    return agent


def add_predator(env: Environment, x: float, y: float) -> Predator:
    predator = Predator(x, y, energy=100.0, rng=env.rng)
    predator.direction = math.pi
    predator.resting = False
    env.predators.append(predator)
    env._update_predator_grid()
    return predator


def action(agent_id: int, move: float = 0.0, direction: float = 0.0,
           turn: float = 0.0, spawn: bool = False) -> ActionRequest:
    return ActionRequest(agent_id=agent_id, move_distance=move,
                         move_direction=direction, turn_angle=turn,
                         spawn_agent=spawn)


def advance(env: Environment, actions: list[ActionRequest]) -> dict:
    return step_environment(env, [(a.agent_id, a) for a in actions])


def emergency_reproduction_trial(seed: int, energy: float, mode: str) -> dict:
    """Arrange a close active predator and apply exactly one compliant action."""
    env = controlled_env(seed)
    parent = add_agent(env, 100.0, 200.0, energy=energy)
    predator = add_predator(env, 129.0, 200.0)
    if mode == "idle":
        request = action(parent.agent_id)
    elif mode == "spawn_stationary":
        request = action(parent.agent_id, spawn=True)
    elif mode == "retreat_walk":
        request = action(parent.agent_id, move=10.0, direction=math.pi)
    elif mode == "spawn_and_retreat":
        request = action(parent.agent_id, move=10.0, direction=math.pi, spawn=True)
    else:
        raise ValueError(mode)
    state = advance(env, [request])
    living_ids = [a.agent_id for a in env.agents]
    child_ids = [agent_id for agent_id in living_ids if agent_id != parent.agent_id]
    return {
        "seed": seed,
        "starting_energy": energy,
        "mode": mode,
        "parent_alive": parent.agent_id in env.agents_dict,
        "living_ids": living_ids,
        "living_children": len(child_ids),
        "population": len(living_ids),
        "score": state["score"],
        "score_effect_excluding_survival_time": state["score"] - state["sim_time"],
        "predator_energy": predator.energy,
        "parent_energy_object": parent.energy,
        "returned_child_has_observation": any(
            obs["agent_id"] in child_ids and bool(obs["observations"])
            for obs in state["observations"]
        ),
    }


def summarize_trials(rows: list[dict]) -> dict:
    return {
        "trials": len(rows),
        "species_survival_rate": sum(r["population"] > 0 for r in rows) / len(rows),
        "parent_survival_rate": sum(r["parent_alive"] for r in rows) / len(rows),
        "child_survival_rate": sum(r["living_children"] > 0 for r in rows) / len(rows),
        "mean_population": statistics.mean(r["population"] for r in rows),
        "mean_score_effect_excluding_survival_time": statistics.mean(
            r["score_effect_excluding_survival_time"] for r in rows
        ),
        "min_score_effect_excluding_survival_time": min(
            r["score_effect_excluding_survival_time"] for r in rows
        ),
        "max_score_effect_excluding_survival_time": max(
            r["score_effect_excluding_survival_time"] for r in rows
        ),
        "returned_child_observation_rate": sum(
            r["returned_child_has_observation"] for r in rows
        ) / len(rows),
    }


def emergency_reproduction_matrix(trials: int = 400) -> dict:
    energies = (100.0, 100.05, 101.0, 150.0)
    modes = ("idle", "spawn_stationary", "retreat_walk", "spawn_and_retreat")
    summaries = {}
    examples = {}
    for energy in energies:
        for mode in modes:
            rows = [emergency_reproduction_trial(seed, energy, mode)
                    for seed in range(trials)]
            key = f"energy={energy:g}|{mode}"
            summaries[key] = summarize_trials(rows)
            examples[key] = rows[0]

    # Required adversarial controls.
    assert summaries["energy=100|spawn_stationary"]["child_survival_rate"] == 0
    assert summaries["energy=100.05|spawn_stationary"]["child_survival_rate"] > 0
    assert summaries["energy=100.05|spawn_stationary"][
        "mean_score_effect_excluding_survival_time"
    ] > summaries["energy=100.05|idle"]["mean_score_effect_excluding_survival_time"]
    assert summaries["energy=100.05|spawn_stationary"]["child_survival_rate"] < 1
    return {
        "fixture": (
            "Synthetic forest; parent at (100,200), active predator at (129,200); "
            "unrelated stochastic tree/predator spawning disabled."
        ),
        "ordinary_action_contract": "one ActionRequest for the only observed parent",
        "artificial_inputs": "arranged geometry and assigned parent energy",
        "summaries": summaries,
        "seed_zero_examples": examples,
    }


def real_map_emergency_checks(seeds: tuple[int, ...] = (1, 2, 3, 42)) -> list[dict]:
    """Repeat the interaction on upstream generated maps.

    Energy and predator placement remain arranged. Other initial agents are removed
    so lineage survival is attributable to this interaction.
    """
    rows = []
    for seed in seeds:
        for mode in ("idle", "spawn_stationary"):
            sim = SimulationCore(starting_predators=0, seed=seed)
            env = sim.env
            parent = env.agents[0]
            for other in list(env.agents[1:]):
                env.kill_agent(other)
            parent.energy = 100.05
            parent.max_age = 1e9
            parent.direction = 0.0
            parent_start = [float(parent.x), float(parent.y)]
            initial_biome = env.biome_map[int(parent.x), int(parent.y)].type
            predator = None
            placement_angle = None
            for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                x = parent.x + 29 * math.cos(angle)
                y = parent.y + 29 * math.sin(angle)
                if 20 <= x < env.width - 20 and 20 <= y < env.height - 20:
                    candidate = env.spawn_predator(x=x, y=y)
                    if candidate is not None:
                        predator = candidate
                        placement_angle = float(angle)
                        break
            if predator is None:
                rows.append({"seed": seed, "mode": mode, "placement_failed": True})
                continue
            predator.direction = (placement_angle + math.pi) % (2 * math.pi)
            predator.resting = False
            predator.energy = 100.0
            predator_start = [float(predator.x), float(predator.y)]
            request = action(parent.agent_id, spawn=(mode == "spawn_stationary"))
            state = advance(env, [request])
            child_ids = [a.agent_id for a in env.agents if a.agent_id != parent.agent_id]
            rows.append({
                "seed": seed,
                "mode": mode,
                "placement_failed": False,
                "parent_start": parent_start,
                "predator_start": predator_start,
                "initial_biome": initial_biome,
                "placement_angle": placement_angle,
                "parent_alive": parent.agent_id in env.agents_dict,
                "living_children": len(child_ids),
                "population": len(env.agents),
                "score_effect_excluding_survival_time": state["score"] - state["sim_time"],
            })
    assert all(not r.get("placement_failed") for r in rows)
    return rows


def stale_observation_probes() -> dict:
    # Fruit is observed before collection, so the response contains a fruit that
    # has already been removed from the world.
    env = controlled_env(10)
    agent = add_agent(env, 100.0, 200.0)
    fruit = Fruit(100.0, 200.0, radius=5.0)
    fruit.fruit_id = 0
    env.fruits.append(fruit)
    env.fruits_dict[0] = fruit
    env._update_fruit_grid()
    first = advance(env, [action(agent.agent_id)])
    first_obs = first["observations"][0]["observations"]
    second = advance(env, [action(agent.agent_id)])
    second_obs = second["observations"][0]["observations"]
    fruit_ghost = {
        "fruit_in_first_response": any(o["type"] == "Fruit" for o in first_obs),
        "fruit_exists_after_first_step": bool(env.fruits),
        "fruit_in_second_response": any(o["type"] == "Fruit" for o in second_obs),
    }
    assert fruit_ghost == {
        "fruit_in_first_response": True,
        "fruit_exists_after_first_step": False,
        "fruit_in_second_response": False,
    }

    # A surviving observer can receive another agent in its cached observation
    # even though that agent is killed later in the same world step.
    env = controlled_env(11)
    observer = add_agent(env, 100.0, 200.0)
    victim = add_agent(env, 140.0, 200.0)
    predator = add_predator(env, 169.0, 200.0)
    result = advance(env, [action(observer.agent_id), action(victim.agent_id)])
    observer_state = next(o for o in result["observations"]
                          if o["agent_id"] == observer.agent_id)
    observed_agent_ids = [o["id"] for o in observer_state["observations"]
                          if o["type"] == "Agent"]
    death_ghost = {
        "victim_id": victim.agent_id,
        "victim_alive_after_step": victim.agent_id in env.agents_dict,
        "victim_in_observer_response": victim.agent_id in observed_agent_ids,
        "returned_agent_ids": [o["agent_id"] for o in result["observations"]],
        "predator_post_step": [predator.x, predator.y],
    }
    assert death_ghost["victim_in_observer_response"]
    assert not death_ghost["victim_alive_after_step"]
    return {"consumed_fruit_ghost": fruit_ghost, "dead_agent_ghost": death_ghost}


def collision_combination_controls() -> dict:
    """Check that duplicate short moves do not inherit single-step tunnelling."""
    def case(agent_speed: float, sprint_speed: float, requests: list[float]) -> dict:
        env = controlled_env(20)
        wall = Obstacle(200.0, 100.0, width=30.0, height=200.0)
        env.obstacles.append(wall)
        env.edges.update(wall.edges)
        env._update_obstacle_grid()
        env._update_edge_grid()
        agent = add_agent(env, 195.0, 200.0, energy=500.0)
        agent.speed, agent.sprint_speed = agent_speed, sprint_speed
        result = advance(env, [action(agent.agent_id, move=d) for d in requests])
        return {
            "requests": requests,
            "speed": agent_speed,
            "sprint_speed": sprint_speed,
            "end": [agent.x, agent.y],
            "crossed_to_far_side": agent.x >= 235.0,
            "energy": agent.energy,
            "score": result["score"],
        }

    evolved_single = case(20.0, 40.0, [40.0])
    evolved_four_short = case(20.0, 40.0, [10.0] * 4)
    founder_oversized = case(10.0, 20.0, [40.0])
    assert evolved_single["crossed_to_far_side"]
    assert not evolved_four_short["crossed_to_far_side"]
    assert not founder_oversized["crossed_to_far_side"]
    return {
        "known_control_evolved_single_sprint": evolved_single,
        "negative_control_duplicate_short_walks": evolved_four_short,
        "negative_control_founder_oversized_request": founder_oversized,
        "contract": {
            "single_sprint": "one action, but requires evolved sprint_speed=40 fixture",
            "duplicate_short_walks": "violates one-action-per-agent documentation",
            "founder_oversized": "one action; capped to founder sprint_speed",
        },
    }


def malformed_input_control() -> dict:
    """Verify non-finite DTO acceptance, then reject it as an actionable tactic."""
    parsed = ActionRequest(agent_id=0, move_distance=float("nan"),
                           move_direction=0.0, turn_angle=0.0,
                           spawn_agent=False)
    standard_json_rejected = False
    try:
        json.dumps(parsed.model_dump(), allow_nan=False)
    except (ValueError, TypeError):
        standard_json_rejected = True
    return {
        "pydantic_accepts_nan_locally": math.isnan(parsed.move_distance),
        "strict_json_serialization_rejects_nan": standard_json_rejected,
        "classification": (
            "Malformed non-standard numeric input, outside the documented contract; "
            "not advanced into the engine because state contamination/crash is not a strategy advantage."
        ),
    }


def run_all(trials: int) -> dict:
    return {
        "source_commit": SOURCE_COMMIT,
        "scope": "local adversarial verification against unmodified engine methods",
        "fixture_disclosures": [
            "Synthetic probes disable unrelated random tree and predator spawning.",
            "Emergency probes arrange predator proximity and assign parent energy.",
            "Real-map probes preserve generated terrain/obstacles but still arrange the encounter.",
            "No controller uses engine handles or hidden predator energy in its action decision.",
        ],
        "emergency_reproduction": emergency_reproduction_matrix(trials),
        "real_map_emergency_checks": real_map_emergency_checks(),
        "stale_observations": stale_observation_probes(),
        "collision_combinations": collision_combination_controls(),
        "malformed_input": malformed_input_control(),
        "hypotheses": {
            "supported": [
                "A threatened parent just above 100 energy can reproduce before passive death/predation.",
                "Passive death itself has no score penalty, while predation subtracts remaining energy / 100.",
                "A consumed fruit and an agent killed later in the tick can remain in the returned cached observation.",
                "A single evolved 40-unit move can tunnel a 30-unit wall in the controlled geometry.",
            ],
            "rejected_or_bounded": [
                "Emergency reproduction does not make the newborn immune; the predator may eat it in the same tick.",
                "Duplicate founder-sized short moves do not combine into wall tunnelling because collision is checked after each action.",
                "A founder oversized move does not tunnel the tested wall because it is capped at sprint_speed=20.",
                "NaN acceptance by the local DTO is not a standard-JSON, contract-compliant gameplay tactic.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "mechanics_hunt" / "10_adversarial.json",
    )
    args = parser.parse_args()
    results = run_all(args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=json_default) + "\n")
    print(args.output)
    print(json.dumps({
        "emergency_reproduction": results["emergency_reproduction"]["summaries"],
        "real_map_emergency_checks": results["real_map_emergency_checks"],
        "stale_observations": results["stale_observations"],
        "collision_combinations": results["collision_combinations"],
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()

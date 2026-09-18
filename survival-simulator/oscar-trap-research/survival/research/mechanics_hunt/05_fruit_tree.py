"""Fruit and tree mechanics probes against the unmodified vendored simulator.

The controlled fixtures deliberately arrange positions and, where stated in the
JSON, disable unrelated stochastic spawning.  The generated-map survey uses the
normal map, obstacle, tree, fruit, predator-spawn, growth and death code, but has
no agents, so its fruit stock is an upper bound with consumption removed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

SURVIVAL = Path(__file__).resolve().parents[2]
VENDOR = SURVIVAL / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.elements.agent import Agent
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.elements.fruit import Fruit
from src.elements.tree import Tree
from src.utils.DTOs import ActionRequest
from src.utils.simulation import create_environment, step_environment


SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"


def action(agent_id: int, distance: float = 0.0, direction: float = 0.0,
           turn: float = 0.0, spawn: bool = False) -> ActionRequest:
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=turn,
                         spawn_agent=spawn)


def fixture(seed: int = 505, with_agent: bool = True):
    env = Environment(400, 400, 400, random.Random(seed))
    forest = Forest_biome()
    env.biome_map.fill(forest)
    # Prevent unrelated global tree/predator creation in isolated probes. This
    # does not replace growth, collection, scoring, rot, or tree-death methods.
    env.spawn_tree = lambda *args, **kwargs: None
    env.spawn_predator = lambda *args, **kwargs: None
    agent = None
    if with_agent:
        agent = env.spawn_agent(x=100, y=200)
        agent.direction = 0.0
        agent.max_age = 1e9
    return env, agent


def quantiles(values):
    ordered = sorted(values)
    def q(p):
        return ordered[round((len(ordered) - 1) * p)]
    return {
        "n": len(values),
        "min": min(values),
        "p10": q(0.1),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p90": q(0.9),
        "max": max(values),
    }


def known_fruit_timing():
    fruit = Fruit(0, 0, radius=5)
    timeline = []
    removed_at = None
    for tick in range(700):
        if tick in (0, 1, 100, 199, 200, 201, 499, 500, 501, 502):
            timeline.append({
                "tick": tick,
                "sim_seconds_before_tick": tick / 10,
                "internal_age": fruit.age,
                "energy": fruit.energy,
                "radius": fruit.radius,
            })
        if fruit.age > 100:
            removed_at = tick / 10
            break
        fruit.grow(0.2)

    observer = Agent(100, 200, rng=random.Random(1))
    observer.direction = 0.0
    fresh = Fruit(130, 200, radius=5)
    ripe = Fruit(130, 200, radius=5)
    for _ in range(200):
        ripe.grow(0.2)
    edges = [((0, 0), (400, 0)), ((400, 0), (400, 400)),
             ((400, 400), (0, 400)), ((0, 400), (0, 0))]
    fresh_obs = observer.observe(fruits=[fresh], edges=edges)
    ripe_obs = observer.observe(fruits=[ripe], edges=edges)
    assert fresh_obs == ripe_obs

    return {
        "timeline": timeline,
        "removed_at_sim_seconds_in_source_order": removed_at,
        "fresh_and_ripe_observations_identical": fresh_obs == ripe_obs,
        "observation_payload": fresh_obs,
        "classification": "known result reproduced; standalone source Fruit plus ordinary observation method",
    }


def scoring_and_same_tick_observation():
    env, agent = fixture()
    fruit = env.spawn_fruit(x=100, y=200, radius=5)
    for _ in range(200):
        fruit.grow(0.2)
    agent.energy = agent.max_energy
    state = step_environment(env, [(agent.agent_id, action(agent.agent_id))])
    first_obs = state["observations"][0]["observations"]
    assert not env.fruits
    assert any(item["type"] == "Fruit" for item in first_obs)
    assert math.isclose(agent.energy, agent.max_energy)
    fruit_bonus = state["score"] - state["sim_time"]
    energy_after_collection = agent.energy
    assert math.isclose(fruit_bonus, fruit.energy / 1000)

    state2 = step_environment(env, [(agent.agent_id, action(agent.agent_id))])
    second_obs = state2["observations"][0]["observations"]
    assert not any(item["type"] == "Fruit" for item in second_obs)
    return {
        "fruit_energy": fruit.energy,
        "fruit_bonus_while_storage_full": fruit_bonus,
        "energy_after_collection": energy_after_collection,
        "collector_response_still_contains_consumed_fruit": True,
        "next_response_contains_fruit": any(item["type"] == "Fruit" for item in second_obs),
        "practical_magnitude": "one stale response can cause one wasted follow-up action; ripe bonus is about 0.06",
        "classification": "contract-compliant; controller sees the stale payload through ordinary observations",
    }


def tree_same_tick_observation():
    env, agent = fixture(seed=506)
    tree = Tree(120, 200)
    tree.age = 100.0
    tree.radius = 20.0
    env.trees = [tree]
    env._update_tree_grid()
    state = step_environment(env, [(agent.agent_id, action(agent.agent_id))])
    first_obs = state["observations"][0]["observations"]
    assert tree not in env.trees
    assert any(item["type"] == "Tree" for item in first_obs)
    state2 = step_environment(env, [(agent.agent_id, action(agent.agent_id))])
    second_obs = state2["observations"][0]["observations"]
    assert not any(item["type"] == "Tree" for item in second_obs)
    return {
        "response_contains_tree_removed_later_in_same_tick": True,
        "next_response_contains_tree": False,
        "classification": "arranged old-tree fixture; stale payload is available through ordinary observations",
    }


def run_fruit_choice(initial_internal_age: float, initial_energy: float,
                     wait_ticks: int, collect_immediately: bool):
    env, agent = fixture(seed=507)
    agent.x = 100.0
    agent.y = 200.0
    agent.energy = initial_energy
    agent.max_energy = 500.0
    env._update_agent_grid()
    fruit = env.spawn_fruit(x=120, y=200, radius=5)
    if initial_internal_age:
        # Source growth has equal increments for internal age and pre-ripe energy.
        remaining = initial_internal_age
        while remaining > 1e-12:
            amount = min(0.2, remaining)
            fruit.grow(amount)
            remaining -= amount

    collected_at = None
    alive_at_end = True
    total_ticks = wait_ticks + 1
    previous_bonus = env.score - env.time
    for tick in range(total_ticks):
        if agent.agent_id not in env.agents_dict:
            alive_at_end = False
            break
        should_collect = tick == 0 if collect_immediately else tick == wait_ticks
        request = action(agent.agent_id, distance=20.0 if should_collect else 0.0)
        step_environment(env, [(agent.agent_id, request)])
        current_bonus = env.score - env.time
        if current_bonus > previous_bonus + 1e-9 and collected_at is None:
            collected_at = env.time
        previous_bonus = current_bonus
    return {
        "alive": agent.agent_id in env.agents_dict,
        "energy": agent.energy if agent.agent_id in env.agents_dict else None,
        "score": env.score,
        "fruit_collected_at": collected_at,
        "fruit_remaining": fruit in env.fruits,
        "sim_time": env.time,
    }


def fruit_waiting_tradeoff():
    fresh_immediate = run_fruit_choice(0.0, 150.0, 200, True)
    fresh_wait = run_fruit_choice(0.0, 150.0, 200, False)
    old_immediate = run_fruit_choice(80.0, 150.0, 200, True)
    old_wait = run_fruit_choice(80.0, 150.0, 200, False)

    # Low-energy negative control: a greedy agent can reach a fresh fruit in two
    # walking steps, whereas standing off for maturation starves first.
    env, agent = fixture(seed=508)
    agent.energy = 18.0
    fruit = env.spawn_fruit(x=120, y=200, radius=5)
    for _ in range(201):
        if agent.agent_id not in env.agents_dict:
            break
        step_environment(env, [(agent.agent_id, action(agent.agent_id, distance=10.0 if env.time < 0.2 else 0.0))])
    low_energy_greedy = {
        "alive": agent.agent_id in env.agents_dict,
        "energy": agent.energy if agent.agent_id in env.agents_dict else None,
        "fruit_collected": fruit not in env.fruits,
        "sim_time": env.time,
    }
    low_energy_wait = run_fruit_choice(0.0, 18.0, 200, False)

    assert fresh_wait["energy"] - fresh_immediate["energy"] > 39.9
    assert old_immediate["fruit_collected_at"] is not None
    assert old_wait["fruit_collected_at"] is None
    assert low_energy_greedy["alive"] and not low_energy_wait["alive"]
    return {
        "known_fresh_fruit": {"immediate": fresh_immediate, "wait_20_seconds": fresh_wait},
        "first_seen_but_already_40_seconds_old": {"immediate": old_immediate, "blind_wait_20_seconds": old_wait},
        "low_energy_fresh_fruit": {"move_to_food": low_energy_greedy, "blind_wait": low_energy_wait},
        "practical_magnitude": "when birth time is known and energy is safe, 20 seconds adds about 40 energy and 0.04 direct score; blind waiting can lose all 60 energy or kill the agent",
        "ordinary_observation_boundary": "fruit age is hidden; known birth time requires continuous coverage of the patch before first sighting. Fixture setup knows age, but the proposed gate uses only first-seen history and current agent energy.",
        "contract": "one action per observed living agent per tick",
    }


def death_age_trials(trials: int, dt: float):
    env, _ = fixture(seed=600, with_agent=False)
    forest = Forest_biome()
    forest.fruit_spawn_rate = 0.0
    env.biome_map.fill(forest)
    ages = []
    for trial in range(trials):
        env.rng = random.Random(600_000 + trial)
        env.time = 0.0
        env.score = 0.0
        env.agents = []
        env.agents_dict = {}
        env.fruits = []
        env.fruits_dict = {}
        env.predators = []
        tree = Tree(200, 200)
        env.trees = [tree]
        env._update_spatial_grid()
        while tree in env.trees and env.time < 120:
            env.non_agent_step(dt)
        assert tree not in env.trees
        ages.append(tree.age)
    result = quantiles(ages)
    result["dt"] = dt
    return result


def tree_lifecycle(trials: int):
    by_dt = {str(dt): death_age_trials(trials, dt) for dt in (0.05, 0.1, 0.2)}

    # Exact engine loop with one new forest tree. Global tree/predator spawning is
    # disabled, but its random checks still occur; fruit creation/growth/rot remain.
    env, _ = fixture(seed=700, with_agent=False)
    yields = []
    for trial in range(trials):
        env.rng = random.Random(700_000 + trial)
        env.time = 0.0
        env.score = 0.0
        env.fruits = []
        env.fruits_dict = {}
        env._next_fruit_id = 0
        env.predators = []
        tree = Tree(200, 200)
        env.trees = [tree]
        env._update_spatial_grid()
        while tree in env.trees and env.time < 120:
            env.non_agent_step(0.1)
        yields.append(env._next_fruit_id)

    return {
        "death_age_seconds": by_dt,
        "fruits_created_per_new_forest_tree_at_official_local_dt_0.1": quantiles(yields),
        "finding": "death threshold is resampled every tick, so a new tree usually dies in the mid-50s rather than receiving one fixed 50-100 second lifetime; death age is dt-sensitive",
        "classification": "engine-executed isolated fixture; unrelated global spawning disabled; no agents or fruit consumption",
    }


def batch_rot_skip():
    env, _ = fixture(seed=800, with_agent=False)
    for i in range(8):
        fruit = env.spawn_fruit(x=100 + i * 20, y=200, radius=5)
        fruit.age = 100.2
        fruit.energy = 60.0
        fruit.radius = 9.0
    counts = [len(env.fruits)]
    for _ in range(4):
        env.non_agent_step(0.1)
        counts.append(len(env.fruits))
    assert counts == [8, 4, 2, 1, 0]
    return {
        "fruit_counts_by_tick": counts,
        "extra_lifetime_for_last_item_seconds": 0.3,
        "classification": "fixture-only synchronized rot batch; list-removal skip, negligible natural strategic value",
    }


def stacked_fruit_collection():
    env, agent = fixture(seed=850)
    agent.energy = agent.max_energy
    fruits = []
    for _ in range(5):
        fruit = env.spawn_fruit(x=100, y=200, radius=5)
        for _ in range(200):
            fruit.grow(0.2)
        fruits.append(fruit)
    state = step_environment(env, [(agent.agent_id, action(agent.agent_id))])
    fruit_observations = [item for item in state["observations"][0]["observations"]
                          if item["type"] == "Fruit"]
    bonus = state["score"] - state["sim_time"]
    assert len(fruit_observations) == 5 and not env.fruits
    assert math.isclose(bonus, sum(f.energy for f in fruits) / 1000)
    return {
        "ripe_fruits_collected_in_one_tick": len(fruits),
        "fruit_observations_in_returned_stale_payload": len(fruit_observations),
        "direct_score_bonus": bonus,
        "classification": "synthetic co-located fixture; contract-compliant automatic collection, while exact co-location is unlikely naturally",
    }


def blocked_tree_spawn():
    env = Environment(400, 400, 400, random.Random(900))
    env.biome_map.fill(Forest_biome())
    blocker = env.spawn_obstacle(x=80, y=180, width=80, height=80)
    assert blocker is not None
    tree = env.spawn_tree(x=100, y=200, max_attempts=0)
    fruit = env.spawn_fruit(x=100, y=200, radius=5, max_attempts=0)
    assert tree is not None and fruit is None
    return {
        "tree_created_inside_obstacle_when_no_attempts": tree is not None,
        "fruit_created_inside_same_obstacle": fruit is not None,
        "classification": "synthetic max_attempts=0 fixture; normal calls retry 50 times, so not a practical tactic",
    }


def cluster_metrics(trees):
    if not trees:
        return {"pairs_within_25": 0, "max_neighbors_within_50_including_self": 0}
    pairs25 = 0
    max50 = 1
    for i, first in enumerate(trees):
        neighbors = 1
        for j, second in enumerate(trees):
            if i == j:
                continue
            distance = math.hypot(first.x - second.x, first.y - second.y)
            if j > i and distance < 25:
                pairs25 += 1
            if distance < 50:
                neighbors += 1
        max50 = max(max50, neighbors)
    return {"pairs_within_25": pairs25,
            "max_neighbors_within_50_including_self": max50}


def fruit_cluster_metrics(fruits):
    if not fruits:
        return {"fruit_pairs_within_10": 0,
                "max_fruits_within_15_of_one_fruit_including_self": 0}
    pairs10 = 0
    max15 = 1
    for i, first in enumerate(fruits):
        neighbors = 1
        for j, second in enumerate(fruits):
            if i == j:
                continue
            distance = math.hypot(first.x - second.x, first.y - second.y)
            if j > i and distance < 10:
                pairs10 += 1
            if distance < 15:
                neighbors += 1
        max15 = max(max15, neighbors)
    return {"fruit_pairs_within_10": pairs10,
            "max_fruits_within_15_of_one_fruit_including_self": max15}


def generated_map_survey(seeds, horizon: float):
    runs = []
    sample_times = [0, 30, 60, 120, 300, horizon]
    sample_ticks = {int(t / 0.1) for t in sample_times if t <= horizon}
    for seed in seeds:
        rng = random.Random(seed)
        env = create_environment(1600, 1200, 400, starting_agents=0,
                                 starting_predators=0, starting_fruits=32,
                                 starting_trees=50, rng=rng)
        known_trees = set(env.trees)
        known_fruits = set(env.fruits)
        initial_tree_ages = [tree.age for tree in env.trees]
        tree_births = tree_deaths = fruit_births = fruit_removals = 0
        samples = []

        def take_sample(tick):
            samples.append({
                "time": round(tick * 0.1, 1),
                "trees": len(env.trees),
                "mature_trees_age_at_least_20": sum(t.age >= 20 for t in env.trees),
                "fruits_in_world": len(env.fruits),
                **cluster_metrics(env.trees),
                **fruit_cluster_metrics(env.fruits),
            })

        take_sample(0)
        total_ticks = int(horizon / 0.1)
        for tick in range(1, total_ticks + 1):
            before_trees = set(env.trees)
            before_fruits = set(env.fruits)
            env.non_agent_step(0.1)
            after_trees = set(env.trees)
            after_fruits = set(env.fruits)
            tree_births += len(after_trees - before_trees)
            tree_deaths += len(before_trees - after_trees)
            fruit_births += len(after_fruits - before_fruits)
            fruit_removals += len(before_fruits - after_fruits)
            known_trees |= after_trees
            known_fruits |= after_fruits
            if tick in sample_ticks:
                take_sample(tick)

        runs.append({
            "seed": seed,
            "initial_tree_attempts": 50,
            "initial_trees_created": samples[0]["trees"],
            "initial_tree_age_range": [
                min(initial_tree_ages, default=None),
                max(initial_tree_ages, default=None),
            ],
            "tree_births": tree_births,
            "tree_deaths": tree_deaths,
            "fruit_births": fruit_births,
            "fruit_removals_without_consumers": fruit_removals,
            "samples": samples,
        })
    return {
        "seeds": list(seeds),
        "horizon": horizon,
        "runs": runs,
        "classification": "unmodified generated maps and normal stochastic entity logic; agents omitted, so fruit stock is an upper bound and no policy advantage is claimed",
    }


def derived_spawn_attempt_table():
    rows = []
    for time_s in (0, 60, 300, 600):
        for tree_count in (1, 10, 50, 100):
            probability = (100 / max(1, tree_count / 2)) * 0.1 * (0.5 ** (time_s / 300))
            rows.append({"time": time_s, "trees": tree_count,
                         "raw_probability_before_biome_acceptance": probability,
                         "effective_attempt_probability": min(1.0, probability)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--output", type=Path,
                        default=SURVIVAL / "results" / "mechanics_hunt" / "05_fruit_tree.json")
    args = parser.parse_args()

    result = {
        "source_commit": SOURCE_COMMIT,
        "scope": "local-only original engine mechanics; fixtures and removed systems explicitly labelled",
        "known_fruit_timing": known_fruit_timing(),
        "full_energy_scoring_and_stale_fruit_observation": scoring_and_same_tick_observation(),
        "stale_tree_observation": tree_same_tick_observation(),
        "fruit_waiting_tradeoff": fruit_waiting_tradeoff(),
        "tree_lifecycle": tree_lifecycle(args.trials),
        "synchronized_rot_list_skip": batch_rot_skip(),
        "stacked_fruit_collection": stacked_fruit_collection(),
        "blocked_tree_spawn_negative_control": blocked_tree_spawn(),
        "tree_spawn_attempt_formula": derived_spawn_attempt_table(),
        "generated_map_survey": generated_map_survey(args.seeds, args.horizon),
        "contract": "all controller probes issue at most one action per observed living agent per tick",
        "privileged_state_boundary": "ages, object identity, exact positions and list membership are diagnostics only; ordinary policies receive type/distance/angle without fruit or tree IDs/ages",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Bounded source-mechanics experiments, not an end-to-end policy benchmark.

Run with the project's survival virtualenv. Genetic search deliberately grants an
immortal best parent and unlimited food: its birth counts are optimistic, not a
claim that an actual game can pay for or survive the required births.
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

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "survival/vendor/survival-simulator"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(VENDOR))


def quantile(values, probability):
    return sorted(values)[int((len(values) - 1) * probability)]


def mutation_birth_budget(trials=10000, max_births=500):
    """Two exact source trait distributions; other traits/resources excluded."""
    rng = random.Random(9172026)
    budgets = (10, 20, 40, 80, 160)
    thresholds = (15.0, 16.0, 19.999999)
    arrival = {str(t): [] for t in thresholds}
    for _ in range(trials):
        walk, sprint = 10.0, 20.0
        hit = {}
        for birth in range(1, max_births + 1):
            w = min(20.0, walk * rng.uniform(0.5, 1.5)) if rng.random() < 0.1 else walk
            s = min(40.0, sprint * rng.uniform(0.5, 1.5)) if rng.random() < 0.1 else sprint
            # Retain best cheap movement, then prefer greater sprint reserve.
            if (min(w, s), s) > (min(walk, sprint), sprint):
                walk, sprint = w, s
            for threshold in thresholds:
                if threshold not in hit and min(walk, sprint) > threshold:
                    hit[threshold] = birth
            if len(hit) == len(thresholds):
                break
        for threshold in thresholds:
            arrival[str(threshold)].append(hit.get(threshold, max_births + 1))
    results = {}
    for threshold, samples in arrival.items():
        results[threshold] = {
            "median_births": statistics.median(samples),
            "p10_births": quantile(samples, 0.1),
            "p90_births": quantile(samples, 0.9),
            "mean_births_censored": statistics.mean(samples),
            "fraction_success_by_birth_budget": {
                str(n): sum(x <= n for x in samples) / trials for n in budgets
            },
            "not_reached_by_max_births": sum(x > max_births for x in samples),
        }
    return {"trials": trials, "seed": 9172026, "max_births": max_births,
            "assumption": "Best parent immortal, unlimited births and food; no ecological validation.",
            "effective_walk_speed_greater_than": results}


def source_probes():
    import numpy as np
    from src.elements.agent import Agent
    from src.elements.biome import Forest_biome, River_biome
    from src.elements.environment import Environment
    from src.elements.fruit import Fruit

    env = Environment.__new__(Environment)
    env.width, env.height = 2000, 100
    env.biome_map = np.full((env.width, env.height), Forest_biome(), dtype=object)

    movement = []
    for label, walk, sprint, energy, request, river in [
        ("baseline_walk", 10, 20, 150, 10, False),
        ("baseline_just_faster_than_predator", 10, 20, 150, 15.1, False),
        ("baseline_full_sprint", 10, 20, 150, 20, False),
        ("low_energy_sprint_request", 10, 20, 75, 20, False),
        ("selected_just_faster_than_predator", 16, 20, 150, 15.1, False),
        ("walk_greater_than_sprint", 20, 12, 150, 20, False),
        ("river_full_sprint", 10, 20, 150, 20, True),
    ]:
        env.biome_map[:] = River_biome() if river else Forest_biome()
        a = Agent(100, 50, speed=walk, sprint_speed=sprint, energy=energy, rng=random.Random(1))
        a.direction = 0
        env.update_entity_position(a, request, 0, [])
        movement.append({"case": label, "distance_actual": a.x - 100,
                         "movement_energy": energy - a.energy,
                         "total_energy_per_second_including_young_idle": 10 * (energy - a.energy) + 1})

    fruit = Fruit(110, 50, radius=5)
    fruit_points = []
    # Reproduce source order: rot check, then grow(2*dt). Use source grow method.
    removed_at = None
    for tick in range(601):
        if tick in (0, 100, 200, 300, 500):
            fruit_points.append({"sim_seconds": tick / 10,
                                 "internal_age": fruit.age, "energy": fruit.energy,
                                 "radius": fruit.radius})
        if fruit.age > 100:
            removed_at = tick / 10
            break
        fruit.grow(0.2)
    a = Agent(100, 50, rng=random.Random(1))
    a.direction = 0
    fresh, ripe = Fruit(130, 50), Fruit(130, 50)
    for _ in range(200):
        ripe.grow(0.2)
    edges = [((0, 0), (2000, 0)), ((2000, 0), (2000, 100)),
             ((2000, 100), (0, 100)), ((0, 100), (0, 0))]
    obs_fresh = a.observe(fruits=[fresh], edges=edges)
    obs_ripe = a.observe(fruits=[ripe], edges=edges)
    assert obs_fresh == obs_ripe, (obs_fresh, obs_ripe)
    return {"movement": movement,
            "fruit_timeline": fruit_points,
            "fruit_removed_at_sim_seconds_by_source_loop": removed_at,
            "fresh_and_ripe_observations_identical": obs_fresh == obs_ripe,
            "fruit_observation": [o for o in obs_fresh if o["type"] == "Fruit"]}


def age_cost_table():
    # Source adds 0.01*age each dt=.1 after max_age, in addition to dt base drain.
    return [{"age_seconds": age, "total_passive_energy_per_second_when_old": 1 + 0.1 * age,
             "mature_fruits_per_minute_to_cover_idle": (1 + 0.1 * age)}
            for age in (60, 90, 100, 120, 200)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "survival/results/speed_and_fruit.json")
    parser.add_argument("--without-source", action="store_true")
    args = parser.parse_args()
    results = {"kind": "controlled mechanics and optimistic genetic model",
               "mutation": mutation_birth_budget(args.trials), "old_agent_costs": age_cost_table()}
    if not args.without_source:
        results["source_probes"] = source_probes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

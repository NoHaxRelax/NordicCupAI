"""Reproduction, inheritance, energy, and placement probes.

All engine calls use the exact vendored simulator.  Privileged fixtures are
labelled in the JSON; the generated-map check uses one ordinary action per
currently observed agent and default map dimensions/seeds.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR = REPO_ROOT / "survival" / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.core import SimulationCore
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"
TRAITS = ("speed", "sprint_speed", "max_energy", "hearing_radius", "vision_radius", "cone_angle")


def action(agent_id: int, *, distance: float = 0.0, direction: float = 0.0,
           turn: float = 0.0, spawn: bool = False) -> ActionRequest:
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=turn,
                         spawn_agent=spawn)


def fixture(seed: int = 6001, x: float = 200.0, y: float = 200.0):
    env = Environment(400, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    parent = env.spawn_agent(x=x, y=y)
    parent.direction = 0.0
    parent.max_age = 1e9
    return env, parent


def advance(env: Environment, actions: list[ActionRequest]):
    return step_environment(env, [(a.agent_id, a) for a in actions], dt=0.1)


def birth_gate_and_action_order():
    cases = []
    for initial_energy, distance, turn in (
        (100.0, 0.0, 0.0),
        (100.0001, 0.0, 0.0),
        (100.11, 0.0, 0.0),
        (150.0, 0.0, 0.0),
        (102.0, 0.0, 0.0),
        (102.0, 20.0, 0.0),
        (102.0, 0.0, math.pi),
    ):
        env, parent = fixture()
        parent.energy = initial_energy
        parent_id = parent.agent_id
        result = advance(env, [action(parent_id, distance=distance, turn=turn, spawn=True)])
        parent_after = env.agents_dict.get(parent_id)
        children = [a for a in env.agents if a.agent_id != parent_id]
        cases.append({
            "initial_energy": initial_energy,
            "move_distance": distance,
            "turn_angle": turn,
            "birth_occurred": bool(children),
            "parent_survived_tick": parent_after is not None,
            "parent_energy_after_tick": None if parent_after is None else parent_after.energy,
            "child_energy_after_tick": None if not children else children[0].energy,
            "child_age_after_tick": None if not children else children[0].age,
            "child_observation_cached": None if not children else children[0].agent_id in env.agent_observations,
            "population_after_tick": result["num_agents"],
        })

    by_key = {(c["initial_energy"], c["move_distance"], c["turn_angle"]): c for c in cases}
    assert not by_key[(100.0, 0.0, 0.0)]["birth_occurred"]
    assert by_key[(100.0001, 0.0, 0.0)]["birth_occurred"]
    assert not by_key[(100.0001, 0.0, 0.0)]["parent_survived_tick"]
    assert by_key[(100.11, 0.0, 0.0)]["parent_survived_tick"]
    assert by_key[(102.0, 0.0, 0.0)]["birth_occurred"]
    assert not by_key[(102.0, 20.0, 0.0)]["birth_occurred"]
    assert by_key[(102.0, 0.0, math.pi)]["birth_occurred"]  # turn costs only 0.5
    return {
        "classification": "engine fixture; all cases use one action for the parent",
        "finding": "movement and turning costs are charged before the strict >100 birth gate; passive drain follows birth",
        "interaction_note": "If the marginal-energy parent dies in non_agent_step, list removal skips the appended child for that tick; the child remains age 0/energy 75 with no cached observation. This is only a 0.1-energy timing effect and does not grow population.",
        "cases": cases,
    }


def inheritance_distribution(samples: int = 20000):
    env, parent = fixture(seed=6002)
    env.rng = random.Random(6172026)
    parent.rng = env.rng
    parent_values = {name: float(getattr(parent, name)) for name in TRAITS}
    changed = {name: 0 for name in TRAITS}
    increased = {name: 0 for name in TRAITS}
    decreased = {name: 0 for name in TRAITS}
    changed_trait_counts = []
    child_max_ages = []

    for _ in range(samples):
        child = env.spawn_agent(parent=parent)
        count = 0
        for name in TRAITS:
            before = parent_values[name]
            after = float(getattr(child, name))
            if not math.isclose(after, before, rel_tol=0.0, abs_tol=1e-12):
                changed[name] += 1
                count += 1
                if after > before:
                    increased[name] += 1
                else:
                    decreased[name] += 1
        changed_trait_counts.append(count)
        child_max_ages.append(float(child.max_age))
        env.agents.remove(child)
        env.agents_dict.pop(child.agent_id, None)

    any_changed = sum(n > 0 for n in changed_trait_counts)
    none_changed = samples - any_changed
    per_trait = {}
    for name in TRAITS:
        per_trait[name] = {
            "apparent_change_rate": changed[name] / samples,
            "increase_rate": increased[name] / samples,
            "decrease_rate": decreased[name] / samples,
            "expected_apparent_change_rate_below_cap": 0.1,
        }
    return {
        "classification": "privileged direct-birth Monte Carlo; exact upstream spawn method, no ecology or energy budget",
        "samples": samples,
        "rng_seed": 6172026,
        "parent_traits": parent_values,
        "per_trait": per_trait,
        "any_apparent_trait_change_rate": any_changed / samples,
        "no_apparent_trait_change_rate": none_changed / samples,
        "expected_any_change_rate": 1 - 0.9 ** 6,
        "changed_trait_count_histogram": {str(i): changed_trait_counts.count(i) for i in range(7)},
        "max_age": {
            "parent_value_fixture": parent.max_age,
            "child_min": min(child_max_ages),
            "child_max": max(child_max_ages),
            "child_mean": statistics.mean(child_max_ages),
            "source_rule": "fresh Uniform(60,120); not inherited and not exposed in ObservationResponse",
        },
    }


def cap_floor_and_overcap_energy(samples: int = 12000):
    env, parent = fixture(seed=6003)
    env.rng = random.Random(63003)
    parent.rng = env.rng
    parent.speed = 20.0
    parent.sprint_speed = 40.0
    parent.max_energy = 1000.0
    parent.hearing_radius = env.chunk_size / 4
    parent.vision_radius = env.chunk_size
    parent.cone_angle = math.pi / 2
    unchanged = {name: 0 for name in TRAITS}
    decreased = {name: 0 for name in TRAITS}
    increased = {name: 0 for name in TRAITS}
    for _ in range(samples):
        child = env.spawn_agent(parent=parent)
        for name in TRAITS:
            before, after = float(getattr(parent, name)), float(getattr(child, name))
            if math.isclose(after, before, rel_tol=0.0, abs_tol=1e-12):
                unchanged[name] += 1
            elif after < before:
                decreased[name] += 1
            else:
                increased[name] += 1
        env.agents.remove(child)
        env.agents_dict.pop(child.agent_id, None)

    cap_rates = {name: {
        "unchanged_rate": unchanged[name] / samples,
        "decrease_rate": decreased[name] / samples,
        "increase_rate": increased[name] / samples,
        "expected_at_cap": {"unchanged": 0.95, "decrease": 0.05, "increase": 0.0},
    } for name in TRAITS}
    assert all(increased[name] == 0 for name in TRAITS)

    sterile_env, sterile = fixture(seed=6004)
    sterile.max_energy = 90.0
    sterile.energy = 90.0
    advance(sterile_env, [action(sterile.agent_id, spawn=True)])
    sterile_case = {
        "max_energy": sterile.max_energy,
        "population": len(sterile_env.agents),
        "can_ever_cross_strict_birth_gate_via_fruit": False,
    }
    assert len(sterile_env.agents) == 1

    # A max-energy-120 parent can produce a child capped below its fixed 75
    # starting energy when the max-energy mutation multiplier is below 0.625.
    low_env, low_parent = fixture(seed=6005)
    low_parent.max_energy = 120.0
    low_env.rng = random.Random(612005)
    low_parent.rng = low_env.rng
    overcap_child = None
    births_to_find = None
    for birth in range(1, 5001):
        child = low_env.spawn_agent(parent=low_parent)
        if child.max_energy < child.energy:
            overcap_child = child
            births_to_find = birth
            break
        low_env.agents.remove(child)
        low_env.agents_dict.pop(child.agent_id, None)
    assert overcap_child is not None
    before_food = overcap_child.energy
    fruit = low_env.spawn_fruit(x=overcap_child.x, y=overcap_child.y, radius=5)
    assert fruit is not None
    low_env.non_agent_step(0.1)
    after_food = overcap_child.energy
    assert after_food == overcap_child.max_energy and after_food < before_food

    low_env2, tiny_parent = fixture(seed=6006)
    for name in TRAITS:
        setattr(tiny_parent, name, 1e-9)
    low_env2.rng = random.Random(66006)
    tiny_parent.rng = low_env2.rng
    minima = {name: float("inf") for name in TRAITS}
    for _ in range(1000):
        child = low_env2.spawn_agent(parent=tiny_parent)
        for name in TRAITS:
            minima[name] = min(minima[name], float(getattr(child, name)))
        low_env2.agents.remove(child)
        low_env2.agents_dict.pop(child.agent_id, None)
    assert all(0 < value < 1e-9 for value in minima.values())

    return {
        "classification": "privileged fixtures; exact upstream inheritance and interaction methods",
        "upper_caps": {"speed": 20, "sprint_speed": 40, "max_energy": 1000,
                       "hearing_radius": "chunk_size/4", "vision_radius": "chunk_size",
                       "cone_angle": "pi/2"},
        "at_cap_monte_carlo": {"samples": samples, "rates": cap_rates},
        "lower_floor": {"explicit_floor_exists": False, "sample_parent_value": 1e-9,
                        "minimum_children": minima},
        "sterile_below_gate": sterile_case,
        "newborn_above_own_cap": {
            "parent_max_energy": 120.0,
            "births_to_first_seeded_example": births_to_find,
            "child_energy_before_touching_fruit": before_food,
            "child_max_energy": overcap_child.max_energy,
            "child_energy_after_touching_fruit": after_food,
            "note": "Fruit was artificially placed for isolation; eating clamps energy downward to max_energy.",
        },
    }


def placement_geometry(samples: int = 12000):
    center_env, center_parent = fixture(seed=6007, x=200, y=200)
    center_env.rng = random.Random(67007)
    center_parent.rng = center_env.rng
    distances = []
    for _ in range(samples):
        child = center_env.spawn_agent(parent=center_parent)
        distances.append(math.hypot(child.x - center_parent.x, child.y - center_parent.y))
        center_env.agents.remove(child)
        center_env.agents_dict.pop(child.agent_id, None)

    corner_env, corner_parent = fixture(seed=6008, x=35, y=35)
    corner_env.rng = random.Random(68008)
    corner_parent.rng = corner_env.rng
    fallback = 0
    corner_distances = []
    for _ in range(samples):
        child = corner_env.spawn_agent(parent=corner_parent)
        d = math.hypot(child.x - corner_parent.x, child.y - corner_parent.y)
        corner_distances.append(d)
        fallback += d == 0.0
        corner_env.agents.remove(child)
        corner_env.agents_dict.pop(child.agent_id, None)

    # Find a deterministic legal spawn under the one-sided 20x20 clearance
    # test whose centered radius-5 circle nevertheless overlaps the obstacle.
    overlap_seed = None
    overlap_record = None
    for seed in range(10000):
        env, parent = fixture(seed=6009, x=150, y=100)
        obstacle = env.spawn_obstacle(x=100, y=70, width=40, height=60)
        env.rng = random.Random(seed)
        parent.rng = env.rng
        child = env.spawn_agent(parent=parent)
        if env._in_obstacle((child.x, child.y), child.size, [obstacle]):
            overlap_seed = seed
            overlap_record = {"seed": seed, "child_x": child.x, "child_y": child.y,
                              "obstacle": {"x": obstacle.x, "y": obstacle.y,
                                           "width": obstacle.width, "height": obstacle.height},
                              "spawn_clearance_function_said_free": True,
                              "centered_collision_function_says_overlap": True}
            break
    assert overlap_seed is not None

    # Spawn ignores predators. Put an active predator exactly at the deterministic
    # proposed child location and show the child can die during the same world tick.
    lethal_seed = 73
    plan_rng = random.Random(lethal_seed)
    angle = plan_rng.uniform(0, 2 * math.pi)
    radius = plan_rng.uniform(10, 30)
    planned = (200 + radius * math.cos(angle), 200 + radius * math.sin(angle))
    assert radius > 15
    lethal_env, lethal_parent = fixture(seed=6010, x=200, y=200)
    lethal_parent.energy = 150
    predator = lethal_env.spawn_predator(x=planned[0], y=planned[1])
    assert predator is not None
    predator.resting = False
    predator.energy = 100
    # Reset only after constructing the predator, whose constructor consumes RNG.
    # This makes the following birth use the two placement draws above.
    lethal_env.rng = random.Random(lethal_seed)
    lethal_parent.rng = lethal_env.rng
    predicted_child_id = lethal_env._next_agent_id
    advance(lethal_env, [action(lethal_parent.agent_id, spawn=True)])
    same_tick_death = predicted_child_id not in lethal_env.agents_dict
    assert same_tick_death

    return {
        "classification": "privileged placement fixtures; source method unmodified",
        "empty_center": {"samples": samples, "mean_radial_distance": statistics.mean(distances),
                         "min": min(distances), "max": max(distances),
                         "source_distribution": "radius Uniform(10,30), not area-uniform"},
        "near_boundary": {"parent": [35, 35], "samples": samples,
                          "exact_parent_fallback_rate": fallback / samples,
                          "mean_resulting_distance": statistics.mean(corner_distances)},
        "one_sided_clearance_mismatch": overlap_record,
        "predator_occupancy_ignored": {
            "fixture_seed": lethal_seed, "planned_spawn_radius": radius,
            "predator_placed_at_planned_child_position": list(planned),
            "child_died_in_birth_tick": same_tick_death,
            "note": "Adversarial arranged geometry; demonstrates hazard, not an acquisition rate.",
        },
    }


def mutate_once(values: dict[str, float], rng: random.Random):
    caps = {"speed": 20.0, "sprint_speed": 40.0, "max_energy": 1000.0,
            "hearing_radius": 100.0, "vision_radius": 400.0,
            "cone_angle": math.pi / 2}
    child = {}
    for name in TRAITS:
        value = values[name]
        if rng.random() < 0.1:
            value *= rng.uniform(0.5, 1.5)
        child[name] = min(value, caps[name])
    return child


def lineage_economics(trials: int = 10000, max_births: int = 5000):
    founder = {"speed": 10.0, "sprint_speed": 20.0, "max_energy": 500.0,
               "hearing_radius": 50.0, "vision_radius": 200.0,
               "cone_angle": math.pi / 3}
    rng = random.Random(696969)
    first_any_no_tradeoff = []
    first_effective_speed_up = []
    first_joint_speed_energy_up = []

    for _ in range(trials):
        hits = {}
        for birth in range(1, max_births + 1):
            child = mutate_once(founder, rng)
            deltas = {name: child[name] - founder[name] for name in TRAITS}
            if "any_no_tradeoff" not in hits and any(v > 0 for v in deltas.values()) and all(v >= 0 for v in deltas.values()):
                hits["any_no_tradeoff"] = birth
            effective_child = min(child["speed"], child["sprint_speed"])
            effective_parent = min(founder["speed"], founder["sprint_speed"])
            if "speed" not in hits and effective_child > effective_parent:
                hits["speed"] = birth
            if "joint" not in hits and effective_child > effective_parent and child["max_energy"] > founder["max_energy"]:
                hits["joint"] = birth
            if len(hits) == 3:
                break
        first_any_no_tradeoff.append(hits.get("any_no_tradeoff", max_births + 1))
        first_effective_speed_up.append(hits.get("speed", max_births + 1))
        first_joint_speed_energy_up.append(hits.get("joint", max_births + 1))

    def summarize(values):
        ordered = sorted(values)
        median = statistics.median(values)
        p90 = ordered[int((len(ordered) - 1) * 0.9)]
        return {
            "median_births": median,
            "p90_births": p90,
            "mean_births": statistics.mean(values),
            "not_reached": sum(v > max_births for v in values),
            "median_birth_energy": median * 100,
            "median_equivalent_ripe_60_energy_fruits_ignoring_passive_cost": median * 100 / 60,
        }

    return {
        "classification": "optimistic exact-distribution Monte Carlo; immortal founder, unlimited food, no predators",
        "trials": trials,
        "max_births": max_births,
        "rng_seed": 696969,
        "first_visible_improvement_with_no_visible_trait_loss": summarize(first_any_no_tradeoff),
        "first_effective_movement_improvement": summarize(first_effective_speed_up),
        "first_joint_effective_movement_and_max_energy_improvement": summarize(first_joint_speed_energy_up),
        "ordinary_observation": "All six inherited traits are returned exactly after birth; parentage and max_age are not.",
    }


def generated_map_births(seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 42)):
    rows = []
    for seed in seeds:
        core = SimulationCore(seed=seed, starting_predators=0)
        env = core.env
        before = {a.agent_id: (a.x, a.y) for a in env.agents}
        first_parent = env.agents[0]
        predicted_id = env._next_agent_id
        actions = []
        for agent in list(env.agents):
            actions.append(action(agent.agent_id, spawn=agent.agent_id == first_parent.agent_id))
        state = core.step([(a.agent_id, a) for a in actions])
        child = env.agents_dict.get(predicted_id)
        parent = env.agents_dict.get(first_parent.agent_id)
        assert child is not None and parent is not None
        distance = math.hypot(child.x - before[first_parent.agent_id][0],
                              child.y - before[first_parent.agent_id][1])
        returned = next(o for o in state["observations"] if o["agent_id"] == predicted_id)
        rows.append({
            "seed": seed,
            "population_after": state["num_agents"],
            "parent_energy_after": parent.energy,
            "child_energy_after": child.energy,
            "child_returned_in_same_response": returned is not None,
            "placement_distance": distance,
            "fell_back_exactly_to_parent": distance == 0.0,
            "child_traits": {name: float(getattr(child, name)) for name in TRAITS},
        })
        assert state["num_agents"] == 6
        assert math.isclose(parent.energy, 49.9, abs_tol=1e-8)
        assert math.isclose(child.energy, 74.9, abs_tol=1e-8)

    return {
        "classification": "contract-compliant default generated maps; one action per five pre-existing agents; no spawned-child action",
        "default_dimensions": [1600, 1200],
        "seeds": list(seeds),
        "runs": rows,
        "fallback_count": sum(r["fell_back_exactly_to_parent"] for r in rows),
        "same_response_child_count": sum(r["child_returned_in_same_response"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "survival/results/mechanics_hunt/06_reproduction.json")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--lineage-trials", type=int, default=10000)
    parser.add_argument("--skip-generated-maps", action="store_true")
    args = parser.parse_args()

    result = {
        "source_commit": SOURCE_COMMIT,
        "scope": "local exact-engine reproduction mechanics; no remote calls or submissions",
        "known_reproductions": [
            "strict parent energy >100 gate and 100-energy charge",
            "newborn starts at 75 energy",
            "six traits independently attempt 10% multiplicative mutation and have upper caps",
        ],
        "birth_gate_and_action_order": birth_gate_and_action_order(),
        "inheritance_distribution": inheritance_distribution(args.samples),
        "caps_floors_and_energy": cap_floor_and_overcap_energy(max(4000, args.samples // 2)),
        "placement_geometry": placement_geometry(max(4000, args.samples // 2)),
        "lineage_economics": lineage_economics(args.lineage_trials),
    }
    if not args.skip_generated_maps:
        result["generated_map_validation"] = generated_map_births()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

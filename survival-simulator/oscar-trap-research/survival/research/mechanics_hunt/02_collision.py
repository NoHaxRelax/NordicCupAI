"""Collision and endpoint-movement probes for the unmodified survival engine.

All cases are local.  The small fixtures deliberately arrange positions to isolate
geometry.  The generated-map scan uses normal maps but privileged coordinates; it
is an incidence measurement, not an observation-only controller benchmark.
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

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "survival-simulator"
sys.path.insert(0, str(VENDOR))

from src.core import SimulationCore
from src.elements.biome import Forest_biome
from src.elements.environment import Environment
from src.elements.tree import Tree
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment

SOURCE_COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"


def fixture(seed: int = 20260917) -> tuple[Environment, object]:
    env = Environment(400, 400, 400, random.Random(seed))
    env.biome_map.fill(Forest_biome())
    agent = env.spawn_agent(x=100, y=200)
    agent.direction = 0.0
    agent.max_age = 1e9
    return env, agent


def action(agent_id: int = 0, distance: float = 0.0,
           direction: float = 0.0) -> ActionRequest:
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=0.0,
                         spawn_agent=False)


def advance(env: Environment, requests: Iterable[ActionRequest]):
    return step_environment(env, [(a.agent_id, a) for a in requests])


def state(entity) -> dict:
    return {
        "x": float(entity.x), "y": float(entity.y),
        "energy": float(entity.energy),
    }


def set_agent(env: Environment, agent, *, x: float, y: float,
              speed: float, sprint: float, energy: float = 150.0) -> None:
    agent.x, agent.y = x, y
    agent.direction = 0.0
    agent.speed, agent.sprint_speed = speed, sprint
    agent.energy = energy
    agent.max_energy = 500.0
    env._update_agent_grid()


def interval_in_open_rect(p0: tuple[float, float], p1: tuple[float, float],
                          rect: tuple[float, float, float, float]):
    """Return the open-segment interval inside an open axis-aligned rectangle."""
    lower, upper = 0.0, 1.0
    for value, delta, lo, hi in (
        (p0[0], p1[0] - p0[0], rect[0], rect[1]),
        (p0[1], p1[1] - p0[1], rect[2], rect[3]),
    ):
        if delta == 0:
            if not lo < value < hi:
                return None
            continue
        a, b = (lo - value) / delta, (hi - value) / delta
        if a > b:
            a, b = b, a
        lower, upper = max(lower, a), min(upper, b)
        if not lower < upper:
            return None
    lo = max(lower, 0.0)
    hi = min(upper, 1.0)
    return (lo, hi) if lo < hi else None


def obstacle_rect(obs, radius: float = 0.0):
    return (obs.x - radius, obs.x + obs.width + radius,
            obs.y - radius, obs.y + obs.height + radius)


def synthetic_probes() -> dict:
    out: dict = {}

    # Known endpoint-only wall and post-collision clamp cases, with controls.
    wall_threshold = {}
    for label, width, sprint in (
        ("founder_width_10_positive", 10.0, 20.0),
        ("founder_width_10_001_control", 10.001, 20.0),
        ("cap_width_30_positive", 30.0, 40.0),
        ("cap_width_30_001_control", 30.001, 40.0),
    ):
        env, agent = fixture()
        env.spawn_obstacle(x=200, y=100, width=width, height=200)
        set_agent(env, agent, x=195, y=200,
                  speed=min(20.0, sprint), sprint=sprint)
        advance(env, [action(distance=sprint)])
        wall_threshold[label] = {
            **state(agent), "wall_width": width, "requested": sprint,
            "endpoint_inside_any_obstacle": env._in_obstacle(
                (agent.x, agent.y), agent.size, env.obstacles),
        }
    assert math.isclose(wall_threshold["founder_width_10_positive"]["x"], 215)
    assert not math.isclose(wall_threshold["founder_width_10_001_control"]["x"], 215)
    assert math.isclose(wall_threshold["cap_width_30_positive"]["x"], 235)
    assert not math.isclose(wall_threshold["cap_width_30_001_control"]["x"], 235)
    out["wall_tunnel_exact_threshold"] = wall_threshold

    boundary = {}
    for label, sprint in (("cap_40_positive", 40.0),
                          ("39_999_control", 39.999)):
        env, agent = fixture()
        set_agent(env, agent, x=35, y=200, speed=20, sprint=sprint)
        advance(env, [action(distance=sprint, direction=math.pi)])
        boundary[label] = {
            **state(agent),
            "inside_boundary_after_move": env._in_obstacle(
                (agent.x, agent.y), agent.size, env.obstacles),
        }
    assert boundary["cap_40_positive"]["x"] == 5
    assert boundary["cap_40_positive"]["inside_boundary_after_move"]
    assert not boundary["39_999_control"]["inside_boundary_after_move"]
    out["boundary_tunnel_then_clamp"] = boundary

    # Strict inequalities: tangent points are clear, any positive inward epsilon is not.
    env, agent = fixture()
    obs = env.spawn_obstacle(x=200, y=100, width=30, height=200)
    tangent = {
        "left_tangent": env._in_obstacle((195.0, 200.0), 5, [obs]),
        "left_inward_1e_9": env._in_obstacle((195.0 + 1e-9, 200.0), 5, [obs]),
        "right_tangent": env._in_obstacle((235.0, 200.0), 5, [obs]),
        "right_inward_1e_9": env._in_obstacle((235.0 - 1e-9, 200.0), 5, [obs]),
    }
    set_agent(env, agent, x=195, y=150, speed=10, sprint=20)
    advance(env, [action(distance=20, direction=math.pi / 2)])
    tangent["move_along_exact_tangent"] = state(agent)
    assert tangent["left_tangent"] is False
    assert tangent["left_inward_1e_9"] is True
    assert math.isclose(agent.x, 195) and math.isclose(agent.y, 170)
    out["strict_tangency"] = tangent

    # A founder can cut through the physical corner of a normal-sized rectangle.
    corners = {}
    for label, requested in (("corner_cut_positive", math.sqrt(2) * 14.0),
                             ("short_endpoint_blocked_control", 19.0)):
        env, agent = fixture()
        obs = env.spawn_obstacle(x=200, y=200, width=50, height=50)
        start = (195.0, 209.0)
        set_agent(env, agent, x=start[0], y=start[1],
                  speed=10, sprint=20)
        advance(env, [action(distance=requested, direction=-math.pi / 4)])
        intended = (start[0] + requested / math.sqrt(2),
                    start[1] - requested / math.sqrt(2))
        physical_hit = interval_in_open_rect(
            start, intended, obstacle_rect(obs)) is not None
        corners[label] = {
            **state(agent), "requested": requested,
            "intended_x": intended[0], "intended_y": intended[1],
            "segment_crosses_physical_rectangle": physical_hit,
            "reached_intended_endpoint": math.isclose(agent.x, intended[0]) and
                                         math.isclose(agent.y, intended[1]),
        }
    assert corners["corner_cut_positive"]["segment_crosses_physical_rectangle"]
    assert corners["corner_cut_positive"]["reached_intended_endpoint"]
    # The shorter control ends inside the expanded AABB and is redirected.
    assert not corners["short_endpoint_blocked_control"]["reached_intended_endpoint"]
    corners["symmetric_max_physical_penetration_by_distance"] = {
        "founder_20": 20 / (2 * math.sqrt(2)) - 5,
        "cap_40": 40 / (2 * math.sqrt(2)) - 5,
    }
    out["corner_cut"] = corners

    # Collision is an expanded square, not circle-vs-rectangle geometry.
    env, _ = fixture()
    obs = env.spawn_obstacle(x=200, y=100, width=30, height=200)
    point = (195.1, 95.1)
    euclidean_corner_distance = math.hypot(200 - point[0], 100 - point[1])
    out["square_corner_false_positive"] = {
        "point": list(point),
        "engine_blocked": env._in_obstacle(point, 5, [obs]),
        "distance_to_physical_corner": euclidean_corner_distance,
        "circular_radius": 5,
        "outside_true_circle_contact": euclidean_corner_distance > 5,
        "maximum_extra_corner_reach": 5 * (math.sqrt(2) - 1),
    }
    assert out["square_corner_false_positive"]["engine_blocked"]
    assert out["square_corner_false_positive"]["outside_true_circle_contact"]

    # A blocked endpoint searches deterministic 10-degree alternatives.
    def redirected(start_x: float, with_obstacle: bool):
        env, agent = fixture()
        if with_obstacle:
            env.spawn_obstacle(x=200, y=100, width=30, height=200)
        set_agent(env, agent, x=start_x, y=200, speed=10, sprint=20)
        advance(env, [action(distance=20)])
        return {
            **state(agent),
            "actual_heading_degrees": math.degrees(
                math.atan2(agent.y - 200, agent.x - start_x)),
        }

    out["blocked_endpoint_direction_search"] = {
        "deep_block_redirect": redirected(190, True),
        "near_edge_redirect": redirected(176, True),
        "no_obstacle_control": redirected(190, False),
    }
    assert math.isclose(
        out["blocked_endpoint_direction_search"]["deep_block_redirect"]["actual_heading_degrees"],
        -80.0)
    assert math.isclose(
        out["blocked_endpoint_direction_search"]["near_edge_redirect"]["actual_heading_degrees"],
        -20.0)

    # Trees and other agents are not movement-collision objects.
    env, agent = fixture()
    tree = Tree(120, 200, radius=10)
    env.trees.append(tree)
    env._update_tree_grid()
    advance(env, [action(distance=20)])
    tree_state = state(agent)
    env, agent = fixture()
    other = env.spawn_agent(x=120, y=200)
    advance(env, [action(distance=20), action(agent_id=other.agent_id)])
    out["non_solid_entities"] = {
        "tree_center_endpoint": tree_state,
        "two_agents_same_endpoint": {
            "first": state(agent), "second": state(other),
            "distance": math.hypot(agent.x - other.x, agent.y - other.y),
        },
    }
    assert tree_state["x"] == 120
    assert out["non_solid_entities"]["two_agents_same_endpoint"]["distance"] == 0

    # Fruit and predation contacts are also strict endpoint tests.
    fruit_cases = {}
    for label, distance in (("cross_and_end_tangent", 20.0),
                            ("end_just_inside", 19.999)):
        env, agent = fixture()
        fruit = env.spawn_fruit(x=110, y=200, radius=5)
        advance(env, [action(distance=distance)])
        fruit_cases[label] = {
            **state(agent), "fruit_remaining": fruit in env.fruits,
            "final_center_distance": abs(agent.x - fruit.x),
        }
    assert fruit_cases["cross_and_end_tangent"]["fruit_remaining"]
    assert not fruit_cases["end_just_inside"]["fruit_remaining"]
    out["fruit_endpoint_contact"] = fruit_cases

    env, agent = fixture()
    predator = env.spawn_predator(x=130, y=200)
    predator.direction = math.pi
    predator.resting = False
    predator.energy = 100
    advance(env, [action()])
    first = {
        "agent_alive": agent in env.agents,
        "predator": state(predator),
        "distance": math.hypot(agent.x - predator.x, agent.y - predator.y),
    }
    advance(env, [action()])
    second = {"agent_alive": agent in env.agents, "predator": state(predator)}
    assert first["agent_alive"] and first["distance"] == 15
    assert not second["agent_alive"]
    out["predator_strict_endpoint_contact"] = {
        "first_tick_exact_tangent": first,
        "next_tick_control": second,
    }

    # Low-energy cap prevents the otherwise exact max-trait wall crossing.
    low_energy = {}
    for label, energy in (("above_threshold", 101.0),
                          ("below_threshold", 99.0)):
        env, agent = fixture()
        env.spawn_obstacle(x=200, y=100, width=30, height=200)
        set_agent(env, agent, x=195, y=200, speed=20, sprint=40, energy=energy)
        advance(env, [action(distance=40)])
        low_energy[label] = state(agent)
    assert low_energy["above_threshold"]["x"] == 235
    assert low_energy["below_threshold"]["x"] != 235
    out["energy_gate_on_cap_tunnel"] = low_energy
    return out


def first_crossed_obstacle(p0, p1, obstacles, radius=0.0):
    for index, obs in enumerate(obstacles):
        interval = interval_in_open_rect(p0, p1, obstacle_rect(obs, radius))
        if interval is not None:
            return index, obs, interval
    return None


def generated_map_scan(seeds=(1, 7, 42), samples_per_distance=50_000) -> dict:
    all_maps = []
    aggregate = {
        str(request): {"valid_clear_endpoint_moves": 0,
                       "swept_expanded_aabb_misses": 0,
                       "physical_rectangle_crossings": 0}
        for request in (20.0, 40.0)
    }
    examples = []

    for seed in seeds:
        env = SimulationCore(seed=seed).env
        internal = env.obstacles[4:]
        dims = [value for obs in internal for value in (obs.width, obs.height)]
        record = {
            "seed": seed, "internal_obstacles": len(internal),
            "minimum_dimension": min(dims),
            "maximum_dimension": max(dims),
            "requests": {},
        }
        sample_rng = random.Random(20_260_917 + seed)
        for request in (20.0, 40.0):
            counters = {"sampled": samples_per_distance, "free_starts": 0,
                        "valid_clear_endpoint_moves": 0,
                        "swept_expanded_aabb_misses": 0,
                        "physical_rectangle_crossings": 0}
            for _ in range(samples_per_distance):
                p0 = (sample_rng.uniform(35, env.width - 35),
                      sample_rng.uniform(35, env.height - 35))
                if env._in_obstacle(p0, 5, env.obstacles):
                    continue
                counters["free_starts"] += 1
                biome = env.biome_map[int(p0[0]), int(p0[1])]
                effective = request * biome.move_penalty
                theta = sample_rng.uniform(-math.pi, math.pi)
                p1 = (p0[0] + effective * math.cos(theta),
                      p0[1] + effective * math.sin(theta))
                if env._in_obstacle(p1, 5, env.obstacles):
                    continue
                counters["valid_clear_endpoint_moves"] += 1
                expanded = first_crossed_obstacle(p0, p1, internal, radius=5)
                physical = first_crossed_obstacle(p0, p1, internal, radius=0)
                if expanded:
                    counters["swept_expanded_aabb_misses"] += 1
                if physical:
                    counters["physical_rectangle_crossings"] += 1
                    if len(examples) < 12:
                        index, obs, interval = physical
                        # Verify the exact candidate through the original movement method.
                        probe = env.agents[0]
                        set_agent(env, probe, x=p0[0], y=p0[1],
                                  speed=10 if request == 20 else 20,
                                  sprint=request, energy=150)
                        env.update_entity_position(
                            probe, request, theta, env._get_local_obstacles(probe))
                        examples.append({
                            "seed": seed, "request": request,
                            "effective_distance": effective,
                            "biome": biome.type,
                            "start": list(p0), "intended_endpoint": list(p1),
                            "engine_endpoint": [float(probe.x), float(probe.y)],
                            "engine_reached_intended": math.isclose(probe.x, p1[0]) and
                                                       math.isclose(probe.y, p1[1]),
                            "obstacle_index_internal": index,
                            "obstacle": {"x": obs.x, "y": obs.y,
                                         "width": obs.width, "height": obs.height},
                            "physical_intersection_t_interval": list(interval),
                        })
            record["requests"][str(request)] = counters
            for key in aggregate[str(request)]:
                aggregate[str(request)][key] += counters[key]
        all_maps.append(record)

    for request, values in aggregate.items():
        denominator = values["valid_clear_endpoint_moves"]
        values["expanded_miss_rate_per_clear_move"] = (
            values["swept_expanded_aabb_misses"] / denominator)
        values["physical_crossing_rate_per_clear_move"] = (
            values["physical_rectangle_crossings"] / denominator)

    assert all(example["engine_reached_intended"] for example in examples)
    assert all(m["minimum_dimension"] > 30 for m in all_maps)
    return {
        "method": (
            "Privileged uniform start/direction Monte Carlo on generated maps. "
            "Start and endpoint must be engine-clear; segment intersections are "
            "then measured analytically and examples replayed through the original "
            "update_entity_position method. This is not controller incidence."
        ),
        "samples_per_seed_per_request": samples_per_distance,
        "maps": all_maps,
        "aggregate": aggregate,
        "verified_physical_crossing_examples": examples,
        "opposite_side_wall_tunnel_at_cap": {
            "possible_on_scanned_internal_obstacles": False,
            "reason": (
                "An agent needs obstacle dimension + 2*radius displacement. "
                "Every scanned internal dimension is >30, so required displacement "
                "is >40 while the trait cap is 40 before terrain slowdown."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "mechanics_hunt" / "02_collision.json")
    parser.add_argument("--samples", type=int, default=50_000)
    args = parser.parse_args()
    result = {
        "source_commit": SOURCE_COMMIT,
        "scope": "local unmodified engine; no remote calls",
        "contract": "every executed case uses at most one action per agent per tick",
        "synthetic_fixtures": synthetic_probes(),
        "generated_maps": generated_map_scan(samples_per_distance=args.samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

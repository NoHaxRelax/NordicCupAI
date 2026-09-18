"""Terrain-transition and energy probes for the vendored survival simulator.

All claims are produced by the unmodified upstream engine.  Synthetic stripe
fixtures isolate sampling order; generated-map checks use the upstream map
generator at default dimensions.  No network or evaluation endpoint is used.
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

import numpy as np

from src.elements.biome import (
    Desert_biome,
    Forest_biome,
    Grassland_biome,
    River_biome,
    Swamp_biome,
)
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment


COMMIT = "acfc31a4003a5f91bf11032a02cd98c178ddbd7e"
BIOMES = {
    "forest": Forest_biome,
    "grassland": Grassland_biome,
    "desert": Desert_biome,
    "swamp": Swamp_biome,
    "river": River_biome,
}


def action(agent_id: int, distance: float = 0.0, direction: float = 0.0) -> ActionRequest:
    return ActionRequest(
        agent_id=agent_id,
        move_distance=distance,
        move_direction=direction,
        turn_angle=0.0,
        spawn_agent=False,
    )


def empty_env(width: int = 400, height: int = 240, seed: int = 808) -> Environment:
    """Generated engine environment with only its normal boundary obstacles."""
    return Environment(width, height, max(width, height), random.Random(seed))


def place_agent(env: Environment, x: float, y: float, energy: float = 150.0):
    a = env.spawn_agent(x=x, y=y)
    a.direction = 0.0
    a.energy = energy
    a.max_age = 1e9
    env._update_agent_grid()
    return a


def biome_at(env: Environment, x: float, y: float) -> str:
    ix = min(max(int(x), 0), env.width - 1)
    iy = min(max(int(y), 0), env.height - 1)
    return env.biome_map[ix, iy].type


def one_action(env: Environment, a, distance: float, direction: float = 0.0) -> dict:
    before = (float(a.x), float(a.y), float(a.energy), biome_at(env, a.x, a.y))
    env.agent_step(a.agent_id, distance, direction, 0.0, False)
    after = (float(a.x), float(a.y), float(a.energy), biome_at(env, a.x, a.y))
    return {
        "start": {"x": before[0], "y": before[1], "energy": before[2], "biome": before[3]},
        "end": {"x": after[0], "y": after[1], "energy": after[2], "biome": after[3]},
        "requested": distance,
        "realized_distance": math.hypot(after[0] - before[0], after[1] - before[1]),
        "action_energy_cost": before[2] - after[2],
    }


def uniform_biome_trials() -> dict:
    rows = {}
    for name, cls in BIOMES.items():
        rows[name] = {}
        for mode, distance in (("walk", 10.0), ("sprint", 20.0)):
            env = empty_env(seed=810)
            env.biome_map.fill(cls())
            a = place_agent(env, 100.0, 120.0)
            trial = one_action(env, a, distance)
            trial["energy_per_realized_unit"] = (
                trial["action_energy_cost"] / trial["realized_distance"]
            )
            rows[name][mode] = trial

    # Same action cost, reduced displacement. Forest and grass are identical.
    assert rows["forest"]["walk"]["realized_distance"] == 10.0
    assert rows["grassland"]["walk"]["realized_distance"] == 10.0
    assert rows["desert"]["walk"]["realized_distance"] == 8.0
    assert rows["swamp"]["walk"]["realized_distance"] == 5.0
    assert rows["river"]["walk"]["realized_distance"] == 3.0
    for row in rows.values():
        assert math.isclose(row["walk"]["action_energy_cost"], 0.5)
        assert math.isclose(row["sprint"]["action_energy_cost"], 5.5)
    return rows


def transition_trials() -> dict:
    def striped() -> Environment:
        env = empty_env(seed=811)
        env.biome_map.fill(Forest_biome())
        env.biome_map[100:, :] = River_biome()
        return env

    env = striped()
    a = place_agent(env, 99.9, 120.0)
    fast_origin = one_action(env, a, 10.0)

    env = striped()
    a = place_agent(env, 100.0, 120.0)
    slow_origin = one_action(env, a, 10.0)

    env = striped()
    a = place_agent(env, 101.0, 120.0)
    slow_to_fast = one_action(env, a, 10.0, math.pi)

    env = striped()
    a = place_agent(env, 90.0, 120.0)
    exact_boundary_endpoint = one_action(env, a, 10.0)

    # Trait-cap upper bound: a naturally reachable but rare evolved creature can
    # request 40.  The 30-pixel river is a synthetic fixture, not a decoy test.
    env = empty_env(seed=811)
    env.biome_map.fill(Forest_biome())
    env.biome_map[100:130, :] = River_biome()
    a = place_agent(env, 99.0, 120.0, energy=150.0)
    a.speed, a.sprint_speed = 20.0, 40.0
    evolved_fast_origin = one_action(env, a, 40.0)

    env = empty_env(seed=811)
    env.biome_map.fill(Forest_biome())
    env.biome_map[100:130, :] = River_biome()
    a = place_agent(env, 100.0, 120.0, energy=150.0)
    a.speed, a.sprint_speed = 20.0, 40.0
    evolved_slow_origin = one_action(env, a, 40.0)

    assert math.isclose(fast_origin["realized_distance"], 10.0)
    assert fast_origin["end"]["biome"] == "river"
    assert math.isclose(slow_origin["realized_distance"], 3.0)
    assert math.isclose(slow_to_fast["realized_distance"], 3.0)
    assert slow_to_fast["end"]["biome"] == "forest"
    assert exact_boundary_endpoint["end"]["biome"] == "river"
    assert math.isclose(evolved_fast_origin["realized_distance"], 40.0)
    assert evolved_fast_origin["end"]["biome"] == "forest"
    assert math.isclose(evolved_slow_origin["realized_distance"], 12.0)
    return {
        "fast_origin_crosses_at_full_speed": fast_origin,
        "integer_boundary_slow_origin": slow_origin,
        "slow_origin_crosses_back_at_slow_speed": slow_to_fast,
        "endpoint_exactly_on_boundary_reports_destination": exact_boundary_endpoint,
        "synthetic_trait_cap_skips_30_pixel_river": evolved_fast_origin,
        "synthetic_trait_cap_slow_origin_control": evolved_slow_origin,
        "interpretation": (
            "The biome at int(start_x), int(start_y) scales the whole action; "
            "the traversed segment and destination biome are not integrated."
        ),
    }


def low_energy_and_idle_trials() -> dict:
    low = {}
    passive = {}
    current = {}
    for name, cls in BIOMES.items():
        env = empty_env(seed=812)
        env.biome_map.fill(cls())
        a = place_agent(env, 100.0, 120.0, energy=99.0)  # below 20% of max_energy=500
        low[name] = one_action(env, a, 20.0)

        env = empty_env(seed=813)
        env.biome_map.fill(cls())
        a = place_agent(env, 100.0, 120.0)
        before = a.energy
        step_environment(env, [(a.agent_id, action(a.agent_id, 0.0))], 0.1)
        passive[name] = {
            "energy_cost_one_tick": before - a.energy,
            "position": [a.x, a.y],
            "reported_biome": env.get_agent_state(a.agent_id)["biome"],
        }

    env = empty_env(seed=814)
    env.biome_map.fill(River_biome())
    a = place_agent(env, 100.0, 120.0)
    start = [a.x, a.y]
    for _ in range(10):
        step_environment(env, [(a.agent_id, action(a.agent_id, 0.0))], 0.1)
    current = {
        "start": start,
        "end": [a.x, a.y],
        "seconds": 1.0,
        "declared_stream_flow_speed": River_biome().stream_flow_speed,
    }

    for row in low.values():
        assert math.isclose(row["action_energy_cost"], 0.5)
    assert math.isclose(low["forest"]["realized_distance"], 10.0)
    assert math.isclose(low["river"]["realized_distance"], 3.0)
    for row in passive.values():
        assert math.isclose(row["energy_cost_one_tick"], 0.1)
    assert current["start"] == current["end"]
    return {
        "below_twenty_percent_sprint_request": low,
        "passive_drain": passive,
        "river_current_negative_control": current,
    }


def realistic_traversals(distance_goal: float = 90.0) -> dict:
    """One legal action per 0.1 s tick, starting with normal founder energy."""
    results = {}
    for name, cls in BIOMES.items():
        results[name] = {}
        for mode, request in (("walk", 10.0), ("sprint", 20.0)):
            env = empty_env(seed=815)
            env.biome_map.fill(cls())
            a = place_agent(env, 100.0, 120.0, energy=150.0)
            start_x = a.x
            ticks = 0
            low_energy_ticks = 0
            while a in env.agents and a.x - start_x < distance_goal and ticks < 200:
                if a.energy < a.max_energy / 5:
                    low_energy_ticks += 1
                step_environment(env, [(a.agent_id, action(a.agent_id, request))], 0.1)
                ticks += 1
            assert a in env.agents
            assert not env.predators, "unexpected predator confounds the isolated route trial"
            results[name][mode] = {
                "goal_distance": distance_goal,
                "realized_distance": a.x - start_x,
                "ticks": ticks,
                "seconds": ticks * 0.1,
                "energy_start": 150.0,
                "energy_end": a.energy,
                "total_energy_cost": 150.0 - a.energy,
                "ticks_after_low_energy_sprint_cap": low_energy_ticks,
                "fixture": "uniform biome; no starting fruit, trees, obstacles or predators beyond engine boundaries",
            }
    return results


def _slow_runs(line: np.ndarray, max_len: int) -> Iterable[tuple[int, int, float]]:
    """Yield slow runs bracketed by full-speed cells: (start, end, penalty)."""
    penalties = np.fromiter((b.move_penalty for b in line), dtype=float, count=len(line))
    i = 1
    while i < len(penalties) - 1:
        if penalties[i] >= 1.0:
            i += 1
            continue
        start = i
        while i + 1 < len(penalties) and penalties[i + 1] < 1.0:
            i += 1
        end = i
        if (
            start > 0
            and end + 1 < len(penalties)
            and penalties[start - 1] == 1.0
            and penalties[end + 1] == 1.0
            and end - start + 1 <= max_len
        ):
            yield start, end, float(np.min(penalties[start : end + 1]))
        i += 1


def find_slivers(env: Environment, max_len: int = 19) -> list[dict]:
    found = []
    # Stay clear of physical boundary walls and sample every row/column.
    for y in range(40, env.height - 40):
        for start, end, penalty in _slow_runs(env.biome_map[:, y], max_len):
            if 40 <= start - 1 and end + 1 < env.width - 40:
                found.append({
                    "axis": "x", "fixed": y, "start": start, "end": end,
                    "length": end - start + 1, "min_penalty": penalty,
                })
    for x in range(40, env.width - 40):
        for start, end, penalty in _slow_runs(env.biome_map[x, :], max_len):
            if 40 <= start - 1 and end + 1 < env.height - 40:
                found.append({
                    "axis": "y", "fixed": x, "start": start, "end": end,
                    "length": end - start + 1, "min_penalty": penalty,
                })
    return found


def generated_map_trials(seeds=(1, 7, 42)) -> dict:
    rows = {}
    for seed in seeds:
        env = Environment(1600, 1200, 400, random.Random(seed))
        slivers = find_slivers(env)
        walk = [s for s in slivers if s["length"] <= 9]
        sprint = [s for s in slivers if s["length"] <= 19]
        evolved = find_slivers(env, max_len=39)
        seed_row = {
            "area_fraction": {},
            "bracketed_slow_scanline_runs": {
                "walk_crossable_scanline_runs_length_1_to_9": len(walk),
                "founder_sprint_crossable_scanline_runs_length_1_to_19": len(sprint),
                "trait_cap_sprint_crossable_scanline_runs_length_1_to_39": len(evolved),
                "counting_note": (
                    "Horizontal and vertical scanline runs, not unique connected terrain components."
                ),
            },
            "sample_trial": None,
        }
        names = np.fromiter((b.type for b in env.biome_map.flat), dtype="U16", count=env.width * env.height)
        unique, counts = np.unique(names, return_counts=True)
        seed_row["area_fraction"] = {
            str(name): float(count / names.size) for name, count in zip(unique, counts)
        }

        sample = walk[0] if walk else (sprint[0] if sprint else None)
        if sample:
            distance = float(sample["length"] + 1)
            if sample["axis"] == "x":
                sx, sy, direction = float(sample["start"] - 1), float(sample["fixed"]), 0.0
                nx, ny = float(sample["start"]), float(sample["fixed"])
            else:
                sx, sy, direction = float(sample["fixed"]), float(sample["start"] - 1), math.pi / 2
                nx, ny = float(sample["fixed"]), float(sample["start"])

            fast_agent = place_agent(env, sx, sy, energy=150.0)
            fast_trial = one_action(env, fast_agent, distance, direction)

            # A separate real map with the same seed is not required: place a
            # second agent at the first slow pixel for the origin-sampling control.
            slow_agent = place_agent(env, nx, ny, energy=150.0)
            slow_trial = one_action(env, slow_agent, distance, direction)
            assert math.isclose(fast_trial["realized_distance"], distance)
            assert fast_trial["start"]["biome"] not in ("river", "swamp", "desert")
            assert fast_trial["end"]["biome"] not in ("river", "swamp", "desert")
            assert slow_trial["realized_distance"] < distance
            seed_row["sample_trial"] = {
                "sliver": sample,
                "request": distance,
                "fast_origin_skips_entire_sliver": fast_trial,
                "slow_origin_negative_control": slow_trial,
                "fixture_note": (
                    "Real generated biome map and founder traits/energy; positions selected "
                    "with privileged biome-map inspection; no interior obstacles/entities spawned."
                ),
            }
        rows[str(seed)] = seed_row
    return rows


def route_break_even() -> dict:
    rows = {}
    for name, cls in BIOMES.items():
        penalty = cls().move_penalty
        rows[name] = {
            "move_penalty": penalty,
            "fast_distance_equivalent_per_slow_distance": 1 / penalty,
            "max_extra_fast_detour_per_slow_distance": 1 / penalty - 1,
            "example_for_100_slow_units": {
                "straight_equivalent_fast_distance": 100 / penalty,
                "detour_can_add_up_to_fast_units_before_tying": 100 / penalty - 100,
            },
            "assumptions": (
                "Founder walking on long homogeneous segments; ignores the one-action "
                "boundary launch, obstacles, danger, fruit and map-discovery cost."
            ),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "mechanics_hunt" / "08_terrain.json",
    )
    args = parser.parse_args()
    result = {
        "source_commit": COMMIT,
        "scope": "local unmodified engine; synthetic controls plus generated maps",
        "contract": "all behavior probes issue at most one action per observed agent per tick",
        "uniform_biome_actions": uniform_biome_trials(),
        "transition_sampling": transition_trials(),
        "low_energy_passive_and_current": low_energy_and_idle_trials(),
        "realistic_90_unit_traversals": realistic_traversals(),
        "generated_default_maps": generated_map_trials(),
        "route_break_even": route_break_even(),
        "rejected_hypotheses": [
            "River current pushes stationary creatures (the declared stream_flow_speed is unused).",
            "Slow terrain reduces the movement-energy charge in proportion to displacement.",
            "Terrain classes impose different passive energy drain (all remain at 1.0).",
            "A move crossing a terrain boundary integrates modifiers along its path.",
            "Requesting sprint below 20% max energy preserves sprint displacement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

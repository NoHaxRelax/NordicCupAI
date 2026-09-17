"""Offline bounded escape search for map 10138 / fixture 20138.

This generates the deterministic static map but never calls agent_step,
non_agent_step, or another simulator step. It applies the published native
movement equations to a discretized guide-action search. The result is evidence,
not a proof of geometric impossibility.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "vendor/survival-simulator"))

from src.core import SimulationCore

GUIDE_START = (1439.8235, 46.3841)
PREDATOR_START = (1490.6028, 67.5138)
PREDATOR_HEADING = 3.5359


def assess(*, headings=32, state_cap=1000, ticks=20):
    env = SimulationCore(starting_agents=0, starting_predators=0, seed=10138).env
    obstacles = [(o.x, o.y, o.width, o.height) for o in env.obstacles]

    def terrain(point):
        x = max(0, min(env.width - 1, int(point[0])))
        y = max(0, min(env.height - 1, int(point[1])))
        return env.biome_map[x, y].move_penalty

    def blocked(point, radius):
        return any(x - radius < point[0] < x + width + radius
                   and y - radius < point[1] < y + height + radius
                   for x, y, width, height in obstacles)

    def move(point, absolute_heading, requested_distance, radius):
        distance = requested_distance * terrain(point)
        for index in range(36):
            adjusted = (absolute_heading + math.pi / 18
                        * ((index + 1) // 2) * (-1) ** index)
            candidate = (point[0] + distance * math.cos(adjusted),
                         point[1] + distance * math.sin(adjusted))
            if not blocked(candidate, radius):
                return (max(radius, min(env.width - radius, candidate[0])),
                        max(radius, min(env.height - radius, candidate[1])))
        return point

    def predator_step(point, heading, guide):
        angle = ((math.atan2(guide[1] - point[1], guide[0] - point[0])
                  - heading + math.pi) % math.tau) - math.pi
        turn = max(-.3, min(.3, angle * .5)) if abs(angle) > .05 else 0.
        direction = heading + (turn if abs(angle) > .05 else angle)
        endpoint = move(point, direction, min(15., math.dist(point, guide)), 10.)
        return endpoint, (heading + turn + math.pi) % math.tau - math.pi

    # A newly inserted guide receives no same-tick precomputed observation and
    # therefore cannot translate before this native predator movement.
    predator, predator_heading = predator_step(
        PREDATOR_START, PREDATOR_HEADING, GUIDE_START)
    blind_tick_separation = math.dist(GUIDE_START, predator)
    beam = [(blind_tick_separation, GUIDE_START, predator,
             predator_heading, [])]
    progression = []
    exhausted_at = None

    for tick in range(1, ticks + 1):
        expanded = []
        for minimum, guide, pred, pred_heading, path in beam:
            for index in range(headings):
                action_heading = math.tau * index / headings
                next_guide = move(guide, action_heading, 20., 5.)
                next_predator, next_heading = predator_step(
                    pred, pred_heading, next_guide)
                separation = math.dist(next_guide, next_predator)
                if separation >= 15.:
                    expanded.append((min(minimum, separation), next_guide,
                                     next_predator, next_heading,
                                     path + [action_heading]))
        if not expanded:
            exhausted_at = tick
            break
        # Fine state bins preserve diverse paths before the explicit memory cap.
        bins = {}
        for row in expanded:
            key = (round(row[1][0]), round(row[1][1]),
                   round(row[2][0]), round(row[2][1]), round(row[3], 1))
            if key not in bins or row[0] > bins[key][0]:
                bins[key] = row
        beam = sorted(bins.values(), key=lambda row: (
            row[0], math.dist(row[1], row[2])), reverse=True)[:state_cap]
        progression.append({
            "tick": tick,
            "retained_states": len(beam),
            "best_minimum_separation": round(beam[0][0], 4),
            "best_current_separation": round(
                math.dist(beam[0][1], beam[0][2]), 4),
        })

    top_boundary = next((row for row in obstacles
                         if row[0] == 0 and row[1] == 0
                         and row[2] == env.width), None)
    return {
        "schema": "map-10138-offline-escape-assessment-v1",
        "native_simulation_stepped": False,
        "map_seed": 10138,
        "fixture_seed": 20138,
        "guide_start": GUIDE_START,
        "predator_start": PREDATOR_START,
        "guide_start_terrain_modifier": terrain(GUIDE_START),
        "predator_start_terrain_modifier": terrain(PREDATOR_START),
        "mandatory_initial_blind_tick": True,
        "separation_after_blind_tick": round(blind_tick_separation, 4),
        "top_boundary": top_boundary,
        "headings_per_state": headings,
        "state_cap": state_cap,
        "requested_ticks": ticks,
        "no_survivor_at_tick": exhausted_at,
        "progression": progression,
        "caveat": (
            "The heading discretization and capped beam make this strong bounded "
            "evidence, not a mathematical proof that no continuous control exists."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headings", type=int, default=32)
    parser.add_argument("--state-cap", type=int, default=1000)
    parser.add_argument("--ticks", type=int, default=20)
    args = parser.parse_args()
    result = assess(headings=args.headings, state_cap=args.state_cap,
                    ticks=args.ticks)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

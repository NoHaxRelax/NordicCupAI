"""Native probe for a diagonal two-corner gate.

Two large rectangles join opposite arena boundaries.  Their facing corners are
offset in both x and y.  A size-5 agent can traverse the diagonal aperture,
while a size-10 predator cannot.  This script exercises 30 settled predators,
one guide/newcomer delivery, and a rear-side bait replacement.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.obstacle import Obstacle
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest


def command_toward(agent, point, distance=None):
    angle = math.atan2(point[1] - agent.y, point[0] - agent.x) - agent.direction
    return ActionRequest(agent_id=agent.agent_id, spawn_agent=False,
                         move_distance=min(agent.sprint_speed if distance is None else distance,
                                           math.dist((agent.x, agent.y), point)),
                         move_direction=angle, turn_angle=angle)


def sensed(predator, bait, env):
    obs = predator.observe(agents=env._get_local_agents(predator),
                           edges=env._get_local_edges(predator))
    return any(o.get("type") == "Agent" and o.get("id") == bait.agent_id for o in obs)


def run_case(output: Path, aperture: float, seed: int, overlap: float = 15.0):
    rng = random.Random(seed)
    core = SimulationCore(seed=seed, starting_agents=0, starting_predators=0)
    env = core.env
    # Barrier from the upper-left to lower-right arena boundaries. The aperture
    # is the equal x/y offset between the two facing corners.
    ax, ay = 300.0, 200.0
    # The second corner sits below and to the left, making a dogleg whose
    # narrow part is longer than one native predator sprint step.
    bx, by = ax - overlap, ay + aperture
    env.obstacles = [Obstacle(0, 0, ax, ay),
                     Obstacle(bx, by, env.width - bx, env.height - by)]
    env.edges = [edge for obstacle in env.obstacles for edge in obstacle.edges]
    env._update_spatial_grid()
    # These constructed rectangles span several chunks; use the same complete
    # static edge set supplied by benchmark fixtures rather than the grid's
    # endpoint-order-sensitive edge indexing.
    env._get_local_edges = lambda creature: env.edges
    env.spawn_predator = lambda *args, **kwargs: None

    # Place bait in the agent-width dogleg itself. This leaves >15 clearance
    # from every collision-valid predator centre while staying well inside the
    # 40-unit retention measurement despite native collision oscillation.
    bait_pos = (ax - overlap / 2.0, ay + 7.0)
    bait = Agent(*bait_pos, rng=env.rng)
    bait.agent_id = 0
    bait.energy = bait.max_energy
    env.agents = [bait]
    env.agents_dict = {0: bait}
    env._next_agent_id = 1

    predators = []
    for i in range(30):
        p = Predator(ax + 11 + rng.uniform(-0.5, 0.5), ay + 2 + rng.uniform(-0.5, 0.5), rng=env.rng)
        p.energy = p.max_energy
        p.resting = False
        p.direction = math.atan2(bait.y - p.y, bait.x - p.x)
        predators.append(p)
    env.predators = list(predators)
    env._update_agent_grid()
    env._update_predator_grid()

    guide = newcomer = replacement = None
    old_bait = bait
    replacement_arrived = False
    delivered_since = None
    minimum_original_held = 30
    maximum_original_distance = 0.0
    rear_intruders = set()
    events = []
    rows = []
    delivery_tick = 150
    replacement_tick = 450
    end_tick = 750

    for tick in range(end_tick + 1):
        now = round(tick * core.dt, 6)
        current_bait = bait
        distances = [math.dist((p.x, p.y), (current_bait.x, current_bait.y)) for p in predators]
        held = [i for i, p in enumerate(predators) if distances[i] <= 40 and sensed(p, current_bait, env)]
        original_held = sum(i < 30 for i in held)
        if tick >= delivery_tick:
            minimum_original_held = min(minimum_original_held, original_held)
            maximum_original_distance = max(maximum_original_distance, max(distances[:30]))
        # The lower-left side of the diagonal barrier is reserved for bait access.
        for i, p in enumerate(predators):
            if p.x < bx - 10 and p.y > ay + 10:
                rear_intruders.add(i + 1)

        newcomer_held = newcomer is not None and len(predators) > 30 and 30 in held
        delivered_since = (now if delivered_since is None else delivered_since) if newcomer_held else None

        if tick == delivery_tick:
            guide = Agent(ax + 85, ay - 70, rng=env.rng)
            guide.agent_id = 1
            guide.energy = guide.max_energy
            newcomer = Predator(ax + 125, ay - 105, rng=env.rng)
            newcomer.energy = newcomer.max_energy
            newcomer.resting = False
            newcomer.direction = math.atan2(guide.y - newcomer.y, guide.x - newcomer.x)
            env.agents.append(guide)
            env.agents_dict[1] = guide
            predators.append(newcomer)
            env.predators.append(newcomer)
            env._update_agent_grid(); env._update_predator_grid()
            events.append({"time": now, "event": "guide and newcomer started"})

        if tick == replacement_tick:
            replacement = Agent(ax - 55, ay + 70, rng=env.rng)
            replacement.agent_id = 2
            replacement.energy = replacement.max_energy
            env.agents.append(replacement)
            env.agents_dict[2] = replacement
            env._update_agent_grid()
            events.append({"time": now, "event": "rear replacement started"})

        actions = []
        if guide is not None and guide in env.agents:
            # Ordinary-information policy: the fixed fixture bait vector is known;
            # no predator positions or identities are read to choose this action.
            actions.append((guide.agent_id, command_toward(guide, bait_pos)))
        if replacement is not None and replacement in env.agents and not replacement_arrived:
            if math.dist((replacement.x, replacement.y), bait_pos) <= 1.0:
                replacement_arrived = True
                old_bait.energy = 0.0
                bait = replacement
                events.append({"time": now, "event": "replacement arrived; old bait exhausted"})
            else:
                actions.append((replacement.agent_id, command_toward(replacement, bait_pos, replacement.speed)))

        rows.append({"tick": tick, "time": now, "original_held": original_held,
                     "held": [i + 1 for i in held], "bait": [bait.x, bait.y],
                     "bait_id": bait.agent_id, "guide_alive": guide in env.agents if guide else None,
                     "replacement_alive": replacement in env.agents if replacement else None,
                     "replacement_arrived": replacement_arrived,
                     "predators": [[p.x, p.y] for p in predators]})
        if tick == end_tick:
            break
        bait.energy = bait.max_energy
        bait.age = 0
        core.step(actions)
        bait.energy = bait.max_energy
        bait.age = 0

    final_held = rows[-1]["held"]
    result = {
        "aperture": aperture, "overlap": overlap, "seed": seed, "dt": core.dt,
        "geometry": {"obstacles": [[0, 0, ax, ay], [bx, by, env.width-bx, env.height-by]],
                     "facing_corners": [[ax, ay], [bx, by]], "bait": list(bait_pos)},
        "policy_information": "guide and replacement use only fixed bait target; evaluation alone reads predator state",
        "events": events, "baseline_30_held": rows[delivery_tick-1]["original_held"] == 30,
        "minimum_original_held_after_delivery": minimum_original_held,
        "maximum_original_distance_after_delivery": maximum_original_distance,
        "newcomer_held_10s": delivered_since is not None and rows[-1]["time"] - delivered_since >= 10,
        "final_held_count": len(final_held), "final_held_ids": final_held,
        "rear_predator_intruders": sorted(rear_intruders),
        "replacement_arrived": replacement_arrived,
        "replacement_alive_at_end": replacement in env.agents if replacement else False,
        "old_bait_retired": old_bait not in env.agents,
    }
    result["passed"] = (result["baseline_30_held"] and minimum_original_held == 30
                        and result["newcomer_held_10s"] and len(final_held) == 31
                        and not rear_intruders and replacement_arrived
                        and result["replacement_alive_at_end"] and result["old_bait_retired"])
    case = output / f"aperture-{aperture:g}-seed-{seed}"
    case.mkdir(parents=True, exist_ok=True)
    (case / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    with gzip.open(case / "ticks.jsonl.gz", "wt") as trace:
        for row in rows:
            trace.write(json.dumps(row) + "\n")
    print(json.dumps({k: result[k] for k in ("aperture", "seed", "passed", "baseline_30_held",
          "minimum_original_held_after_delivery", "newcomer_held_10s", "final_held_count",
          "rear_predator_intruders", "replacement_arrived", "replacement_alive_at_end")}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apertures", type=float, nargs="+", default=[14, 16, 18])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 19])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_case(args.output, aperture, seed)
               for aperture in args.apertures for seed in args.seeds]
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

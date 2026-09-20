#!/usr/bin/env python3
"""Benchmark the published Python reference server's duplicate-action path.

This is deliberately local-only: it constructs the same JSON body the agent
returns, parses Pydantic ActionRequest objects, and advances exactly one
reference SimulationCore tick. It does not open a network connection.
"""

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip().replace("_", "")) for part in raw.split(",")]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--sizes must contain positive integers")
    return sizes


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="20000,100000,200000,400000")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "vendor" / "survival-simulator",
        help="directory containing src/core.py from the published reference server",
    )
    args = parser.parse_args()
    sim_root = args.sim_root.resolve()
    if not (sim_root / "src" / "core.py").is_file():
        raise SystemExit(f"Reference simulator not found: {sim_root}")
    sys.path.insert(0, str(sim_root))
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    print(
        "actions,agents,payload_mb,serialize_s,loads_s,pydantic_s,step_s,total_s,peak_rss_mb",
        flush=True,
    )
    for count in parse_sizes(args.sizes):
        sim = SimulationCore(seed=args.seed)
        target = sim.env.agents[0].agent_id
        action = {
            "agent_id": target,
            "move_distance": 0.0,
            "move_direction": 0.0,
            "turn_angle": 0.0,
            "spawn_agent": False,
        }
        started = time.perf_counter()
        body = json.dumps({"actions": [action] * count}, separators=(",", ":"))
        serialized = time.perf_counter() - started

        started = time.perf_counter()
        decoded = json.loads(body)["actions"]
        loaded = time.perf_counter() - started

        started = time.perf_counter()
        models = [ActionRequest(**item) for item in decoded]
        parsed_actions = [(model.agent_id, model) for model in models]
        validated = time.perf_counter() - started

        started = time.perf_counter()
        sim.step(parsed_actions)
        stepped = time.perf_counter() - started
        total = serialized + loaded + validated + stepped
        print(
            f"{count},{len(sim.env.agents)},{len(body) / 1e6:.3f},"
            f"{serialized:.3f},{loaded:.3f},{validated:.3f},{stepped:.3f},"
            f"{total:.3f},{rss_mb():.1f}",
            flush=True,
        )
        del body, decoded, models, parsed_actions, sim
        gc.collect()


if __name__ == "__main__":
    main()

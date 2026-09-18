"""Benchmark identical ticks, checking exact parity outside the timed sections.

  .venv/bin/python survival-simulator/scripts/benchmark_rust_engine.py --ticks 300
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time

import numpy as np
import shapely
from test_rust_engine import (
    ConstructorReplay, SimulationCore, Snapshot, actions_for, assert_same, exact, state,
)
from src.backend import resolve_backend


def benchmark(seed, ticks, agents, predators, width, height):
    start = time.perf_counter()
    core = SimulationCore(seed=seed, env_width=width, env_height=height,
                          starting_agents=agents, starting_predators=predators,
                          backend="python")
    setup_seconds = time.perf_counter() - start
    env = core.env
    rng = random.Random(seed + 1000)
    timings = {"python": [], "rust": []}
    populations = []
    for tick in range(ticks):
        populations.append((len(env.agents), len(env.predators)))
        actions = actions_for(env, rng, tick)
        snapshot = Snapshot(env)
        with ConstructorReplay().installed() as replay:
            results = {}
            for backend in ("python", "rust"):
                snapshot.restore(backend)
                replay.replaying = backend == "rust"
                start = time.perf_counter()
                result = core.step(actions)
                timings[backend].append(time.perf_counter() - start)
                results[backend] = (exact(result), state(env))
            assert_same(results["python"], results["rust"], f"seed={seed},tick={tick}")
            if replay.index != len(replay.recorded):
                raise AssertionError("constructor count differs")
    totals = {backend: sum(values) for backend, values in timings.items()}
    return {
        "seed": seed, "ticks": ticks, "initial_agents": agents,
        "requested_predators": predators, "map": [width, height],
        "setup_seconds": setup_seconds,
        "mean_agents": statistics.mean(p[0] for p in populations),
        "mean_predators": statistics.mean(p[1] for p in populations),
        "step_seconds": totals,
        "ticks_per_second": {backend: ticks / seconds for backend, seconds in totals.items()},
        "speedup": totals["python"] / totals["rust"],
        "exact_parity": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 17])
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--agents", type=int, default=31)
    parser.add_argument("--predators", type=int, default=30)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")
    resolve_backend("rust")
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "shapely": shapely.__version__, "platform": platform.platform()}), flush=True)
    for seed in args.seeds:
        print(json.dumps(benchmark(seed, args.ticks, args.agents, args.predators,
                                   args.width, args.height)), flush=True)


if __name__ == "__main__":
    main()

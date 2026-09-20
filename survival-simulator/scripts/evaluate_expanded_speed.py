"""Time a frozen native policy over a reproducible full-game seed panel."""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

CORE = None
CONFIG = None


def one(seed):
    start_wall = time.perf_counter_ns()
    start_cpu = time.process_time_ns()
    sim = CORE(seed=seed, predators=True)
    sim.policy_init(0, CONFIG)
    steps, peak, interface, policy, engine = sim._engine.run_policy(3000.0, 3000.0, True)
    cpu = time.process_time_ns() - start_cpu
    wall = time.perf_counter_ns() - start_wall
    return {
        "seed": seed,
        "score": sim.env.score,
        "survival": sim.env.time,
        "steps": steps,
        "peak": peak,
        "ns_interface": interface,
        "ns_policy": policy,
        "ns_engine": engine,
        "ns_loop_cpu": cpu,
        "ns_loop_wall": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    sys.path.insert(0, str(root))
    global CORE, CONFIG
    from fastsim.fastpolicy import PolicySimulationCore

    CORE = PolicySimulationCore
    CONFIG = json.loads(pathlib.Path(args.config).read_text())
    if args.model:
        CONFIG = CONFIG["configs"][args.model]
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    started = time.monotonic()
    with Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(one, seeds))
    rows.sort(key=lambda row: row["seed"])
    ticks = sum(row["steps"] for row in rows)
    summary = {
        "games": len(rows),
        "mean_score": statistics.mean(row["score"] for row in rows),
        "mean_survival": statistics.mean(row["survival"] for row in rows),
        "mean_seconds_per_game": statistics.mean(row["ns_loop_wall"] for row in rows) / 1e9,
        "policy_us_per_tick": sum(row["ns_policy"] for row in rows) / ticks / 1000,
        "engine_us_per_tick": sum(row["ns_engine"] for row in rows) / ticks / 1000,
        "loop_cpu_us_per_tick": sum(row["ns_loop_cpu"] for row in rows) / ticks / 1000,
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}))


if __name__ == "__main__":
    main()

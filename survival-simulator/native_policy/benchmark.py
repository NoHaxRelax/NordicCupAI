import argparse
import json
import multiprocessing as mp
import pathlib
import time

import nightsim


def run_one(job):
    seed, params, horizon = job
    started = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=True)
    engine = sim._engine
    sim.step([])
    engine.policy_init(nightsim.seed_key(seed), params)
    engine.run_policy(horizon, horizon)
    return {
        "seed": seed,
        "simulated_seconds": round(engine.info()["time"], 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=930001)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--horizon", type=float, default=3000.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configs = json.loads(pathlib.Path(args.config).read_text())
    params = next(iter(configs.values()))
    jobs = [(args.seed_start + i, params, args.horizon) for i in range(args.games)]
    batch_started = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(run_one, jobs))
    result = {
        "games": args.games,
        "workers": args.workers,
        "batch_wall_seconds": round(time.perf_counter() - batch_started, 3),
        "runs": sorted(rows, key=lambda row: row["seed"]),
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

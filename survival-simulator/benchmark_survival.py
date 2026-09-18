"""Checkpointed predator-free survival checks over the full game horizon.

Example: python benchmark_survival.py --seeds 1001 1007 1042 --workers 3
Only public observations enter the policy; world counts and death details are
recorded by the evaluator exclusively for diagnosis.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import signal
import statistics
import time

from tuning_evaluation import atomic_json, configure_process, evaluate_seed, initialize_worker

configure_process()

from tune_policy import check_manifest, coordinator_lock, source_hash
from src.utils.controllers.expert_policy import load_config
from src.utils.controllers.global_planner import load_planner_config


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seeds", type=int, nargs="+", default=[1001, 1007, 1042])
    cli.add_argument("--seconds", type=float, default=3000.)
    cli.add_argument("--workers", type=int, default=3)
    cli.add_argument("--hours", type=float, default=2.)
    cli.add_argument("--output", type=Path, default=Path("runs/survival-validation"))
    cli.add_argument("--expert-config", type=Path)
    cli.add_argument("--policy", choices=("standard", "simple"), default=None)
    cli.add_argument("--planner-config", type=Path)
    args = cli.parse_args()
    if not 0 < args.seconds <= 3000 or args.workers < 1 or not 0 < args.hours < float("inf"):
        cli.error("Require 0 < seconds <= 3000, positive workers and finite positive hours")
    if len(args.seeds) != len(set(args.seeds)):
        cli.error("Seed list must not contain duplicates")
    expert = load_config(args.expert_config).model_dump(mode="json")
    if args.policy is not None:
        expert["policy_mode"] = args.policy
    expert["harvest"]["enabled"] = True
    planner = load_planner_config(args.planner_config).model_dump(mode="json")
    planner["biome_inference"]["adapt_to_wall_clock"] = False
    planner["draw_overlay"] = False
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()

    def request_stop(signum, frame):
        print("Saving active episodes; repeat this command to resume.", flush=True)
        stop.set()

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    deadline = time.monotonic() + args.hours * 3600
    try:
        with coordinator_lock(output):
            check_manifest(output / "manifest.json", dict(source_hash=source_hash(),
                expert=expert, planner=planner, seconds=args.seconds, seeds=args.seeds, predators_enabled=False))
            atomic_json(output / "expert_policy.json", expert)
            atomic_json(output / "global_planner.json", planner)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                    initializer=initialize_worker, initargs=(stop,)) as pool:
                jobs = {pool.submit(evaluate_seed, expert, planner, seed, args.seconds,
                        str(output / f"seed-{seed}.json"), deadline, 30.): seed for seed in args.seeds}
                results = []
                try:
                    for job in as_completed(jobs):
                        result = job.result()
                        if result is not None:
                            results.append(result)
                            print(f"Seed {result['seed']}: {result['seconds']:.1f}s, "
                                  f"score {result['score']:.2f}, population {result['population']}", flush=True)
                        if results:
                            atomic_json(output / "summary.json", dict(
                                completed=len(results), requested=len(args.seeds),
                                survivors=sum(not r["extinct"] for r in results),
                                mean_score=statistics.mean(r["score"] for r in results),
                                mean_survival_seconds=statistics.mean(r["seconds"] for r in results),
                                seeds=[{k: r[k] for k in ("seed", "seconds", "score", "population", "extinct")}
                                       for r in sorted(results, key=lambda r: r["seed"])]))
                finally:
                    stop.set()
            print(f"Saved survival results: {output}", flush=True)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()

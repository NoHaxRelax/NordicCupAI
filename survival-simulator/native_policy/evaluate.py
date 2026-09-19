#!/usr/bin/env python3
"""Reproducible multi-game evaluation for the C++ native policy.

The worker process only initializes and invokes the native policy.  Results are
appended to JSONL as games finish, so an interrupted batch can be resumed with
the same command without discarding completed games.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

# Each game owns one process/core; prevent numerical libraries from adding a
# second layer of worker threads. Preserve an explicit caller setting.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import nightsim


HERE = pathlib.Path(__file__).resolve().parent
SOURCE_FILES = (HERE / "evaluate.py", HERE / "nightsim" / "__init__.py") + tuple(
    sorted(path for path in (HERE / "nightsim").iterdir()
           if path.suffix in {".cpp", ".h", ".hpp"})
)
UNKNOWN_METRICS = {
    "deaths_by_role": "the native API does not expose policy role at death",
    "premature_captures_with_sprint_available":
        "the native API exposes neither capture events nor sprint availability at capture",
    "bait_gap_seconds_total": "the native API does not expose bait occupancy intervals",
    "bait_gap_seconds_max": "the native API does not expose bait occupancy intervals",
    "confirmed_predator_retention":
        "the native API has no confirmed-retention event; proximity is not treated as capture",
}

ADDITIVE_NATIVE_METRICS = {
    "total_premature_captures",
    "premature_captures_with_sprint_available",
    "intentional_delivery_sacrifices",
    "guide_attempts",
    "guide_deliveries",
    "fruit_eaten",
    "ripe_fruit_eaten",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args):
    result = subprocess.run(
        ["git", *args], cwd=HERE, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_params(config_path, config_name):
    configs = json.loads(config_path.read_text())
    if config_name is None:
        if len(configs) != 1:
            raise ValueError("config contains multiple variants; pass --config-name")
        config_name, params = next(iter(configs.items()))
    else:
        params = configs[config_name]
    if not isinstance(params, dict):
        raise ValueError("selected configuration must be a JSON object")
    return config_name, params


def run_one(job):
    seed, params, horizon = job
    started = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=True)
    engine = sim._engine
    sim.step([])
    engine.policy_init(nightsim.seed_key(seed), params)
    steps, peak_agents = engine.run_policy(horizon, horizon)
    info = engine.info()
    events = engine.pop_events()
    deaths = {"starvation": 0, "predator": 0}
    for event in events:
        if event[0] in deaths:
            deaths[event[0]] += 1

    trap = engine.dbg_trap()
    guide_attempts = None
    guide_deliveries = None
    if trap is not None:
        guide_deliveries = int(trap[6])
        guide_attempts = int(trap[8][0])

    simulated = float(info["time"])
    reached_horizon = simulated >= horizon - 1e-6
    native_evaluation = engine.evaluation() if hasattr(engine, "evaluation") else {}
    unknown = dict(UNKNOWN_METRICS)
    for name in tuple(unknown):
        if name in native_evaluation and native_evaluation[name] is not None:
            del unknown[name]
    return {
        "seed": seed,
        "status": "reached_horizon" if reached_horizon else "extinct",
        "score": float(info["score"]),
        "simulated_seconds": simulated,
        "wall_seconds": time.perf_counter() - started,
        "reached_horizon": reached_horizon,
        "steps": int(steps),
        "peak_agents": int(peak_agents),
        "final_agents": len(engine.agents()),
        "total_agents_created": int(info["next_agent_id"]),
        "deaths_by_cause": deaths,
        "guide_attempts": guide_attempts,
        "guide_deliveries": guide_deliveries,
        "native_evaluation": native_evaluation,
        "metrics_unknown": unknown,
    }


def percentile(values, p):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_mean_ci(values, bootstrap_seed, samples=20000):
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(bootstrap_seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def describe(values, bootstrap_seed):
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "mean_95pct_bootstrap_ci": bootstrap_mean_ci(values, bootstrap_seed),
    }


def sum_numeric_trees(values):
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return sum(values)
    if all(isinstance(value, dict) or value == 0 for value in values) and any(
            isinstance(value, dict) for value in values):
        mappings = [value if isinstance(value, dict) else {} for value in values]
        keys = set().union(*(value.keys() for value in mappings))
        result = {}
        for key in sorted(keys):
            child = sum_numeric_trees([value.get(key, 0) for value in mappings])
            if child is None:
                return None
            result[key] = child
        return result
    return None


def summarize_native_evaluation(rows, bootstrap_seed):
    """Aggregate counters and describe per-game measurements by their semantics."""
    result = {"totals": {}, "distributions": {}, "definitions": {}}
    fields = set().union(*(row["native_evaluation"].keys() for row in rows))
    for field in sorted(fields):
        values = [row["native_evaluation"].get(field) for row in rows]
        if field == "ripe_fraction":
            present = [value for value in values if value is not None]
            if present:
                stats = describe(present, bootstrap_seed + len(result["distributions"]))
                stats["missing_count"] = len(values) - len(present)
                result["distributions"][field] = stats
            continue
        if any(value is None for value in values):
            continue
        if field == "metric_definitions":
            if all(value == values[0] for value in values[1:]):
                result["definitions"] = values[0]
        elif field in ADDITIVE_NATIVE_METRICS and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result["totals"][field] = sum(values)
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result["distributions"][field] = describe(values, bootstrap_seed + len(result["distributions"]))
        elif all(isinstance(value, bool) for value in values):
            true_count = sum(values)
            result["distributions"][field] = {
                "count": len(values), "true_count": true_count,
                "true_rate": true_count / len(values),
            }
        elif all(isinstance(value, dict) for value in values):
            total = sum_numeric_trees(values)
            if total is not None:
                result["totals"][field] = total
    return result


def histogram_svg(values, path, bins=10):
    low, high = min(values), max(values)
    width, height, pad = 760, 360, 50
    if high == low:
        counts, edges = [len(values)], [low, high]
    else:
        step = (high - low) / bins
        counts = [0] * bins
        for value in values:
            counts[min(bins - 1, int((value - low) / step))] += 1
        edges = [low + i * step for i in range(bins + 1)]
    max_count = max(counts)
    bar_w = (width - 2 * pad) / len(counts)
    bars = []
    for i, count in enumerate(counts):
        bar_h = (height - 2 * pad) * count / max_count
        x, y = pad + i * bar_w, height - pad - bar_h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="#4776b4"/>')
        bars.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="12">{count}</text>')
    labels = (
        f'<text x="{pad}" y="{height - 14}" font-size="12">{edges[0]:.2f}</text>'
        f'<text x="{width - pad}" y="{height - 14}" text-anchor="end" font-size="12">{edges[-1]:.2f}</text>'
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="16">Final score histogram (n={len(values)})</text>'
    )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>{"".join(bars)}{labels}</svg>\n'
    path.write_text(svg)


def read_completed(path):
    rows = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row["seed"])
        if seed in rows:
            raise ValueError(f"duplicate seed {seed} in {path}:{line_number}")
        rows[seed] = row
    return rows


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def manifest_identity(manifest):
    """Fields that must match before completed games may be reused."""
    return {key: manifest[key] for key in
            ("git_commit", "config", "run", "source_sha256", "runtime")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--config-name")
    parser.add_argument("--seed-start", type=int, required=True,
                        help="first seed; the batch uses the contiguous range [start, start + games)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--horizon", type=float, default=3000.0)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1 or args.workers > 10 or args.horizon <= 0:
        parser.error("games and horizon must be positive; workers must be in 1..10")

    config_path = args.config.resolve()
    config_name, params = load_params(config_path, args.config_name)
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "games.jsonl"
    completed = read_completed(results_path)
    unexpected = set(completed) - set(seeds)
    if unexpected:
        raise ValueError(f"existing results contain seeds outside requested batch: {sorted(unexpected)[:5]}")

    extension = pathlib.Path(nightsim._engine.__file__).resolve()
    python_executable = pathlib.Path(sys.executable).resolve()
    numpy_module = pathlib.Path(nightsim._np.__file__).resolve()
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "config": {"path": str(config_path), "name": config_name, "sha256": sha256(config_path)},
        "run": {"seed_start": args.seed_start, "games": args.games, "seeds": seeds,
                "horizon": args.horizon, "workers": args.workers,
                "bootstrap_seed": args.bootstrap_seed},
        "source_sha256": {str(path.relative_to(HERE)): sha256(path) for path in SOURCE_FILES},
        "runtime": {"python": sys.version, "implementation": platform.python_implementation(),
                    "platform": platform.platform(), "machine": platform.machine(),
                    "cpu_count": os.cpu_count(), "numpy": nightsim._np.__version__,
                    "python_executable": str(python_executable),
                    "python_executable_sha256": sha256(python_executable),
                    "numpy_module": str(numpy_module), "numpy_module_sha256": sha256(numpy_module),
                    "native_extension": str(extension), "native_extension_sha256": sha256(extension)},
        "metric_availability": {name: {"status": "unknown", "reason": reason}
                                for name, reason in UNKNOWN_METRICS.items()},
    }
    manifest_path = args.out_dir / "manifest.json"
    if completed:
        if not manifest_path.exists():
            raise ValueError("games.jsonl exists without manifest.json; refusing to mix results")
        previous_manifest = json.loads(manifest_path.read_text())
        if manifest_identity(previous_manifest) != manifest_identity(manifest):
            raise ValueError("existing manifest does not match this source/config/runtime/run")
        manifest = previous_manifest
    else:
        atomic_json(manifest_path, manifest)

    pending = [(seed, params, args.horizon) for seed in seeds if seed not in completed]
    batch_started = time.perf_counter()
    if pending:
        with results_path.open("a", buffering=1) as output, mp.Pool(args.workers) as pool:
            for row in pool.imap_unordered(run_one, pending, chunksize=1):
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
                completed[row["seed"]] = row
                print(f'{len(completed)}/{args.games} seed={row["seed"]} score={row["score"]:.3f} sim={row["simulated_seconds"]:.1f}', flush=True)

    rows = [completed[seed] for seed in seeds]
    scores = [row["score"] for row in rows]
    durations = [row["simulated_seconds"] for row in rows]
    batch_unknown = {name: reason for name, reason in UNKNOWN_METRICS.items()
                     if any(name in row["metrics_unknown"] for row in rows)}
    summary = {
        "games": len(rows),
        "score": describe(scores, args.bootstrap_seed),
        "simulated_seconds": describe(durations, args.bootstrap_seed + 1),
        "reached_horizon": sum(row["reached_horizon"] for row in rows),
        "reached_horizon_rate": sum(row["reached_horizon"] for row in rows) / len(rows),
        "deaths_by_cause": {cause: sum(row["deaths_by_cause"][cause] for row in rows)
                            for cause in ("starvation", "predator")},
        "guide_attempts": sum(row["native_evaluation"].get("guide_attempts", row["guide_attempts"] or 0)
                              for row in rows),
        "guide_deliveries": sum(row["native_evaluation"].get("guide_deliveries", row["guide_deliveries"] or 0)
                                for row in rows),
        "native_evaluation": summarize_native_evaluation(rows, args.bootstrap_seed + 100),
        "batch_wall_seconds_this_invocation": time.perf_counter() - batch_started,
        "metrics_unknown": batch_unknown,
    }
    atomic_json(args.out_dir / "summary.json", summary)
    histogram_svg(scores, args.out_dir / "score-histogram.svg")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["results_sha256"] = sha256(results_path)
    manifest["metric_availability"] = {
        name: ({"status": "unknown", "reason": reason} if name in batch_unknown
               else {"status": "available", "source": "engine.evaluation"})
        for name, reason in UNKNOWN_METRICS.items()
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

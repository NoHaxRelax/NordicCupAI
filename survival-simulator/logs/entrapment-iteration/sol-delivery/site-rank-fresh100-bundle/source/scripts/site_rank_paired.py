"""Paired native evaluation of default versus frozen static site ranking."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rank(site):
    """Frozen development score: boundary support, gap, overlap."""
    return (bool(site.get("boundary_indices")), float(site["gap"]), float(site["overlap"]))


def inventory(seed):
    import guide_lab as lab
    from models.entrapment.observed_trap_sites import available_sites
    core = lab.SimulationCore(seed=seed, starting_agents=0, starting_predators=0)
    env = core.env
    static = dict(width=env.width, height=env.height,
                  obstacles=[(o.x, o.y, o.width, o.height) for o in env.obstacles])
    return available_sites(static)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--encounters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026091904)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.maps, args.encounters, args.workers) < 1:
        parser.error("Counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ | {key: "1" for key in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    sources = sorted(path for directory in ("src", "models", "scripts")
                     for path in (ROOT / directory).rglob("*") if path.is_file())
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    rng = random.Random(args.seed)
    map_seeds = rng.sample(range(2**31), args.maps)
    selections, jobs = [], []
    for map_index, seed in enumerate(map_seeds):
        sites = inventory(seed)
        encounters = [rng.randrange(2**31) for _ in range(args.encounters)]
        if not sites:
            selections.append(dict(map_index=map_index, seed=seed, eligible_sites=0,
                                   encounters=encounters, default=None, candidate=None,
                                   choice="no_site"))
            continue
        default_index = 0
        candidate_index = max(range(len(sites)), key=lambda index: rank(sites[index]))
        choice = "shared" if candidate_index == default_index else "paired"
        selections.append(dict(map_index=map_index, seed=seed, eligible_sites=len(sites),
                               encounters=encounters, choice=choice,
                               default=dict(site_index=default_index, goal=sites[default_index]["goal"],
                                            score=rank(sites[default_index])),
                               candidate=dict(site_index=candidate_index, goal=sites[candidate_index]["goal"],
                                              score=rank(sites[candidate_index]))))
        for encounter_index, encounter_seed in enumerate(encounters):
            indices = [default_index] if choice == "shared" else [default_index, candidate_index]
            for site_index in indices:
                jobs.append(dict(index=len(jobs), map_index=map_index, seed=seed,
                                 encounter_index=encounter_index, encounter_seed=encounter_seed,
                                 site_index=site_index,
                                 arm="shared" if choice == "shared" else
                                     ("default" if site_index == default_index else "candidate")))
    manifest = dict(config=dict(maps=args.maps, encounters=args.encounters, seed=args.seed,
                                workers=args.workers, timeout=args.timeout,
                                policy="native guide_multi: 30 preloaded + one newcomer",
                                candidate_rank="boundary support, then gap, then overlap"),
                    map_seeds=map_seeds, jobs=jobs, source_hashes=source_hashes)
    manifest_path = args.output / "manifest.json"
    selection_path = args.output / "selection.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        comparable = {key: manifest[key] for key in ("map_seeds", "jobs", "source_hashes")}
        if any(previous.get(key) != value for key, value in comparable.items()):
            parser.error("Existing output has different seeds, jobs, or frozen source")
    else:
        write(manifest_path, manifest)
        write(selection_path, selections)
    plan = dict(maps=args.maps, all_map_attempts_per_arm=args.maps * args.encounters,
                no_site_maps=sum(row["choice"] == "no_site" for row in selections),
                single_site_maps=sum(row["eligible_sites"] == 1 for row in selections),
                shared_choice_maps=sum(row["choice"] == "shared" for row in selections),
                paired_choice_maps=sum(row["choice"] == "paired" for row in selections),
                native_jobs=len(jobs), deduplicated_jobs=args.maps * args.encounters * 2 - len(jobs))
    write(args.output / "plan.json", plan)
    print(json.dumps(plan), flush=True)
    if args.prepare_only:
        return

    def execute(job):
        folder = args.output / f"case-{job['index']:04d}"
        folder.mkdir(exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        command = [sys.executable, str(ROOT / "scripts/guide_multi.py"), "--bulk", "--deliveries", "1",
                   "--seed", str(job["seed"]), "--encounter-seed", str(job["encounter_seed"]),
                   "--site", str(job["site_index"]), "--output", str(folder)]
        started = time.monotonic()
        try:
            with (folder / "worker.log").open("w") as handle:
                process = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                         timeout=args.timeout)
            paths = list(folder.glob("multi-*/summary.json"))
            result = json.loads(paths[0].read_text()) if paths else {
                "outcome": "worker_error", "returncode": process.returncode}
            if result["outcome"] == "running": result["outcome"] = "worker_error"
        except subprocess.TimeoutExpired:
            result = {"outcome": "worker_timeout"}
        result.update(job, wall_seconds=round(time.monotonic() - started, 3))
        write(result_path, result)
        return result

    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            results.append(future.result())
            write(args.output / "results.json", sorted(results, key=lambda row: row["index"]))
            if len(results) % 20 == 0 or len(results) == len(jobs):
                print(json.dumps(dict(completed=len(results), total=len(jobs),
                                      elapsed=round(time.monotonic() - started, 1))), flush=True)

    by_key = {(row["map_index"], row["encounter_index"], row["site_index"]): row
              for row in results}
    arm_rows = {"default": [], "candidate": []}
    paired = []
    for selection in selections:
        for encounter_index in range(args.encounters):
            if selection["choice"] == "no_site":
                default = candidate = {"outcome": "no_usable_bait_site"}
            else:
                default = by_key[(selection["map_index"], encounter_index,
                                  selection["default"]["site_index"])]
                candidate = by_key[(selection["map_index"], encounter_index,
                                    selection["candidate"]["site_index"])]
            arm_rows["default"].append(default)
            arm_rows["candidate"].append(candidate)
            paired.append(dict(map_index=selection["map_index"], encounter_index=encounter_index,
                               default=default["outcome"], candidate=candidate["outcome"]))
    report = dict(plan=plan, arms={}, paired=dict(
        wins=sum(row["candidate"] == "delivery_pass" and row["default"] != "delivery_pass" for row in paired),
        losses=sum(row["default"] == "delivery_pass" and row["candidate"] != "delivery_pass" for row in paired),
        ties=sum((row["default"] == "delivery_pass") == (row["candidate"] == "delivery_pass") for row in paired)))
    for arm, rows in arm_rows.items():
        successes = sum(row["outcome"] == "delivery_pass" for row in rows)
        report["arms"][arm] = dict(successes=successes, all_map_attempts=len(rows),
                                    all_map_rate=successes / len(rows),
                                    outcomes={outcome: sum(row["outcome"] == outcome for row in rows)
                                              for outcome in sorted({row["outcome"] for row in rows})})
    write(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

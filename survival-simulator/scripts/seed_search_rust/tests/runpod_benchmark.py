"""Linux benchmark: all allocated vCPUs, measured CPU use and native replay.

Pass the allocation returned by Runpod as --threads, not the physical host count.
Writes evidence incrementally; never treats a cross-platform hash mismatch as a match.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from search_survey_rust import DEFAULT_BINARY, run
from seed_survey_probe import collect, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 256:
        parser.error("Choose a fresh output and 1..256 allocated threads")
    input_path = ROOT / "docs/seed_survey_heldout_2026-09-19.json"
    document = json.loads(input_path.read_text())
    row = document["targets"][0]
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.processor())
    sources = [ROOT / "scripts/seed_search_rust" / name for name in
               ("Cargo.toml", "Cargo.lock", "src/lib.rs", "src/main.rs", "tests/runpod_benchmark.py")]
    data = dict(complete=False, pod_id=args.pod_id, allocated_vcpus=args.threads,
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                platform=platform.platform(), python=platform.python_version(), cpu=cpu_model,
                host_logical_cpus=os.cpu_count(), affinity_cpus=len(os.sched_getaffinity(0)),
                cgroup_cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip()
                    if Path("/sys/fs/cgroup/cpu.max").exists() else None,
                rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
                sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                scans=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    save()
    for count, threads in [(1_000_000, 1)] + [(10_000_000, args.threads)] * 3:
        options = argparse.Namespace(input=input_path, binary=DEFAULT_BINARY, target=None, checkpoint=None,
            start=0, count=count, threads=threads, max_results=100000, verify=False, max_native_worlds=20, output=None)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        measured = run(options)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_seconds = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        measured["child_cpu_seconds"] = cpu_seconds
        measured["average_busy_cpu_threads"] = cpu_seconds / measured["elapsed_seconds"]
        assert measured["complete"] and not measured["survivors_truncated"]
        data["scans"].append(measured)
        save()
        print(json.dumps({"count":count, "threads":threads, "seconds":measured["elapsed_seconds"],
                          "average_busy_threads":measured["average_busy_cpu_threads"],
                          "survivors":measured["survivors"]}), flush=True)
    candidates = data["scans"][-1]["survivors"]
    assert all(scan["survivors"] == candidates for scan in data["scans"][1:])
    if len(candidates) > 20:
        raise RuntimeError("Too many candidates for native verification")
    started = time.perf_counter()
    data["windows_record_native_verification"] = [verify(seed, row) for seed in candidates]
    data["windows_record_verification_seconds"] = time.perf_counter() - started
    save()

    # Windows and Linux libm can differ in the last bits of observation floats.
    # A separately collected Linux survey tests exact replay on the same platform.
    with contextlib.redirect_stdout(sys.stderr):
        local_row, settings = collect(78431, 30.)
    local_input = args.output.with_name("linux_survey.json")
    local_input.write_text(json.dumps(dict(targets=[local_row], survey_settings=settings), allow_nan=False))
    local_options = argparse.Namespace(input=local_input, binary=DEFAULT_BINARY, target=None, checkpoint=None,
        start=0, count=10_000_000, threads=args.threads, max_results=100000,
        verify=True, max_native_worlds=20, output=None)
    data["linux_survey_collection_wall_seconds"] = local_row["collection_wall_seconds"]
    data["linux_survey_search_and_verification"] = run(local_options)
    data["median_ten_million_seconds"] = statistics.median(scan["elapsed_seconds"] for scan in data["scans"][1:])
    data["complete"] = True
    save()
    assert data["linux_survey_search_and_verification"]["unique_verified_seed_in_range"] == 78431
    print(json.dumps({"complete":True, "median_ten_million_seconds":data["median_ten_million_seconds"],
                      "linux_verified_seed":78431}), flush=True)


if __name__ == "__main__":
    main()

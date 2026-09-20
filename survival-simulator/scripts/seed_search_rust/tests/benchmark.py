"""Measure sequential 1M/one-thread and 10M/all-thread scans, then native replay.

Run from survival-simulator after building the release binary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from search_survey_rust import DEFAULT_BINARY, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "docs/seed_survey_heldout_2026-09-19.json")
    parser.add_argument("--target", type=int)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--threads", type=int, help="10M worker count; default is available CPUs")
    parser.add_argument("--cpu-name", default=platform.processor())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    paths = [ROOT / "scripts/seed_search_rust" / name for name in (
        "Cargo.toml", "Cargo.lock", "src/lib.rs", "src/main.rs", "tests/python_vectors.json", "tests/benchmark.py")]
    paths += [ROOT / "scripts/search_survey_rust.py"]
    result = dict(complete=False, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  platform=platform.platform(), cpu=args.cpu_name, logical_cpus=os.cpu_count(),
                  sources={str(path.relative_to(ROOT)).replace("\\", "/"):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, runs=[])
    for count, threads, verify in [(1_000_000, 1, False), (10_000_000, args.threads, True)]:
        options = argparse.Namespace(input=args.input, binary=args.binary, target=args.target,
            checkpoint=None, start=0, count=count, threads=threads, max_results=100_000,
            verify=verify, max_native_worlds=20, output=None)
        measured = run(options)
        result["runs"].append(measured)
        print(json.dumps({"count":count, "threads":measured["threads"],
                          "elapsed_seconds":measured["elapsed_seconds"],
                          "survivors":measured["survivors"],
                          "verified_seeds":measured["verified_seeds"],
                          "total_wall_seconds":measured["total_wall_seconds"]}), flush=True)
    result["complete"] = True
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["runs"][-1]["verification_complete"] else 2


if __name__ == "__main__":
    sys.exit(main())

"""Run the Rust prefix scanner and optionally verify survivors in the native world.

Consumes seed_survey_probe JSON, or the scanner's standalone samples format.
--target only selects an existing survey record; it never restricts the search.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "scripts/seed_search_rust/target/release" / (
    "seed-search.exe" if os.name == "nt" else "seed-search")


def survey_record(document, target):
    rows = document.get("targets", [])
    if target is None and len(rows) == 1:
        return rows[0]
    for row in rows:
        if row.get("target_seed") == target:
            return row
    raise ValueError("Native verification requires a survey record; select one with --target")


def validate_replay(row):
    hashes, actions = row.get("public_frame_sha256"), row.get("action_log")
    if not isinstance(hashes, list) or not hashes or not isinstance(actions, list) or len(actions) != len(hashes) - 1:
        raise ValueError("Survey must contain one hash per frame and one action list per intervening tick")
    if any(not isinstance(value, str) or len(value) != 64 or
           any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("Survey contains an invalid public-frame hash")
    if not isinstance(row.get("founder_headings"), list) or not isinstance(row.get("rock_lengths"), list):
        raise ValueError("Survey must contain founder_headings and rock_lengths lists")
    frames = row.get("public_frames")
    if frames is not None:
        from seed_joint_constraints_probe import canonical_frame
        if not isinstance(frames, list) or len(frames) != len(hashes) or any(
                hashlib.sha256(canonical_frame(frame).encode()).hexdigest() != expected
                for frame, expected in zip(frames, hashes)):
            raise ValueError("Saved public frames do not match their recorded hashes")
    if row.get("schema_version", 1) >= 2:
        if frames is None or not isinstance(row.get("seed_evidence"), dict):
            raise ValueError("Version 2 surveys require public_frames and seed_evidence")
        from seed_survey_evidence import extract_evidence
        # Derived constraints must be reproducible without target_seed, audit or
        # hidden state. Reject edited/stale evidence before it can reject seeds.
        if extract_evidence(frames, actions) != row["seed_evidence"]:
            raise ValueError("Seed evidence does not match the saved public history; rebuild it")


def run(args):
    if args.output and args.output.exists():
        raise ValueError("Output already exists; choose a new filename")
    if args.max_native_worlds < 1:
        raise ValueError("--max-native-worlds must be positive")
    if not args.binary.is_file():
        raise ValueError(f"Build the Rust release binary first: {args.binary}")
    started = time.perf_counter()
    source = args.input.read_bytes()
    document = json.loads(source)
    row = None
    if args.verify:
        row = survey_record(document, args.target)
        validate_replay(row)
    command = [str(args.binary.resolve()), "--input", str(args.input.resolve()),
               "--start", str(args.start), "--count", str(args.count),
               "--max-results", str(args.max_results)]
    for name in ("target", "checkpoint", "threads"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name}", str(value)])
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise ValueError(process.stderr.strip() or "Rust scanner failed")
    if args.input.read_bytes() != source:
        raise ValueError("Input changed during search; repeat with an immutable survey file")
    result = json.loads(process.stdout)
    result["input_sha256"] = hashlib.sha256(source).hexdigest()
    result["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    result["native_verification"] = []
    result["verified_seeds"] = []
    result["verification_complete"] = False
    if args.verify:
        if result["survivors_truncated"]:
            result["verification_skipped"] = "Candidate output was truncated; increase --max-results"
        elif result["survivor_count"] > args.max_native_worlds:
            result["verification_skipped"] = "Native-world cap exceeded; collect more evidence or raise the cap"
        else:
            # Simulator imports/logging must not corrupt JSON on stdout.
            with contextlib.redirect_stdout(sys.stderr):
                from seed_survey_probe import verify
                from seed_recovery_probe import source_manifest
                result["verification_sources"] = source_manifest()
                for name in ("seed_survey_probe.py", "seed_survey_evidence.py", "search_survey_rust.py",
                             "seed_joint_constraints_probe.py", "seed_biome_prefix_probe.py"):
                    result["verification_sources"][f"scripts/{name}"] = hashlib.sha256(
                        (ROOT / "scripts" / name).read_bytes()).hexdigest()
                verification_started = time.perf_counter()
                for seed in result["survivors"]:
                    checked = verify(seed, row)
                    result["native_verification"].append(checked)
                    print(json.dumps({"native_verification": checked}), file=sys.stderr, flush=True)
            result["native_verification_seconds"] = time.perf_counter() - verification_started
            result["verification_complete"] = True
            result["verified_seeds"] = [item["seed"] for item in result["native_verification"]
                                        if item["public_replay_match"] is True]
            result["verification"] = ("Checked headings, rock dimensions, available spawn/rock/tree/biome constraints "
                                      "and every recorded public frame.")
    result["total_wall_seconds"] = time.perf_counter() - started
    result["unique_verified_seed_in_range"] = (result["verified_seeds"][0]
        if result["verification_complete"] and len(result["verified_seeds"]) == 1 else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--target", type=int)
    parser.add_argument("--checkpoint", type=float)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10_000_000)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-results", type=int, default=100_000)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-native-worlds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        if args.verify and not result["verification_complete"]:
            print(result["verification_skipped"], file=sys.stderr)
            return 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())

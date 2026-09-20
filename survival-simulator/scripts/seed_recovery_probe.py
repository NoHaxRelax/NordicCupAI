"""Test seed identification within an explicit finite catalog using public geometry.

This is an offline feasibility probe, not a general 32-bit seed inverter.
World generation and the first empty tick use the unmodified native simulator.
The matching functions only receive public observations and candidate signatures.

Example, from survival-simulator:
    python scripts/seed_recovery_probe.py --output docs/seed_recovery_probe.json
"""

import argparse
from bisect import bisect_left
import gc
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def distinct(values, tolerance):
    result = []
    for value in sorted(values):
        if not result or value - result[-1] > tolerance:
            result.append(value)
    return result


def public_features(agent_states, tolerance=1e-6):
    """No seed, world coordinates, simulator instance, or RNG state is used."""
    lengths, tree_distances = [], []
    for state in agent_states:
        trees = []
        for observation in state["observations"]:
            kind = observation["type"].lower()
            if kind == "edge":
                first, second = observation["coords"]
                length = math.dist(first, second)
                # Native generated rock sides are 30..100. Boundary thickness
                # is exactly 30; its other dimensions exceed this interval.
                if 30 + tolerance < length < 100 - tolerance:
                    lengths.append(length)
            elif kind == "tree":
                distance, angle = observation["distance"], observation["angle"]
                trees.append((distance * math.cos(angle), distance * math.sin(angle)))
        # Compare pairs within one observer's frame, never across unknown poses.
        tree_distances.extend(math.dist(a, b) for a, b in combinations(trees, 2))
    return dict(rock_lengths=distinct(lengths, tolerance),
                tree_pair_distances=distinct(tree_distances, tolerance))


def match_candidates(observed, catalog, feature, tolerance=1e-6):
    """Return all compatible catalog entries; empty evidence yields no inference."""
    if not observed:
        return []
    matches = []
    for candidate in catalog:
        reference = candidate[feature]
        compatible = True
        for value in observed:
            index = bisect_left(reference, value - tolerance)
            if index == len(reference) or reference[index] > value + tolerance:
                compatible = False
                break
        if compatible:
            matches.append(candidate["seed"])
    return matches


def native_case(seed, observe=False):
    from src.core import SimulationCore

    start = time.perf_counter()
    sim = SimulationCore(seed=seed)
    generation_seconds = time.perf_counter() - start
    lengths = [value for rock in sim.env.obstacles for value in (rock.width, rock.height)
               if 30 + 1e-6 < value < 100 - 1e-6]
    trees = [(tree.x, tree.y) for tree in sim.env.trees]
    signature = dict(seed=seed, generation_seconds=generation_seconds,
                     rock_lengths=distinct(lengths, 1e-6),
                     tree_pair_distances=distinct(
                         (math.dist(a, b) for a, b in combinations(trees, 2)), 1e-6))
    # In this first tick, agent observations are computed before new trees spawn.
    # Do not extend the tree matcher to later ticks without modeling tree changes.
    observations = sim.step([])["observations"] if observe else None
    del sim
    gc.collect()
    return signature, observations


def source_manifest():
    files = [Path(__file__).resolve(), *sorted((ROOT / "src").rglob("*.py"))]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def atomic_json(destination, value):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--targets", type=int, nargs="+", default=[3, 11, 20260919])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(len(values) != len(set(values)) for values in (args.candidates, args.targets)):
        parser.error("Candidate and target lists must each contain unique seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.candidates + args.targets):
        parser.error("This probe uses unsigned 32-bit seeds")
    if args.output.exists():
        parser.error("Output exists; choose a new file to preserve earlier evidence")

    result = dict(format="seed-recovery-feasibility-v1", complete=False,
                  python=platform.python_version(), platform=platform.platform(),
                  packages={name: version(name) for name in ("numpy", "scipy", "pygame", "shapely")},
                  sources=source_manifest(), tolerance=1e-6,
                  candidate_seeds=args.candidates, target_seeds=args.targets,
                  catalog=[], targets=[],
                  limitations=[
                      "Identifies candidates only within the supplied seed catalog.",
                      "Targets are independently regenerated native first-tick observations.",
                      "Matcher sees only public geometry; target seed is an evaluation label.",
                      "No RNG-state recovery, seed-space inversion or long-term prediction is attempted.",
                      "Tree-pair matching assumes the first empty tick, before newly spawned trees are observed.",
                  ])
    atomic_json(args.output, result)
    for seed in args.candidates:
        signature, _ = native_case(seed)
        result["catalog"].append(signature)
        atomic_json(args.output, result)
        print(f"Catalog seed {seed}: {signature['generation_seconds']:.3f}s", flush=True)

    for seed in args.targets:
        signature, observations = native_case(seed, observe=True)
        # Round-trip exactly the observation-shaped input a JSON endpoint sees.
        observations = json.loads(json.dumps(observations, allow_nan=False))
        features = public_features(observations)
        start = time.perf_counter()
        matches = {feature: match_candidates(values, result["catalog"], feature)
                   for feature, values in features.items()}
        query_ms = (time.perf_counter() - start) * 1000
        rounded = []
        for digits, tolerance in ((3, .002), (1, .2)):
            # Quantize observed coordinates/angles, not the derived fingerprint.
            def quantize(value):
                if isinstance(value, float):
                    return round(value, digits)
                if isinstance(value, list):
                    return [quantize(item) for item in value]
                if isinstance(value, dict):
                    return {key: quantize(item) for key, item in value.items()}
                return value
            rock_lengths = public_features(quantize(observations), tolerance)["rock_lengths"]
            rounded.append(dict(coordinate_decimals=digits, tolerance=tolerance,
                                distinct_lengths=len(rock_lengths),
                                matches=match_candidates(rock_lengths, result["catalog"],
                                                         "rock_lengths", tolerance)))
        row = dict(target_seed=seed, in_catalog=seed in args.candidates,
                   generation_seconds=signature["generation_seconds"],
                   public_features=features, matches=matches,
                   first_rock_matches=match_candidates(features["rock_lengths"][:1],
                                                       result["catalog"], "rock_lengths"),
                   query_ms=query_ms, rounded_coordinate_checks=rounded)
        result["targets"].append(row)
        atomic_json(args.output, result)
        print(json.dumps({key: value for key, value in row.items() if key != "public_features"}), flush=True)

    result["mean_candidate_generation_seconds"] = statistics.mean(
        row["generation_seconds"] for row in result["catalog"])
    result["complete"] = True
    atomic_json(args.output, result)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

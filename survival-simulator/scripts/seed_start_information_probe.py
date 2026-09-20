"""Measure the extra filtering from public starting-agent biome labels.

Run from survival-simulator:
    python scripts/seed_start_information_probe.py --output docs/start_probe.json

This finite-catalog experiment does not recover an unrestricted seed or RNG state.
It uses the unmodified native engine and independently regenerates target frames.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

from seed_recovery_probe import (
    atomic_json, match_candidates, native_case, public_features, source_manifest,
)


def biome_signature(states):
    """Keep each public label associated with its public agent ID."""
    return [[state["agent_id"], state["biome"]]
            for state in sorted(states, key=lambda state: state["agent_id"])]


def summarize(states):
    return [{
        "fields": {key: value for key, value in state.items() if key != "observations"},
        "observation_counts": dict(Counter(obs["type"] for obs in state["observations"])),
        "other_agents": [obs for obs in state["observations"] if obs["type"] == "Agent"],
    } for state in sorted(states, key=lambda state: state["agent_id"])]


def quantize(value, digits):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [quantize(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: quantize(item, digits) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--targets", type=int, nargs="+", default=[3, 11, 20260919])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file to preserve earlier evidence")
    if any(len(values) != len(set(values)) for values in (args.candidates, args.targets)):
        parser.error("Candidate and target lists must each contain unique seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.candidates + args.targets):
        parser.error("This probe uses unsigned 32-bit seeds")

    sources = source_manifest()
    sources["scripts/seed_start_information_probe.py"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    result = dict(
        complete=False, python=platform.python_version(), sources=sources,
        candidate_seeds=args.candidates, target_seeds=args.targets,
        method="Native first empty tick; JSON-round-tripped public fields; no actions.",
        limitations=[
            "Catalog filtering only; no algebraic seed or state recovery.",
            "Known-target labels never enter the matcher.",
            "Five biome labels are correlated categorical observations.",
            "Rounded-coordinate check rounds edges only and keeps biome strings exact.",
            "Observation counts may contain repeated edges and are not independent draws.",
        ], catalog=[], targets=[],
    )
    atomic_json(args.output, result)
    for seed in args.candidates:
        signature, states = native_case(seed, observe=True)
        states = json.loads(json.dumps(states, allow_nan=False))
        signature.update(biomes=biome_signature(states), public_agents=summarize(states))
        result["catalog"].append(signature)
        atomic_json(args.output, result)
        print(f"Catalog seed {seed}: {signature['biomes']}", flush=True)

    for seed in args.targets:
        _, states = native_case(seed, observe=True)
        states = json.loads(json.dumps(states, allow_nan=False))
        biomes = biome_signature(states)
        biome_matches = [candidate["seed"] for candidate in result["catalog"]
                         if candidate["biomes"] == biomes]
        checks = []
        for digits, tolerance in ((None, 1e-6), (1, .2)):
            observed = states if digits is None else quantize(states, digits)
            lengths = public_features(observed, tolerance)["rock_lengths"]
            rock_matches = match_candidates(lengths, result["catalog"],
                                            "rock_lengths", tolerance)
            # No observed rocks means no rock constraint, not a rejection.
            combined = [value for value in biome_matches
                        if not lengths or value in rock_matches]
            checks.append(dict(coordinate_decimals=digits, tolerance=tolerance,
                               rock_evidence_count=len(lengths),
                               rock_matches=rock_matches,
                               combined_matches=combined))
        row = dict(target_seed=seed, biomes=biomes, biome_matches=biome_matches,
                   public_agents=summarize(states), checks=checks)
        result["targets"].append(row)
        atomic_json(args.output, result)
        print(json.dumps({key: value for key, value in row.items()
                          if key != "public_agents"}), flush=True)

    trait_keys = ("speed", "sprint_speed", "max_energy", "hearing_radius",
                  "vision_range", "vision_angle", "age", "energy")
    rows = result["catalog"] + result["targets"]
    result["distinct_public_values"] = {
        key: sorted({agent["fields"][key] for row in rows for agent in row["public_agents"]})
        for key in trait_keys
    }
    result["distinct_catalog_biome_signatures"] = len({
        tuple(tuple(pair) for pair in row["biomes"]) for row in result["catalog"]})
    result["complete"] = True
    atomic_json(args.output, result)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Test public boundary localization and a cheap necessary biome seed condition.

The collector sees public observations only. Hidden poses are used separately to
validate localization, never to supply samples to the candidate filter. Candidate
checks generate only the ten initial Voronoi sites/types, omitting all later work.
This remains enumeration and produces survivors, not recovered seeds.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

from seed_recovery_probe import atomic_json, distinct, public_features, source_manifest


WIDTH, HEIGHT, WALL = 1600, 1200, 30
LAND_TYPES = ("forest", "swamp", "desert", "grassland")


def boundary_pose(state, tolerance=1e-6):
    """Localize from directed whole inner-wall edges and known map dimensions."""
    proposals = []
    for obs in state["observations"]:
        if obs["type"] != "Edge":
            continue
        a, b = obs["coords"]
        length = math.dist(a, b)
        horizontal = abs(length - WIDTH) <= tolerance
        vertical = abs(length - HEIGHT) <= tolerance
        if not (horizontal or vertical):
            continue
        heading = (0 if horizontal else math.pi / 2) - math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(heading), math.sin(heading)
        rx, ry = c * a[0] - s * a[1], s * a[0] + c * a[1]
        # All native edge endpoints are ordered along positive x or positive y.
        # An interior observer sees the inner wall face, not its outer face.
        if horizontal:
            x, y = -rx, (WALL if ry < 0 else HEIGHT - WALL) - ry
        else:
            x, y = (WALL if rx < 0 else WIDTH - WALL) - rx, -ry
        if WALL - tolerance <= x <= WIDTH - WALL + tolerance and WALL - tolerance <= y <= HEIGHT - WALL + tolerance:
            proposals.append((x, y, heading))
    if not proposals:
        return None
    x, y, heading = proposals[0]
    if any(math.dist((x, y), proposal[:2]) > tolerance or
           abs(math.remainder(heading - proposal[2], 2 * math.pi)) > tolerance
           for proposal in proposals[1:]):
        return None
    return x, y, heading


def candidate_prefix(seed):
    rng = random.Random(seed)
    sites = [(rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)) for _ in range(10)]
    types = [rng.choice(LAND_TYPES) for _ in range(10)]
    return sites, types


def prefix_compatible(sites, types, samples):
    for sample in samples:
        # Rivers overwrite land; a river label supplies no land-type restriction.
        if sample["biome"] == "river":
            continue
        possible = set()
        # Preserve integer-pixel alternatives near rounding boundaries.
        for x, y in sample["possible_pixels"]:
            nearest = min(range(10), key=lambda i: (x - sites[i][0]) ** 2 + (y - sites[i][1]) ** 2)
            possible.add(types[nearest])
        if sample["biome"] not in possible:
            return False
    return True


def public_sample(state, pose):
    x, y, _ = pose
    xs = range(max(0, math.floor(x - 1e-6)), min(WIDTH - 1, math.floor(x + 1e-6)) + 1)
    ys = range(max(0, math.floor(y - 1e-6)), min(HEIGHT - 1, math.floor(y + 1e-6)) + 1)
    return dict(agent_id=state["agent_id"], biome=state["biome"], x=x, y=y,
                possible_pixels=[[ix, iy] for ix in xs for iy in ys])


def collect(seed):
    from src.core import SimulationCore

    started = time.perf_counter()
    sim = SimulationCore(seed=seed)
    native_seconds = time.perf_counter() - started
    lengths, samples, frames, errors = [], {}, [], []
    payload = sim.step([])
    for frame in range(7):
        states = json.loads(json.dumps(payload["observations"], allow_nan=False))
        lengths.extend(public_features(states)["rock_lengths"])
        for state in states:
            pose = boundary_pose(state)
            if pose is None:
                continue
            samples[state["agent_id"]] = public_sample(state, pose)
            # Validation only. No hidden coordinates enter public_sample/filter.
            truth = sim.env.agents_dict[state["agent_id"]]
            position_error = math.dist(pose[:2], (truth.x, truth.y))
            heading_error = abs(math.remainder(pose[2] - truth.direction, 2 * math.pi))
            if position_error > 1e-6 or heading_error > 1e-6:
                raise AssertionError(f"Boundary localization failed: {position_error}, {heading_error}")
            errors.append(position_error)
        frames.append(dict(sim_time=payload["sim_time"],
                           distinct_rock_lengths=len(distinct(lengths, 1e-6)),
                           localized_agents=len(samples)))
        if frame < 6:
            action = SimpleNamespace(move_distance=0, move_direction=0,
                                     turn_angle=math.pi / 3, spawn_agent=False)
            payload = sim.step([(state["agent_id"], action) for state in states])
    result = dict(target_seed=seed, native_generation_seconds=native_seconds,
                  frames=frames, public_samples=list(samples.values()),
                  localization_checks=len(errors),
                  max_position_error=max(errors, default=None))
    del sim
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=int, default=[3, 11, 20260919])
    parser.add_argument("--scan-count", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    if not 0 < args.scan_count <= 2**32 or any(not 0 <= seed < 2**32 for seed in args.targets):
        parser.error("Use unsigned 32-bit seeds and a positive scan count up to 2**32")
    sources = source_manifest()
    sources["scripts/seed_biome_prefix_probe.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(complete=False, sources=sources, scan_count=args.scan_count, targets=[],
                  limitations=[
                      "Native default 1600x1200 map and 30-unit fixed walls assumed.",
                      "Seven frames: one empty tick plus six 60-degree stationary turns.",
                      "The filter receives only public boundary-localized biome samples.",
                      "Survivors still need river, geometry and full observation verification.",
                      "No broad-range seed inversion or recovery is demonstrated.",
                  ])
    atomic_json(args.output, result)
    for seed in args.targets:
        row = collect(seed)
        sites, types = candidate_prefix(seed)
        row["true_seed_passes_necessary_condition"] = prefix_compatible(sites, types, row["public_samples"])
        if not row["true_seed_passes_necessary_condition"]:
            raise AssertionError("True seed incorrectly rejected")
        print(json.dumps(row), flush=True)
        result["targets"].append(row)
        atomic_json(args.output, result)

    counts = [0] * len(result["targets"])
    started = time.perf_counter()
    for seed in range(args.scan_count):
        sites, types = candidate_prefix(seed)
        for i, row in enumerate(result["targets"]):
            if prefix_compatible(sites, types, row["public_samples"]):
                counts[i] += 1
    elapsed = time.perf_counter() - started
    result["prefix_scan_seconds_all_targets"] = elapsed
    result["candidate_prefixes_per_second"] = args.scan_count / elapsed
    for row, count in zip(result["targets"], counts):
        row["prefix_survivors"] = count
        row["prefix_survival_fraction"] = count / args.scan_count
        print(f"Target {row['target_seed']}: {count}/{args.scan_count} prefix survivors", flush=True)
    result["complete"] = True
    atomic_json(args.output, result)
    print(f"Scanned {args.scan_count} candidate prefixes for all targets in {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()

"""Combine public biomes, founder headings/positions, and observed rock edges.

Default experiment enumerates 100,000 seed prefixes, then checks native worlds
only for survivors. Source-specific directed edges reveal heading hypotheses;
boundary edges additionally reveal position. This is not algebraic seed recovery.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from seed_biome_prefix_probe import (
    boundary_pose, candidate_prefix, prefix_compatible, public_sample,
)
from seed_recovery_probe import atomic_json, distinct, public_features, source_manifest


ANGLE_TOLERANCE = 1e-8
POSITION_TOLERANCE = 1e-6
EDGE_TOLERANCE = 1e-6


def angle_close(a, b):
    return abs(math.remainder(a - b, 2 * math.pi)) <= ANGLE_TOLERANCE


def heading_hypotheses(observations, accumulated_turn=0):
    """Environment.edges, unlike Obstacle.edges, orders every edge in +x or +y."""
    hypotheses = None
    for observation in observations:
        if observation["type"] != "Edge":
            continue
        a, b = observation["coords"]
        if math.dist(a, b) < 1e-8:
            continue
        bearing = math.atan2(b[1] - a[1], b[0] - a[0])
        possibilities = [(axis - bearing - accumulated_turn) % (2 * math.pi)
                         for axis in (0, math.pi / 2)]
        hypotheses = possibilities if hypotheses is None else [
            value for value in hypotheses if any(angle_close(value, other) for other in possibilities)]
    return hypotheses


def canonical_frame(payload):
    # Entity observation iteration order is not stable across independent worlds.
    normalized = json.loads(json.dumps(payload, allow_nan=False))
    for state in normalized["observations"]:
        state["observations"] = sorted(state["observations"],
                                        key=lambda value: json.dumps(value, sort_keys=True))
    normalized["observations"].sort(key=lambda value: value["agent_id"])
    return json.dumps(normalized, sort_keys=True, allow_nan=False)


def rotation_actions(states):
    return [(state["agent_id"], SimpleNamespace(move_distance=0, move_direction=0,
                                               turn_angle=math.pi / 3, spawn_agent=False))
            for state in states]


def collect(target_seed):
    from src.core import SimulationCore

    sim = SimulationCore(seed=target_seed)
    # Oracle values below validate extraction; they do not enter the signature.
    original_headings = {agent.agent_id: agent.direction for agent in sim.env.agents}
    frames, headings, poses, samples, lengths = [], {}, {}, {}, []
    heading_errors = []
    payload = sim.step([])
    for frame_index in range(7):
        payload = json.loads(json.dumps(payload, allow_nan=False))
        frames.append(payload)
        states = payload["observations"]
        lengths.extend(public_features(states)["rock_lengths"])
        for state in states:
            agent_id = state["agent_id"]
            current = heading_hypotheses(state["observations"], frame_index * math.pi / 3)
            if current is not None:
                previous = headings.get(agent_id)
                merged = current if previous is None else [value for value in previous
                    if any(angle_close(value, other) for other in current)]
                if not merged:
                    raise AssertionError("Inconsistent public heading evidence")
                headings[agent_id] = merged
                error = min(abs(math.remainder(value - original_headings[agent_id], 2 * math.pi))
                            for value in merged)
                if error > ANGLE_TOLERANCE:
                    raise AssertionError("Public heading does not include true heading")
                heading_errors.append(error)
            pose = boundary_pose(state)
            if pose is not None:
                truth = sim.env.agents_dict[agent_id]
                if math.dist(pose[:2], (truth.x, truth.y)) > POSITION_TOLERANCE:
                    raise AssertionError("Public position does not match true position")
                poses[agent_id] = list(pose[:2])
                samples[agent_id] = public_sample(state, pose)
        if frame_index < 6:
            payload = sim.step(rotation_actions(states))

    world_edges = {}
    for frame_index, payload in enumerate(frames):
        for state in payload["observations"]:
            agent_id = state["agent_id"]
            if agent_id not in poses or len(headings.get(agent_id, [])) != 1:
                continue
            x, y = poses[agent_id]
            heading = headings[agent_id][0] + frame_index * math.pi / 3
            c, s = math.cos(heading), math.sin(heading)
            for observation in state["observations"]:
                if observation["type"] != "Edge":
                    continue
                a, b = observation["coords"]
                if not 30 + EDGE_TOLERANCE < math.dist(a, b) < 100 - EDGE_TOLERANCE:
                    continue
                edge = [[x + c * px - s * py, y + s * px + c * py] for px, py in (a, b)]
                key = tuple(round(value, 6) for point in edge for value in point)
                world_edges.setdefault(key, edge)
    signature = dict(
        founder_headings=[dict(agent_id=key, hypotheses=value) for key, value in sorted(headings.items())],
        founder_positions=[dict(agent_id=key, xy=value) for key, value in sorted(poses.items())],
        biome_samples=list(samples.values()), rock_lengths=distinct(lengths, EDGE_TOLERANCE),
        world_rock_edges=list(world_edges.values()),
    )
    validation = dict(heading_checks=len(heading_errors), max_heading_error=max(heading_errors, default=None),
                      public_frame_sha256=[hashlib.sha256(canonical_frame(frame).encode()).hexdigest()
                                           for frame in frames])
    del sim
    gc.collect()
    return signature, frames, validation


def verify_candidate(seed, signature, expected_frames):
    from src.core import SimulationCore

    started = time.perf_counter()
    sim = SimulationCore(seed=seed)
    generation_seconds = time.perf_counter() - started
    heading_match = all(any(angle_close(sim.env.agents_dict[item["agent_id"]].direction, value)
                           for value in item["hypotheses"]) for item in signature["founder_headings"])
    position_match = all(math.dist(item["xy"], (sim.env.agents_dict[item["agent_id"]].x,
                                              sim.env.agents_dict[item["agent_id"]].y)) <= POSITION_TOLERANCE
                         for item in signature["founder_positions"])
    dimensions = [value for rock in sim.env.obstacles for value in (rock.width, rock.height)]
    length_match = all(any(abs(value - dimension) <= EDGE_TOLERANCE for dimension in dimensions)
                       for value in signature["rock_lengths"])
    edge_match = all(any(max(math.dist(a, c), math.dist(b, d)) <= EDGE_TOLERANCE
                        for c, d in sim.env.edges) for a, b in signature["world_rock_edges"])
    joint_match = heading_match and position_match and length_match and edge_match
    replay_match = None
    if joint_match:
        replay_match = True
        payload = sim.step([])
        for frame_index, expected in enumerate(expected_frames):
            if canonical_frame(payload) != canonical_frame(expected):
                replay_match = False
                break
            if frame_index + 1 < len(expected_frames):
                payload = sim.step(rotation_actions(payload["observations"]))
    row = dict(seed=seed, generation_seconds=generation_seconds,
               heading_match=heading_match, position_match=position_match,
               rock_length_match=length_match, world_edge_match=edge_match,
               joint_match=joint_match, public_replay_match=replay_match)
    del sim
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--scan-start", type=int, default=0)
    parser.add_argument("--scan-count", type=int, default=100000)
    parser.add_argument("--max-native-worlds", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    if not 0 <= args.target < 2**32 or not 0 <= args.scan_start < 2**32 or not 0 < args.scan_count <= 2**32 - args.scan_start:
        parser.error("Use a positive scan count and unsigned 32-bit seed range")
    if args.max_native_worlds < 1:
        parser.error("--max-native-worlds must be positive")
    started = time.perf_counter()
    signature, frames, validation = collect(args.target)
    sources = source_manifest()
    for name in ("seed_biome_prefix_probe.py", "seed_joint_constraints_probe.py"):
        sources[f"scripts/{name}"] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    result = dict(complete=False, target_seed=args.target, scan_start=args.scan_start,
                  scan_count=args.scan_count, sources=sources, public_signature=signature,
                  extraction_validation=validation, candidates=[], limitations=[
                      "Bounded enumeration, not algebraic seed inversion.",
                      "One target; no claim of population-wide recovery rate.",
                      "Seven native frames with only known stationary turns.",
                      "Matching receives public features and known candidate worlds only.",
                      "Headings and edges are checked after native initialization.",
                      "Native-world cap prevents unexpectedly long runs on weak biome evidence.",
                  ])
    atomic_json(args.output, result)
    prefix_started = time.perf_counter()
    survivors = []
    for seed in range(args.scan_start, args.scan_start + args.scan_count):
        sites, types = candidate_prefix(seed)
        if prefix_compatible(sites, types, signature["biome_samples"]):
            survivors.append(seed)
    result["prefix_seconds"] = time.perf_counter() - prefix_started
    result["prefix_survivors"] = survivors
    atomic_json(args.output, result)
    print(f"Public headings: {[len(row['hypotheses']) for row in signature['founder_headings']]}; "
          f"positions: {len(signature['founder_positions'])}; world edges: {len(signature['world_rock_edges'])}", flush=True)
    print(f"Biome prefix: {len(survivors)}/{args.scan_count} survive in {result['prefix_seconds']:.3f}s", flush=True)
    if len(survivors) > args.max_native_worlds:
        result["stopped_reason"] = "Survivors exceed --max-native-worlds; no native verification performed"
        atomic_json(args.output, result)
        print(result["stopped_reason"], flush=True)
        return
    for i, seed in enumerate(survivors, 1):
        row = verify_candidate(seed, signature, frames)
        result["candidates"].append(row)
        atomic_json(args.output, result)
        if row["joint_match"] or i % 10 == 0 or i == len(survivors):
            print(f"Verified {i}/{len(survivors)}: " + json.dumps(row), flush=True)
    result["matches"] = {key: [row["seed"] for row in result["candidates"] if row[key]]
                         for key in ("heading_match", "position_match", "rock_length_match", "world_edge_match",
                                     "joint_match", "public_replay_match")}
    result["complete"] = True
    result["total_seconds"] = time.perf_counter() - started
    atomic_json(args.output, result)
    print(json.dumps(dict(matches=result["matches"], total_seconds=result["total_seconds"])), flush=True)


if __name__ == "__main__":
    main()

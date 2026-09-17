"""Synthetic public-DTO planner timing; this does not instantiate a simulator."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research"), str(ROOT)]

from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import Pose
from reliability_eval.policy_robust import SimpleChase


def main():
    policy = object.__new__(SimpleChase)
    policy.width = policy.height = 800.0
    policy.rects = [
        (0.0, 0.0, 800.0, 10.0), (0.0, 790.0, 800.0, 10.0),
        (0.0, 0.0, 10.0, 800.0), (790.0, 0.0, 10.0, 800.0),
        (320.0, 250.0, 20.0, 300.0),
    ]
    policy._terrain_codes = bytes([1]) * (800 * 800)
    policy._terrain_penalties = [1.0, 1.0, 0.5, 0.8, 0.3]
    policy.site = {"far": [500.0, 400.0]}
    policy.last_predator_point = (310.0, 400.0)
    policy._mpc_observation = {
        "type": "Predator", "distance": 55.0, "angle": 3.141592653589793,
        "rel_dir": 3.141592653589793,
    }
    pose = Pose((365.0, 400.0), 0.0)
    # Only public DTO fields consumed by the planner are provided.
    state = {"speed": 10.0, "sprint_speed": 20.0, "biome": "grassland"}
    samples = []
    result = None
    for _ in range(25):
        start = time.perf_counter()
        result = policy._escape_plan(state, pose, (500.0, 400.0))
        samples.append((time.perf_counter() - start) * 1000)
    print(json.dumps({
        "schema": "robust-policy-synthetic-dto-benchmark-v2",
        "zero_speed_preserves_heading": True,
        "native_simulation_used": False,
        "iterations": len(samples),
        "result": result,
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "model_cap": policy.MAX_PREDATOR_MODELS,
        "beam": policy.ESCAPE_BEAM,
        "depth": policy.ESCAPE_DEPTH,
    }, indent=2))


if __name__ == "__main__":
    main()

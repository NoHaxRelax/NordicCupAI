#!/usr/bin/env python3
"""Build additive validation-only probes from disjoint frame ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_range(value: str) -> tuple[str, int, int]:
    name, bounds = value.split(":", 1)
    start, end = bounds.split("-", 1)
    return name, int(start), int(end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--range", dest="ranges", action="append", required=True)
    args = parser.parse_args()

    source = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_plans = []

    for raw_range in args.ranges:
        shard, start, end = parse_range(raw_range)
        predictions = {
            frame: rows
            for frame, rows in source["predictions_by_frame"].items()
            if start <= int(frame) <= end and rows
        }
        plan = {
            "name": f"{args.name_prefix}-{shard}",
            "target": source["target"],
            "predictions_by_frame": predictions,
        }
        filename = f"range-{shard}.json"
        payload = json.dumps(plan, indent=2) + "\n"
        (args.output / filename).write_text(payload)
        manifest_plans.append(
            {
                "shard": shard,
                "file": filename,
                "name": plan["name"],
                "frame_range": [start, end],
                "prediction_count": sum(len(rows) for rows in predictions.values()),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )

    manifest = {
        "description": "Additive frame-range validation-only diagnostic probes.",
        "plans": manifest_plans,
        "source_plan": str(args.plan),
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

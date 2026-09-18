#!/usr/bin/env python3
"""Merge class-disjoint shard plans into one complete validation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    source_manifest = json.loads((args.shard_bundle / "manifest.json").read_text())
    merged: dict[str, list[dict]] = {}
    source_files = []
    for entry in source_manifest["plans"]:
        source_files.append(entry["file"])
        plan = json.loads((args.shard_bundle / entry["file"]).read_text())
        for frame, rows in plan["predictions_by_frame"].items():
            merged.setdefault(frame, []).extend(rows)
    merged = {frame: rows for frame, rows in sorted(merged.items(), key=lambda item: int(item[0])) if rows}
    plan = {
        "name": args.name,
        "target": [1920, 1080],
        "predictions_by_frame": merged,
        "score_assessment": {
            "scope": "complete participant-created validation pseudo-label set",
            "frame_range_inclusive": [1, 249],
            "prediction_count": sum(len(rows) for rows in merged.values()),
            "competition_evaluation": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_file = "full-validation-plan.json"
    payload = json.dumps(plan, indent=2) + "\n"
    (args.output / plan_file).write_text(payload)
    manifest = {
        "description": "One complete validation-only prediction plan.",
        "plans": [
            {
                "shard": "full",
                "file": plan_file,
                "name": args.name,
                "prediction_count": plan["score_assessment"]["prediction_count"],
                "frames_with_predictions": len(merged),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        ],
        "source_shard_bundle": str(args.shard_bundle),
        "source_files": source_files,
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

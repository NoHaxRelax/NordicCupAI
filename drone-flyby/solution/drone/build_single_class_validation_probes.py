#!/usr/bin/env python3
"""Split an existing validation plan into additive single-class probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    args = parser.parse_args()

    source = json.loads(args.plan.read_text())
    present = {
        row["object_id"]
        for rows in source["predictions_by_frame"].values()
        for row in rows
    }
    unknown = set(args.classes) - present
    if unknown:
        raise ValueError(f"Requested classes absent from source plan: {sorted(unknown)}")

    args.output.mkdir(parents=True, exist_ok=True)
    plans = []
    for label in args.classes:
        predictions = {
            frame: [row for row in rows if row["object_id"] == label]
            for frame, rows in source["predictions_by_frame"].items()
        }
        predictions = {frame: rows for frame, rows in predictions.items() if rows}
        name = f"{args.name_prefix}-{label}"
        filename = f"class-{label}.json"
        payload = {
            "name": name,
            "target": source["target"],
            "predictions_by_frame": predictions,
            "score_assessment": {
                "class": label,
                "prediction_count": sum(map(len, predictions.values())),
                "maximum_score_if_class_is_present": 1 / 16,
                "source_plan": str(args.plan),
                "validation_only": True,
            },
        }
        path = args.output / filename
        path.write_text(json.dumps(payload, indent=2) + "\n")
        plans.append({
            "shard": label,
            "file": filename,
            "name": name,
            "classes": [label],
            "prediction_count": payload["score_assessment"]["prediction_count"],
            "maximum_score_if_all_16_classes_are_present": 1 / 16,
            "sha256": digest(path),
        })

    manifest = {
        "description": "Additive single-class validation-only diagnostic probes.",
        "plans": plans,
        "source_plan": str(args.plan),
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"plans": plans}, indent=2))


if __name__ == "__main__":
    main()

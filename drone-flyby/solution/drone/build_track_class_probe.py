#!/usr/bin/env python3
"""Build one validation-only class hypothesis for an existing participant track."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


WIDTH, HEIGHT = 3840.0, 2160.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", type=Path)
    parser.add_argument("class_name")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.track.read_text())
    predictions = {}
    for row in payload["annotations"]:
        x1, y1, x2, y2 = map(float, row["bbox_source_xyxy"])
        predictions[str(int(row["frame"]))] = [{
            "object_id": args.class_name,
            "bbox": [x1 / WIDTH, y1 / HEIGHT, x2 / WIDTH, y2 / HEIGHT],
            "confidence": 1.0,
        }]
    plan = {
        "name": args.name,
        "target": [1920, 1080],
        "predictions_by_frame": predictions,
        "validation_only_hypothesis": {
            "source_track": str(args.track),
            "source_class": payload["annotations"][0]["class"],
            "hypothesis_class": args.class_name,
            "frames": sorted(map(int, predictions)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "class-shard-a.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    manifest = {
        "description": "Single-track, single-class validation-only hypothesis.",
        "plans": [{
            "shard": "a",
            "file": plan_path.name,
            "name": args.name,
            "prediction_count": len(predictions),
            "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        }],
        "execution_gate": "Validation only. No evaluation command is implemented.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"name": args.name, "frames": len(predictions), "class": args.class_name}))


if __name__ == "__main__":
    main()

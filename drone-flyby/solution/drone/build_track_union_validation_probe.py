#!/usr/bin/env python3
"""Build a validation-only probe that unions component boxes per frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=Path, action="append", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_frame: dict[int, list[list[float]]] = defaultdict(list)
    for track_path in args.track:
        track = json.loads(track_path.read_text())
        for annotation in track["annotations"]:
            by_frame[int(annotation["frame"])].append(annotation["bbox_source_xyxy"])

    predictions_by_frame = {}
    union_rows = []
    for frame, boxes in sorted(by_frame.items()):
        union = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        normalized = [union[0] / 3840, union[1] / 2160, union[2] / 3840, union[3] / 2160]
        predictions_by_frame[str(frame)] = [
            {"object_id": args.class_name, "bbox": normalized, "confidence": 0.9}
        ]
        union_rows.append({"frame": frame, "bbox_source_xyxy": union, "component_count": len(boxes)})

    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "name": args.name,
        "target": [1920, 1080],
        "predictions_by_frame": predictions_by_frame,
    }
    plan_payload = json.dumps(plan, indent=2) + "\n"
    plan_file = "track-union.json"
    (args.output / plan_file).write_text(plan_payload)
    (args.output / "union-boxes.json").write_text(json.dumps(union_rows, indent=2) + "\n")
    manifest = {
        "description": "Validation-only probe for one object represented by the per-frame union of component tracks.",
        "plans": [
            {
                "shard": "union",
                "file": plan_file,
                "name": args.name,
                "class": args.class_name,
                "prediction_count": len(predictions_by_frame),
                "sha256": hashlib.sha256(plan_payload.encode()).hexdigest(),
            }
        ],
        "source_tracks": [str(path) for path in args.track],
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

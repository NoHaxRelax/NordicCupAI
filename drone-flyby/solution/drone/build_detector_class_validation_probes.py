#!/usr/bin/env python3
"""Build class-isolated validation probes from detector candidate reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--class-name", action="append", required=True)
    parser.add_argument("--maximum-candidates", type=int, default=100)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_plans = []
    for class_name in args.class_name:
        candidates = [row for row in report["predictions"] if row["class"] == class_name]
        candidates.sort(key=lambda row: -float(row["confidence"]))
        candidates = candidates[: args.maximum_candidates]
        by_frame: dict[str, list[dict]] = defaultdict(list)
        for row in candidates:
            x1, y1, x2, y2 = row["bbox_source_xyxy"]
            by_frame[str(row["frame"])].append(
                {
                    "object_id": class_name,
                    "bbox": [x1 / 3840, y1 / 2160, x2 / 3840, y2 / 2160],
                    "confidence": float(row["confidence"]),
                }
            )
        plan = {
            "name": f"{args.name_prefix}-{class_name}"[:80],
            "target": [1920, 1080],
            "predictions_by_frame": dict(sorted(by_frame.items(), key=lambda item: int(item[0]))),
        }
        filename = f"class-{class_name}.json"
        payload = json.dumps(plan, indent=2) + "\n"
        (args.output / filename).write_text(payload)
        manifest_plans.append(
            {
                "shard": class_name,
                "file": filename,
                "name": plan["name"],
                "class": class_name,
                "prediction_count": len(candidates),
                "frames_with_predictions": len(by_frame),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    manifest = {
        "description": "Detector-candidate class probes for validation-only missing-class diagnosis.",
        "plans": manifest_plans,
        "source_report": str(args.report),
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

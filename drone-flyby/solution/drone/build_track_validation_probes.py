#!/usr/bin/env python3
"""Build validation-only probes for individual annotation tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CONFIDENCE_BY_PROVENANCE = {
    "score_confirmed_match": 1.0,
    "score_anchored_projective": 0.98,
    "reviewed_positive": 0.90,
    "algorithmic_shared_motion": 0.80,
}


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--track", action="append", required=True)
    args = parser.parse_args()

    requested = set(args.track)
    rows_by_track: dict[str, list[dict]] = {track: [] for track in requested}
    metadata_dir = args.dataset / "finetune-metadata" / "annotations"
    for path in sorted(metadata_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        frame = int(payload["frame"])
        for row in payload["annotations"]:
            track_id = row.get("track_id")
            if track_id in rows_by_track:
                rows_by_track[track_id].append({"frame": frame, **row})

    missing = sorted(track for track, rows in rows_by_track.items() if not rows)
    if missing:
        raise ValueError(f"Tracks not found: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_plans = []
    for track_id in args.track:
        rows = rows_by_track[track_id]
        predictions = {}
        classes = set()
        for row in rows:
            label = row["object_id"]
            classes.add(label)
            provenance = row["provenance"]
            confidence = CONFIDENCE_BY_PROVENANCE[provenance]
            x1, y1, x2, y2 = row["bbox"]
            predictions[str(row["frame"])] = [
                {
                    "object_id": label,
                    "bbox": [x1 / 3840, y1 / 2160, x2 / 3840, y2 / 2160],
                    "confidence": confidence,
                }
            ]
        if len(classes) != 1:
            raise ValueError(f"Track {track_id} has classes {sorted(classes)}")
        shard = safe_name(track_id)
        filename = f"track-{shard}.json"
        plan = {
            "name": f"{args.name_prefix}-{shard}"[:80],
            "target": [1920, 1080],
            "predictions_by_frame": predictions,
        }
        payload = json.dumps(plan, indent=2) + "\n"
        (args.output / filename).write_text(payload)
        manifest_plans.append(
            {
                "shard": shard,
                "track_id": track_id,
                "class": next(iter(classes)),
                "file": filename,
                "name": plan["name"],
                "frame_range": [min(row["frame"] for row in rows), max(row["frame"] for row in rows)],
                "prediction_count": len(rows),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )

    manifest = {
        "description": "Track-isolated validation-only diagnostic probes.",
        "plans": manifest_plans,
        "source_dataset": str(args.dataset),
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

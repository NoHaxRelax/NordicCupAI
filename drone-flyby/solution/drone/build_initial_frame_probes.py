#!/usr/bin/env python3
"""Build isolated validation-only probes for projected frame 1-4 tracks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/drone-validation-coverage/initial-frame-audit/projected-boxes.json"
OUTPUT = ROOT / "artifacts/drone-api-tests/initial-frame-track-probes-20260918"
WIDTH, HEIGHT = 3840.0, 2160.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = json.loads(INPUT.read_text())
    by_class: dict[str, dict[str, list[dict]]] = {}
    for frame_row in rows:
        for annotation in frame_row["annotations"]:
            label = annotation["class"]
            x1, y1, x2, y2 = map(float, annotation["bbox_source_xyxy"])
            by_class.setdefault(label, {})[str(frame_row["frame"])] = [
                {
                    "object_id": label,
                    "bbox": [x1 / WIDTH, y1 / HEIGHT, x2 / WIDTH, y2 / HEIGHT],
                    "confidence": 1.0,
                }
            ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plans = []
    for index, label in enumerate(sorted(by_class), start=1):
        name = f"initial-frames-1-4-{label}-20260918"
        plan = {
            "name": name,
            "target": [1920, 1080],
            "predictions_by_frame": by_class[label],
            "validation_only_hypothesis": {
                "class": label,
                "frames": [1, 2, 3, 4],
                "source": str(INPUT),
                "geometry": "frame-5 reviewed box projected backward with calibrated shared camera motion",
                "interpretation": "A positive score proves at least one submitted box matches at IoU >= 0.50.",
            },
        }
        filename = f"class-shard-{index}.json"
        path = OUTPUT / filename
        path.write_text(json.dumps(plan, indent=2) + "\n")
        plans.append(
            {
                "shard": str(index),
                "file": filename,
                "name": name,
                "class": label,
                "prediction_count": 4,
                "sha256": digest(path),
            }
        )
    manifest = {
        "description": "Two isolated frame 1-4 class-track hypotheses; validation only.",
        "plans": plans,
        "execution_gate": "No evaluation route is implemented. Each plan is an isolated validation attempt.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

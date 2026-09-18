#!/usr/bin/env python3
"""Prepare native-pixel images for the remote missing-class detector scan."""
from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/drone/reconstructed-validation"
OUTPUT = ROOT / "artifacts/drone-validation-coverage/yolo-missing-scan-inputs"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    rows = []
    frames = list(range(5, 13)) + list(range(13, 250, 4))
    for frame in frames:
        source_path = SOURCE / f"frame_{frame:06d}.png"
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2160, 3840):
            raise ValueError(f"Unexpected image: {source_path}")
        height = 2160 if frame <= 12 else 700
        target = OUTPUT / f"frame_{frame:06d}.png"
        if not cv2.imwrite(str(target), image[:height], [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"Failed to write {target}")
        rows.append(
            {
                "frame": frame,
                "file": target.name,
                "source_region_xyxy": [0, 0, 3840, height],
            }
        )
    manifest = {
        "description": "Native-pixel validation inputs for missing-class proposal inference.",
        "frames": rows,
        "classes_of_interest": ["small_plane", "ta-ta", "jammer", "spacecraft"],
        "live_queries": 0,
        "training": False,
        "competition_evaluation": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"frames": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()

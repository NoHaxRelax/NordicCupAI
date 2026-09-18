#!/usr/bin/env python3
"""Render frames 1-4 with camera-motion back-projections from frame 5."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drone"))

from perspective_tracking.motion import MotionModel  # noqa: E402


ANCHORS = {
    "large_launcher": [1936.25, 1208.16, 2018.75, 1267.84],
    "mine_roller": [1647.0, 810.0, 1710.0, 855.0],
}
COLORS = {"large_launcher": (0, 220, 255), "mine_roller": (255, 220, 0)}


def capture_row(frame: int) -> tuple[Path, dict]:
    matches = []
    capture_root = (
        ROOT
        / "data/drone/capture/score-anchored-v3-c-20260918"
        / "score-anchored-v3-class-c-20260918"
    )
    for path in sorted(capture_root.glob("*/*.json")):
        row = json.loads(path.read_text())
        if int(row["frame"]) == frame:
            matches.append((path.with_name(row["image_file"]), row))
    if len(matches) != 1:
        raise ValueError(f"Expected one captured input for frame {frame}, got {matches}")
    return matches[0]


def main() -> None:
    model = MotionModel.from_dict(
        json.loads(
            (ROOT / "artifacts/drone-conditioned-shape/validation-calibration.json").read_text()
        )
    )
    output = ROOT / "artifacts/drone-validation-coverage/initial-frame-audit"
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    rows = []
    for frame in range(1, 5):
        image_path, capture = capture_row(frame)
        image = cv2.imread(str(image_path))
        sx1, sy1, sx2, sy2 = map(float, capture["view"]["source_region_xyxy"])
        scale = np.asarray(
            [image.shape[1] / (sx2 - sx1), image.shape[0] / (sy2 - sy1)] * 2
        )
        offset = np.asarray([sx1, sy1, sx1, sy1])
        frame_rows = []
        for label, anchor in ANCHORS.items():
            projected = model.box(np.asarray(anchor, dtype=float), 4.0, float(frame - 1))
            x1, y1, x2, y2 = np.round((projected - offset) * scale).astype(int)
            color = COLORS[label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            frame_rows.append(
                {"class": label, "bbox_source_xyxy": np.round(projected, 2).tolist()}
            )
        cv2.putText(image, f"frame {frame}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(output / f"frame_{frame:06d}.png"), image)
        panels.append(image)
        rows.append(
            {
                "frame": frame,
                "source_region_xyxy": capture["view"]["source_region_xyxy"],
                "annotations": frame_rows,
            }
        )
    contact = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(str(output / "contact.png"), contact)
    (output / "projected-boxes.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": len(rows), "boxes": 8}, indent=2))


if __name__ == "__main__":
    main()

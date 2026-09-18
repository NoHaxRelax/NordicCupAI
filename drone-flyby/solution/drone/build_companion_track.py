#!/usr/bin/env python3
"""Build a proposal for an object with a fixed offset from a reviewed track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--offset", nargs=4, type=float, metavar=("DX1", "DY1", "DX2", "DY2"), required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    dx1, dy1, dx2, dy2 = args.offset
    boxes = []
    for row in reference["boxes"]:
        x1, y1, _, _ = map(float, row["bbox_source_xyxy"])
        boxes.append(
            {
                "frame": int(row["frame"]),
                "bbox_source_xyxy": [
                    round(x1 + dx1, 3),
                    round(y1 + dy1, 3),
                    round(x1 + dx2, 3),
                    round(y1 + dy2, 3),
                ],
            }
        )
    payload = {
        "track_id": args.track_id,
        "class_hypothesis": args.label,
        "proposal_method": "fixed companion offset from a visually reviewed reference track; requires its own visual review",
        "boxes": boxes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frames": len(boxes), "first": boxes[0]["frame"], "last": boxes[-1]["frame"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a reviewable CSRT tracking proposal from a manually selected box."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def tracker() -> cv2.Tracker:
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


def image_path(images: Path, frame: int) -> Path:
    return images / f"frame_{frame:06d}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.start <= args.seed_frame <= args.end:
        raise SystemExit("Seed frame must fall inside --start/--end")
    x1, y1, x2, y2 = args.bbox
    initial = tuple(map(int, (x1, y1, x2 - x1, y2 - y1)))

    def run(frames: range) -> dict[int, list[float]]:
        first = cv2.imread(str(image_path(args.images, args.seed_frame)))
        active = tracker()
        active.init(first, initial)
        boxes: dict[int, list[float]] = {args.seed_frame: [x1, y1, x2, y2]}
        for frame in frames:
            image = cv2.imread(str(image_path(args.images, frame)))
            ok, (x, y, w, h) = active.update(image)
            if not ok:
                break
            boxes[frame] = [x, y, x + w, y + h]
        return boxes

    boxes = run(range(args.seed_frame - 1, args.start - 1, -1))
    boxes.update(run(range(args.seed_frame + 1, args.end + 1)))
    payload = {
        "track_id": args.track_id,
        "class_hypothesis": args.label,
        "proposal_method": "manual seed plus CSRT; requires visual review before materialization",
        "boxes": [
            {"frame": frame, "bbox_source_xyxy": [round(v, 2) for v in boxes[frame]]}
            for frame in sorted(boxes)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frames": len(boxes), "first": min(boxes), "last": max(boxes)}))


if __name__ == "__main__":
    main()

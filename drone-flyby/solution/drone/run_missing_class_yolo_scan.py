#!/usr/bin/env python3
"""Run tiled YOLO proposal inference; no training or scoring."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_INTEREST = {"small_plane", "ta-ta", "jammer", "spacecraft"}


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def starts(length: int, window: int, overlap: float) -> list[int]:
    if length <= window:
        return [0]
    step = max(1, round(window * (1 - overlap)))
    values = list(range(0, length - window + 1, step))
    if values[-1] != length - window:
        values.append(length - window)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.005)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--top-height",
        type=int,
        help="Optionally scan only this many source pixels from the top of each input image.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Class names to scan, or the single value 'all'. Defaults to the four classes absent from v4.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.inputs / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if isinstance(manifest, dict) and "frames" in manifest:
            items = manifest["frames"]
        elif args.frame_start is not None and args.frame_end is not None:
            items = [
                {"frame": frame, "file": f"frame_{frame:06d}.png"}
                for frame in range(args.frame_start, args.frame_end + 1, args.frame_step)
            ]
        else:
            raise ValueError("Unsupported manifest format; pass --frame-start and --frame-end")
    else:
        if args.frame_start is None or args.frame_end is None:
            raise ValueError("An input directory without manifest.json requires --frame-start and --frame-end")
        if args.frame_step <= 0:
            raise ValueError("--frame-step must be positive")
        items = [
            {"frame": frame, "file": f"frame_{frame:06d}.png"}
            for frame in range(args.frame_start, args.frame_end + 1, args.frame_step)
        ]
    model = YOLO(str(args.weights))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    names = {int(key): value for key, value in model.names.items()}
    interest = set(names.values()) if args.classes == ["all"] else set(args.classes or DEFAULT_INTEREST)
    class_ids = [key for key, value in names.items() if value in interest]
    if {names[key] for key in class_ids} != interest:
        missing = sorted(interest - set(names.values()))
        raise ValueError(f"Checkpoint does not cover requested classes {missing}: {names}")
    all_rows = []
    timing = []
    for item in items:
        started = time.monotonic()
        image = cv2.imread(str(args.inputs / item["file"]))
        if image is None:
            raise FileNotFoundError(args.inputs / item["file"])
        if args.top_height is not None:
            image = image[: args.top_height]
        height, width = image.shape[:2]
        tiles = []
        origins = []
        for top in starts(height, 540, 0.25):
            for left in starts(width, 960, 0.25):
                tiles.append(image[top : top + 540, left : left + 960])
                origins.append((left, top))
        frame_rows = []
        results = model.predict(
            tiles,
            imgsz=960,
            conf=args.confidence,
            iou=0.5,
            max_det=300,
            classes=class_ids,
            device=device,
            batch=8 if device == "mps" else 4,
            verbose=False,
        )
        for result, (left, top) in zip(results, origins):
            if result.boxes is None:
                continue
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(float, box)
                frame_rows.append(
                    {
                        "frame": int(item["frame"]),
                        "class": names[int(class_id)],
                        "confidence": float(confidence),
                        "bbox_source_xyxy": [x1 + left, y1 + top, x2 + left, y2 + top],
                        "tile_origin": [left, top],
                    }
                )
        frame_rows.sort(key=lambda row: row["confidence"], reverse=True)
        unique = []
        for row in frame_rows:
            duplicate = next(
                (
                    other
                    for other in unique
                    if other["class"] == row["class"]
                    and iou(other["bbox_source_xyxy"], row["bbox_source_xyxy"]) > 0.35
                ),
                None,
            )
            if duplicate is None:
                unique.append(row)
            else:
                duplicate.setdefault("duplicate_tile_votes", 1)
                duplicate["duplicate_tile_votes"] += 1
        all_rows.extend(unique)
        timing.append(
            {
                "frame": int(item["frame"]),
                "tiles": len(tiles),
                "raw_predictions": len(frame_rows),
                "fused_predictions": len(unique),
                "seconds": time.monotonic() - started,
            }
        )
        print(json.dumps(timing[-1]), flush=True)
    all_rows.sort(key=lambda row: row["confidence"], reverse=True)
    report = {
        "description": "Tiled YOLO validation proposals; candidates only.",
        "weights": str(args.weights),
        "settings": {
            "confidence": args.confidence,
            "imgsz": 960,
            "tile": [960, 540],
            "overlap": 0.25,
            "classes": sorted(interest),
            "device": device,
            "frame_range": [items[0]["frame"], items[-1]["frame"]],
            "frame_step": args.frame_step,
            "top_height": args.top_height,
        },
        "predictions": all_rows,
        "class_counts": {
            label: sum(row["class"] == label for row in all_rows) for label in sorted(interest)
        },
        "timing": timing,
        "training": False,
        "live_queries": 0,
        "competition_evaluation": False,
        "limitations": [
            "The checkpoint was trained on public reference objects and proposals require visual or validation-score confirmation.",
            (
                f"Frames {items[0]['frame']}-{items[-1]['frame']} use step {args.frame_step}; "
                + (f"only the top {args.top_height} source pixels are scanned." if args.top_height else "full inputs are scanned.")
            ),
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"predictions": len(all_rows), "class_counts": report["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()

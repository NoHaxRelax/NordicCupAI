#!/usr/bin/env python3
"""Render native-resolution contact sheets for missing-class YOLO proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PANEL_WIDTH = 480
PANEL_HEIGHT = 340
GRID_COLUMNS = 3
GRID_ROWS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument("--class-name", action="append")
    parser.add_argument("--maximum-candidates", type=int)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    manifest_path = args.inputs / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if isinstance(input_manifest, dict) and "frames" in input_manifest:
        files_by_frame = {
            int(item["frame"]): args.inputs / item["file"]
            for item in input_manifest["frames"]
        }
    else:
        files_by_frame = {
            int(path.stem.rsplit("_", 1)[-1]): path
            for path in args.inputs.glob("frame_*.png")
        }
    rows = [
        row for row in report["predictions"]
        if float(row["confidence"]) >= args.minimum_confidence
        and (not args.class_name or row["class"] in set(args.class_name))
    ]
    if args.maximum_candidates is not None:
        rows.sort(key=lambda row: -float(row["confidence"]))
        rows = rows[: args.maximum_candidates]
    rows.sort(key=lambda row: (int(row["frame"]), -float(row["confidence"])))

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("candidate-audit-*.png"):
        stale.unlink()

    capacity = GRID_COLUMNS * GRID_ROWS
    page_files: list[str] = []
    rendered: list[dict] = []
    for page_index, start in enumerate(range(0, len(rows), capacity), start=1):
        page = np.full(
            (GRID_ROWS * PANEL_HEIGHT, GRID_COLUMNS * PANEL_WIDTH, 3),
            24,
            dtype=np.uint8,
        )
        for slot, row in enumerate(rows[start : start + capacity]):
            frame = int(row["frame"])
            image = cv2.imread(str(files_by_frame[frame]), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Unable to read input for frame {frame}")
            box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            centre = (box[:2] + box[2:]) / 2
            extent = np.maximum(box[2:] - box[:2], 1)
            crop_width = max(360.0, float(extent[0]) * 7)
            crop_height = max(240.0, float(extent[1]) * 6)
            x1 = max(0, int(round(centre[0] - crop_width / 2)))
            y1 = max(0, int(round(centre[1] - crop_height / 2)))
            x2 = min(image.shape[1], int(round(centre[0] + crop_width / 2)))
            y2 = min(image.shape[0], int(round(centre[1] + crop_height / 2)))
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            scale = min(PANEL_WIDTH / crop.shape[1], (PANEL_HEIGHT - 42) / crop.shape[0])
            resized = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), np.uint8)
            offset_x = (PANEL_WIDTH - resized.shape[1]) // 2
            offset_y = 38 + (PANEL_HEIGHT - 38 - resized.shape[0]) // 2
            panel[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized

            drawn = np.r_[box[:2] - [x1, y1], box[2:] - [x1, y1]] * scale
            drawn[[0, 2]] += offset_x
            drawn[[1, 3]] += offset_y
            cv2.rectangle(
                panel,
                tuple(np.round(drawn[:2]).astype(int)),
                tuple(np.round(drawn[2:]).astype(int)),
                (0, 255, 255),
                2,
            )
            title = f"f{frame:03d} {row['class']} {float(row['confidence']):.4f}"
            cv2.putText(
                panel,
                title,
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            grid_x = (slot % GRID_COLUMNS) * PANEL_WIDTH
            grid_y = (slot // GRID_COLUMNS) * PANEL_HEIGHT
            page[grid_y : grid_y + PANEL_HEIGHT, grid_x : grid_x + PANEL_WIDTH] = panel
            rendered.append({**row, "page": page_index, "slot": slot})

        filename = f"candidate-audit-{page_index:02d}.png"
        if not cv2.imwrite(str(args.output / filename), page):
            raise SystemExit(f"Unable to write {filename}")
        page_files.append(filename)

    manifest = {
        "description": "Native-resolution visual audit of missing-class detector proposals; no proposal is accepted by this rendering step.",
        "source_report": str(args.report),
        "minimum_confidence": args.minimum_confidence,
        "class_names": args.class_name,
        "maximum_candidates": args.maximum_candidates,
        "pages": page_files,
        "candidates": rendered,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidates": len(rendered), "pages": len(page_files)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render native-resolution crops for visual auditing of algorithmic completions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


PANEL_SIZE = (420, 320)
GRID = (3, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_track: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.annotations.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            if row.get("review_status") == "algorithmic_bottom_completion":
                by_track[row.get("track_id", path.stem)].append(row)

    selections = []
    for track_id, rows in sorted(by_track.items()):
        ordered = sorted(rows, key=lambda row: int(row["frame"]))
        indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
        selections.extend((track_id, ordered[index]) for index in indices)

    args.output.mkdir(parents=True, exist_ok=True)
    for stale_page in args.output.glob("completion-audit-*.png"):
        stale_page.unlink()
    page_capacity = GRID[0] * GRID[1]
    page_files = []
    audit_rows = []
    for page_index, start in enumerate(range(0, len(selections), page_capacity), start=1):
        page = np.full((GRID[1] * PANEL_SIZE[1], GRID[0] * PANEL_SIZE[0], 3), 28, np.uint8)
        for slot, (track_id, row) in enumerate(selections[start : start + page_capacity]):
            frame = int(row["frame"])
            image_path = args.images / f"frame_{frame:06d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Unable to read {image_path}")
            box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            centre = (box[:2] + box[2:]) / 2
            extent = np.maximum(box[2:] - box[:2], 1)
            crop_width = max(320.0, extent[0] * 5)
            crop_height = max(220.0, extent[1] * 4)
            x1 = max(0, int(round(centre[0] - crop_width / 2)))
            y1 = max(0, int(round(centre[1] - crop_height / 2)))
            x2 = min(image.shape[1], int(round(centre[0] + crop_width / 2)))
            y2 = min(image.shape[0], int(round(centre[1] + crop_height / 2)))
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                raise ValueError(f"Empty crop for {track_id} frame {frame}")
            scale = min(PANEL_SIZE[0] / crop.shape[1], (PANEL_SIZE[1] - 40) / crop.shape[0])
            resized = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0], 3), np.uint8)
            offset_x = (PANEL_SIZE[0] - resized.shape[1]) // 2
            offset_y = 32 + (PANEL_SIZE[1] - 32 - resized.shape[0]) // 2
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
            cv2.putText(
                panel,
                f"{track_id}  f{frame}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            grid_x = (slot % GRID[0]) * PANEL_SIZE[0]
            grid_y = (slot // GRID[0]) * PANEL_SIZE[1]
            page[grid_y : grid_y + PANEL_SIZE[1], grid_x : grid_x + PANEL_SIZE[0]] = panel
            audit_rows.append(
                {
                    "track_id": track_id,
                    "frame": frame,
                    "bbox_source_xyxy": row["bbox_source_xyxy"],
                    "page": page_index,
                    "slot": slot,
                }
            )
        filename = f"completion-audit-{page_index:02d}.png"
        if not cv2.imwrite(str(args.output / filename), page):
            raise SystemExit(f"Unable to write {args.output / filename}")
        page_files.append(filename)

    manifest = {
        "description": "First, middle and last native-resolution crop checks for every algorithmically extended track.",
        "pages": page_files,
        "samples": audit_rows,
        "box_colour_bgr": [0, 255, 255],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tracks": len(by_track), "samples": len(audit_rows), "pages": len(page_files)}))


if __name__ == "__main__":
    main()

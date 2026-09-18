#!/usr/bin/env python3
"""Render evenly spaced native-resolution samples for every track of one class."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PANEL_W, PANEL_H = 520, 360
COLS, ROWS = 3, 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-track", type=int, default=5)
    args = parser.parse_args()

    selections = []
    for path in sorted(args.annotations.glob("*.json")):
        if path.name in {"completion-report.json", "score-anchor-repair-report.json"}:
            continue
        payload = json.loads(path.read_text())
        rows = sorted(
            [row for row in payload.get("annotations", []) if row["class"] == args.class_name],
            key=lambda row: int(row["frame"]),
        )
        if not rows:
            continue
        indices = sorted(set(np.linspace(0, len(rows) - 1, min(args.samples_per_track, len(rows))).round().astype(int)))
        selections.extend((path.stem, rows[index]) for index in indices)

    args.output.mkdir(parents=True, exist_ok=True)
    pages, records = [], []
    capacity = COLS * ROWS
    for page_index, start in enumerate(range(0, len(selections), capacity), start=1):
        page = np.full((ROWS * PANEL_H, COLS * PANEL_W, 3), 25, np.uint8)
        for slot, (track_id, row) in enumerate(selections[start : start + capacity]):
            frame = int(row["frame"])
            image = cv2.imread(str(args.images / f"frame_{frame:06d}.png"), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Missing frame {frame}")
            box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            centre = (box[:2] + box[2:]) / 2
            extent = np.maximum(box[2:] - box[:2], 1)
            crop_w, crop_h = max(360.0, extent[0] * 7), max(260.0, extent[1] * 6)
            x1 = max(0, int(round(centre[0] - crop_w / 2)))
            y1 = max(0, int(round(centre[1] - crop_h / 2)))
            x2 = min(image.shape[1], int(round(centre[0] + crop_w / 2)))
            y2 = min(image.shape[0], int(round(centre[1] + crop_h / 2)))
            crop = image[y1:y2, x1:x2]
            scale = min(PANEL_W / crop.shape[1], (PANEL_H - 42) / crop.shape[0])
            resized = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            ox, oy = (PANEL_W - resized.shape[1]) // 2, 42 + (PANEL_H - 42 - resized.shape[0]) // 2
            panel[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
            drawn = np.r_[box[:2] - [x1, y1], box[2:] - [x1, y1]] * scale
            drawn[[0, 2]] += ox
            drawn[[1, 3]] += oy
            cv2.rectangle(panel, tuple(np.round(drawn[:2]).astype(int)), tuple(np.round(drawn[2:]).astype(int)), (0,255,255), 2)
            cv2.putText(panel, f"{track_id}  f{frame}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
            gx, gy = (slot % COLS) * PANEL_W, (slot // COLS) * PANEL_H
            page[gy : gy + PANEL_H, gx : gx + PANEL_W] = panel
            records.append({"track_id": track_id, "frame": frame, "page": page_index, "slot": slot})
        filename = f"{args.class_name}-track-audit-{page_index:02d}.png"
        cv2.imwrite(str(args.output / filename), page)
        pages.append(filename)
    (args.output / "manifest.json").write_text(json.dumps({"class": args.class_name, "pages": pages, "samples": records}, indent=2) + "\n")
    print(json.dumps({"class": args.class_name, "tracks": len({row[0] for row in selections}), "samples": len(selections), "pages": len(pages)}))


if __name__ == "__main__":
    main()

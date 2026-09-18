#!/usr/bin/env python3
"""Render old/new native-resolution crops for every score-anchor-repaired track."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", help="Limit rendering to one track filename stem.")
    parser.add_argument("--all-rows", action="store_true", help="Render every row instead of first/middle/last.")
    args = parser.parse_args()

    by_track: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.annotations.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("annotations", [])
        if rows and "pre_score_anchor_repair_bbox_source_xyxy" in rows[0]:
            by_track[path.stem].extend(rows)

    samples: list[tuple[str, dict]] = []
    for track_id, rows in sorted(by_track.items()):
        if args.track and track_id != args.track:
            continue
        ordered = sorted(rows, key=lambda row: int(row["frame"]))
        indices = range(len(ordered)) if args.all_rows else sorted({0, len(ordered) // 2, len(ordered) - 1})
        samples.extend((track_id, ordered[index]) for index in indices)

    args.output.mkdir(parents=True, exist_ok=True)
    page_files = []
    manifest_rows = []
    capacity = COLS * ROWS
    for page_index, start in enumerate(range(0, len(samples), capacity), start=1):
        page = np.full((ROWS * PANEL_H, COLS * PANEL_W, 3), 25, np.uint8)
        for slot, (track_id, row) in enumerate(samples[start : start + capacity]):
            frame = int(row["frame"])
            image_path = args.images / f"frame_{frame:06d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Unable to read {image_path}")
            new_box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            old_box = np.asarray(row["pre_score_anchor_repair_bbox_source_xyxy"], dtype=float)
            union = np.r_[np.minimum(new_box[:2], old_box[:2]), np.maximum(new_box[2:], old_box[2:])]
            centre = (union[:2] + union[2:]) / 2
            extent = np.maximum(union[2:] - union[:2], 1)
            crop_w = max(420.0, extent[0] * 5)
            crop_h = max(280.0, extent[1] * 4)
            x1 = max(0, int(round(centre[0] - crop_w / 2)))
            y1 = max(0, int(round(centre[1] - crop_h / 2)))
            x2 = min(image.shape[1], int(round(centre[0] + crop_w / 2)))
            y2 = min(image.shape[0], int(round(centre[1] + crop_h / 2)))
            crop = image[y1:y2, x1:x2].copy()
            scale = min(PANEL_W / crop.shape[1], (PANEL_H - 52) / crop.shape[0])
            resized = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            ox = (PANEL_W - resized.shape[1]) // 2
            oy = 46 + (PANEL_H - 46 - resized.shape[0]) // 2
            panel[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
            for box, colour in ((old_box, (255, 0, 255)), (new_box, (0, 255, 255))):
                drawn = np.r_[box[:2] - [x1, y1], box[2:] - [x1, y1]] * scale
                drawn[[0, 2]] += ox
                drawn[[1, 3]] += oy
                cv2.rectangle(
                    panel,
                    tuple(np.round(drawn[:2]).astype(int)),
                    tuple(np.round(drawn[2:]).astype(int)),
                    colour,
                    2,
                )
            cv2.putText(panel, f"{track_id}  f{frame}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(panel, "old=magenta  repaired=yellow", (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210,210,210), 1, cv2.LINE_AA)
            gx = (slot % COLS) * PANEL_W
            gy = (slot // COLS) * PANEL_H
            page[gy : gy + PANEL_H, gx : gx + PANEL_W] = panel
            manifest_rows.append({"track_id": track_id, "frame": frame, "page": page_index, "slot": slot})
        filename = f"score-anchor-repair-audit-{page_index:02d}.png"
        if not cv2.imwrite(str(args.output / filename), page):
            raise SystemExit(f"Unable to write {args.output / filename}")
        page_files.append(filename)

    manifest = {
        "description": "First, middle and last native-resolution samples for score-anchor-repaired tracks.",
        "legend": {"old": "magenta", "repaired": "yellow"},
        "pages": page_files,
        "samples": manifest_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tracks": len(by_track), "samples": len(samples), "pages": len(page_files)}))


if __name__ == "__main__":
    main()

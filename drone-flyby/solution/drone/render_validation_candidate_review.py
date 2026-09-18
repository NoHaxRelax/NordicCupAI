#!/usr/bin/env python3
"""Render score-confirmed validation candidates with native contextual detail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 3840, 2160


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", default="requires_manual_track")
    parser.add_argument("--context", type=int, default=150)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text())
    rows = [
        row for row in payload["candidates"] if row.get("resolution_status") == args.status
    ]
    cells = []
    for row in rows:
        image = cv2.imread(
            str(args.images / f'frame_{int(row["frame"]):06d}.png'), cv2.IMREAD_COLOR
        )
        x1, y1, x2, y2 = map(float, row["bbox_source_xyxy"])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = args.context
        sx, sy = max(0, round(cx - half)), max(0, round(cy - half))
        ex, ey = min(WIDTH, round(cx + half)), min(HEIGHT, round(cy + half))
        crop = image[sy:ey, sx:ex].copy()
        cv2.rectangle(
            crop,
            (round(x1 - sx), round(y1 - sy)),
            (round(x2 - sx), round(y2 - sy)),
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        target = np.zeros((360, 360, 3), dtype=np.uint8)
        scale = min(360 / crop.shape[1], 330 / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA,
        )
        ox = (360 - resized.shape[1]) // 2
        oy = 30 + (330 - resized.shape[0]) // 2
        target[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        title = f'{row["candidate_id"]} | f{row["frame"]} | {row["class"]}'
        cv2.putText(
            target,
            title[:52],
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(target)
    if not cells:
        raise SystemExit(f"No candidates with status {args.status}")
    columns = 3
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    sheet = cv2.vconcat(
        [cv2.hconcat(cells[index : index + columns]) for index in range(0, len(cells), columns)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Could not write {args.output}")
    print(json.dumps({"rendered": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render every frame of a tracking proposal as contextual review pages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 3840, 2160


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--cell", type=int, default=360)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    proposal = json.loads(args.proposal.read_text())
    if args.track_id:
        matches = [row for row in proposal.get("tracks", []) if row.get("track_id") == args.track_id]
        if len(matches) != 1:
            raise SystemExit("--track-id must select exactly one track")
        proposal = matches[0]
    boxes = proposal["boxes"]
    cells = []
    for row in boxes:
        frame = int(row["frame"])
        image = cv2.imread(str(args.images / f"frame_{frame:06d}.png"), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = map(float, row["bbox_source_xyxy"])
        sx = max(0, round(x1 - args.context))
        sy = max(0, round(y1 - args.context))
        ex = min(WIDTH, round(x2 + args.context))
        ey = min(HEIGHT, round(y2 + args.context))
        crop = image[sy:ey, sx:ex].copy()
        cv2.rectangle(
            crop,
            (round(x1 - sx), round(y1 - sy)),
            (round(x2 - sx), round(y2 - sy)),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        body_height = args.cell - 28
        scale = min(args.cell / crop.shape[1], body_height / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA,
        )
        cell = np.zeros((args.cell, args.cell, 3), dtype=np.uint8)
        ox = (args.cell - resized.shape[1]) // 2
        oy = 28 + (body_height - resized.shape[0]) // 2
        cell[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        cv2.putText(
            cell,
            f"frame {frame:03d}",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(cell)

    page_size = args.columns * args.rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_number, start in enumerate(range(0, len(cells), page_size), 1):
        page_cells = cells[start : start + page_size]
        while len(page_cells) < page_size:
            page_cells.append(np.zeros_like(cells[0]))
        page = cv2.vconcat(
            [
                cv2.hconcat(page_cells[index : index + args.columns])
                for index in range(0, page_size, args.columns)
            ]
        )
        path = args.output_dir / f"page-{page_number:02d}.png"
        if not cv2.imwrite(str(path), page):
            raise RuntimeError(f"Could not write {path}")
        pages.append(str(path))
    manifest = {
        "proposal": str(args.proposal),
        "track_id": proposal.get("track_id"),
        "class_hypothesis": proposal.get("class_hypothesis"),
        "box_count": len(boxes),
        "pages": pages,
        "review_status": "not_reviewed",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"boxes": len(boxes), "pages": len(pages)}))


if __name__ == "__main__":
    main()

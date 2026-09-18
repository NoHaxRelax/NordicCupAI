#!/usr/bin/env python3
"""Render reviewed manual validation annotations in the training-video style."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2

from compile_videos import OUTPUT_SIZE, draw_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(args.annotations.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            by_frame[int(row["frame"])].append({
                "object_id": row["class"],
                "bbox": row["bbox_source_xyxy"],
            })

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for frame in sorted(by_frame):
        if args.start is not None and frame < args.start:
            continue
        if args.end is not None and frame > args.end:
            continue
        image_path = args.images / f"frame_{frame:06d}.png"
        output_path = args.output / f"frame_{frame:06d}_annotated.png"
        if args.overwrite or not output_path.exists():
            image = cv2.imread(str(image_path))
            if image is None:
                raise SystemExit(f"Unable to read {image_path}")
            image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            draw_annotations(image, by_frame[frame])
            if not cv2.imwrite(str(output_path), image):
                raise SystemExit(f"Unable to write {output_path}")
        manifest.append({
            "frame": frame,
            "image": output_path.name,
            "annotations": len(by_frame[frame]),
            "classes": sorted({row["object_id"] for row in by_frame[frame]}),
        })
    if args.start is None and args.end is None:
        (args.output / "manifest.json").write_text(json.dumps({
            "description": "Manual training pseudo-label overlays, not organizer annotations.",
            "render_style": "Matches drone/compile_videos.py training overlays.",
            "frames": manifest,
        }, indent=2) + "\n")
    print(json.dumps({"frames": len(manifest), "annotations": sum(x["annotations"] for x in manifest)}))


if __name__ == "__main__":
    main()

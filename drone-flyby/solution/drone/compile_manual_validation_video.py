#!/usr/bin/env python3
"""Compile validation pseudo-labels in the existing training-video style."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2

from compile_videos import OUTPUT_SIZE, add_banner, draw_annotations, open_writer


def load_annotations(root: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            by_frame[int(row["frame"])].append({
                "object_id": row["class"],
                "bbox": row["bbox_source_xyxy"],
                "review_status": row.get("review_status", "directly_reviewed"),
                "provenance": row.get("provenance", "unknown"),
            })
    return by_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=249)
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.start > args.end:
        raise SystemExit("--start must not exceed --end")

    by_frame = load_annotations(args.annotations)
    frames = list(range(args.start, args.end + 1))
    missing = [
        frame for frame in frames
        if not (args.images / f"frame_{frame:06d}.png").is_file()
    ]
    if missing:
        raise SystemExit(f"Missing source frames: {missing}")

    writer = open_writer(args.output, args.fps, OUTPUT_SIZE)
    annotation_count = 0
    annotated_frame_count = 0
    reviewed_count = 0
    algorithmic_count = 0
    score_projected_count = 0
    try:
        for position, frame in enumerate(frames, start=1):
            image_path = args.images / f"frame_{frame:06d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")
            image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            annotations = by_frame.get(frame, [])
            if annotations:
                annotated_frame_count += 1
                annotation_count += len(annotations)
                algorithmic = sum(
                    row["review_status"] == "algorithmic_bottom_completion"
                    for row in annotations
                )
                score_projected = sum(
                    row["review_status"] == "score_positive_group_projected_track"
                    for row in annotations
                )
                reviewed = len(annotations) - algorithmic - score_projected
                reviewed_count += reviewed
                algorithmic_count += algorithmic
                score_projected_count += score_projected
                draw_annotations(image, annotations)
            else:
                reviewed = algorithmic = score_projected = 0
            if frame <= 4:
                review_scope = "coarse camera input + score-supported projection"
            elif frame == 5:
                review_scope = "full-frame discovery review"
            else:
                review_scope = "top review + shared-motion continuation"
            add_banner(
                image,
                f"VALIDATION PSEUDO-LABELS  |  frame {frame:03d}/{args.end:03d}",
                f"{reviewed} reviewed + {algorithmic + score_projected} projected  |  {review_scope}",
            )
            writer.write(image)
            if position % 25 == 0 or position == len(frames):
                print(f"validation annotations: {position}/{len(frames)} frames", flush=True)
    finally:
        writer.release()

    manifest = {
        "description": "Continuous validation video with participant-created reviewed pseudo-labels.",
        "video": str(args.output),
        "source_images": str(args.images),
        "annotation_source": str(args.annotations),
        "frame_range_inclusive": [args.start, args.end],
        "frames": len(frames),
        "annotated_frames": annotated_frame_count,
        "annotations_drawn": annotation_count,
        "reviewed_annotations_drawn": reviewed_count,
        "algorithmic_annotations_drawn": algorithmic_count,
        "score_supported_projected_annotations_drawn": score_projected_count,
        "fps": args.fps,
        "duration_seconds": len(frames) / args.fps,
        "resolution": list(OUTPUT_SIZE),
        "render_style": "Matches drone/compile_videos.py class colours, labels, and banner.",
        "review_scope": {
            "frames_1_to_4": "Coarse camera inputs with a score-supported four-frame large-launcher projection.",
            "frame_5": "Complete 3840 x 2160 frame.",
            "later_frames": "Top 540 px discovery review plus shared-motion continuation of eligible reviewed tracks.",
            "negative_training_warning": "Full-frame use is conditional on the supplied top-entry invariant and complete top-band discovery; projected boxes are not direct reviews.",
        },
        "limitations": [
            "Participant-created pseudo-labels, not organizer ground truth.",
            "Frames 1-4 lack complete native imagery; exact coarse inputs and source-region metadata are bundled separately.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

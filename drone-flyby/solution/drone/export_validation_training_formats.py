#!/usr/bin/env python3
"""Export validation pseudo-labels to COCO and Ultralytics YOLO layouts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path


CLASSES = [
    "condor", "hangar", "helicopter", "jammer", "jet_plane",
    "large_launcher", "large_tower", "medium_launcher", "medium_plane",
    "mine_roller", "small_launcher", "small_plane", "small_tower",
    "spacecraft", "ta-ta", "tank",
]
WIDTH, HEIGHT = 3840, 2160


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(args.dataset.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            by_frame[int(row["frame"])].append(row)

    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    coco_dir = args.output / "coco"
    yolo_dir = args.output / "yolo"
    for directory in (images_dir, labels_dir, coco_dir, yolo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    counts = Counter()
    annotation_id = 1
    for frame in range(1, 250):
        filename = f"frame_{frame:06d}.png"
        source = (args.images / filename).resolve()
        if not source.is_file():
            raise SystemExit(f"Missing source image {source}")
        link = images_dir / filename
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(source, link.parent))
        coco_images.append({"id": frame, "file_name": filename, "width": WIDTH, "height": HEIGHT})

        yolo_rows = []
        for row in sorted(by_frame.get(frame, []), key=lambda item: (item["class"], item.get("track_id", ""))):
            class_name = row["class"]
            class_index = CLASSES.index(class_name)
            x1, y1, x2, y2 = map(float, row["bbox_source_xyxy"])
            width, height = x2 - x1, y2 - y1
            coco_annotations.append({
                "id": annotation_id,
                "image_id": frame,
                "category_id": class_index + 1,
                "bbox": [round(x1, 4), round(y1, 4), round(width, 4), round(height, 4)],
                "area": round(width * height, 4),
                "iscrowd": 0,
                "track_id": row.get("track_id"),
                "provenance": row.get("provenance", "reviewed_positive"),
                "review_status": row.get("review_status"),
            })
            annotation_id += 1
            counts[class_name] += 1
            cx, cy = (x1 + x2) / (2 * WIDTH), (y1 + y2) / (2 * HEIGHT)
            yolo_rows.append(f"{class_index} {cx:.8f} {cy:.8f} {width / WIDTH:.8f} {height / HEIGHT:.8f}")
        (labels_dir / filename.replace(".png", ".txt")).write_text("\n".join(yolo_rows) + ("\n" if yolo_rows else ""))

    coco = {
        "info": {
            "description": "Participant-created Drone Flyby validation pseudo-labels v8; not organizer ground truth.",
            "validation_status": "native_4k_reviewed_pending_validation_confirmation",
            "evaluation_calls": 0,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": index + 1, "name": name, "supercategory": "sprite"} for index, name in enumerate(CLASSES)],
    }
    (coco_dir / "instances.json").write_text(json.dumps(coco, indent=2) + "\n")
    (yolo_dir / "data.yaml").write_text(
        "path: ..\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    )
    readme = """# Training-format export

This export contains 249 image symlinks, matching YOLO label files, and a COCO
`instances.json`. It is derived from the parent v8 pseudo-label set and is not
organizer ground truth. The `val: images` entry is a loader-compatible default,
not an independent held-out set. Split by whole object track or scene before
using it for model selection; never use random adjacent frames as a holdout.

Frames 1-4 retain the parent dataset's coarse/native-image limitation. Training
on competition validation footage remains conditional on the competition rules
and explicit authorization. No final evaluation was used to create this export.
"""
    (args.output / "README.md").write_text(readme)
    manifest = {
        "schema": 1,
        "source_dataset": str(args.dataset),
        "image_source": str(args.images),
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "class_order": CLASSES,
        "class_counts": dict(sorted(counts.items())),
        "formats": {
            "coco": "coco/instances.json",
            "yolo": "yolo/data.yaml",
            "images": "images/",
            "yolo_labels": "labels/",
        },
        "image_storage": "relative_symlinks_to_native_4k_sources",
        "validation_status": "native_4k_reviewed_pending_validation_confirmation",
        "evaluation_calls": 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compile the captured validation sequence and labelled Helsinki train set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160
OUTPUT_SIZE = (1920, 1080)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4 video at {path}")
    return writer


def captured_records(roots: Iterable[Path]) -> dict[int, list[tuple[Path, dict]]]:
    by_frame: dict[int, list[tuple[Path, dict]]] = defaultdict(list)
    for root in roots:
        for metadata_path in sorted(root.rglob("*.json")):
            try:
                record = json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "view" not in record or "image_file" not in record:
                continue
            image_path = metadata_path.parent / record["image_file"]
            if image_path.is_file():
                by_frame[int(record["frame"])].append((image_path, record))
    return by_frame


def validation_frame(records: list[tuple[Path, dict]]) -> tuple[np.ndarray, float, float]:
    width, height = OUTPUT_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    quality = np.full((height, width), -1, dtype=np.int8)

    # Identical views occur in repeated validation passes. Decode only one copy.
    unique: dict[tuple[int, tuple[int, int, int, int]], tuple[Path, dict]] = {}
    for image_path, record in records:
        view = record["view"]
        key = (int(view["resolution_level"]), tuple(view["source_region_xyxy"]))
        unique.setdefault(key, (image_path, record))

    # Coarse full-frame views fill early-frame gaps. Native crops then overwrite them.
    for (level, region), (image_path, _) in sorted(unique.items()):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        x1, y1, x2, y2 = region
        ox1, oy1, ox2, oy2 = x1 // 2, y1 // 2, x2 // 2, y2 // 2
        resized = cv2.resize(image, (ox2 - ox1, oy2 - oy1), interpolation=cv2.INTER_AREA)
        canvas[oy1:oy2, ox1:ox2] = resized
        quality[oy1:oy2, ox1:ox2] = level

    native_coverage = float(np.mean(quality == 2))
    visible_coverage = float(np.mean(quality >= 0))
    return canvas, native_coverage, visible_coverage


def add_banner(image: np.ndarray, title: str, detail: str) -> None:
    cv2.rectangle(image, (24, 22), (720, 112), (16, 16, 16), -1)
    cv2.putText(image, title, (44, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, detail, (44, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (210, 210, 210), 1, cv2.LINE_AA)


def compile_validation(roots: list[Path], output: Path, fps: float) -> dict:
    by_frame = captured_records(roots)
    if not by_frame:
        raise RuntimeError("No captured validation frames were found")

    writer = open_writer(output, fps, OUTPUT_SIZE)
    coverage = []
    frame_numbers = sorted(by_frame)
    try:
        for position, frame_number in enumerate(frame_numbers, start=1):
            image, native, visible = validation_frame(by_frame[frame_number])
            add_banner(
                image,
                f"VALIDATION  |  frame {frame_number:03d}/{frame_numbers[-1]:03d}",
                f"reconstructed native coverage: {native:.1%}",
            )
            writer.write(image)
            coverage.append({
                "frame": frame_number,
                "native_coverage": native,
                "visible_coverage": visible,
                "capture_records": len(by_frame[frame_number]),
            })
            if position % 25 == 0 or position == len(frame_numbers):
                print(f"validation: {position}/{len(frame_numbers)} frames", flush=True)
    finally:
        writer.release()

    return {
        "video": str(output),
        "frames": len(frame_numbers),
        "fps": fps,
        "duration_seconds": len(frame_numbers) / fps,
        "resolution": list(OUTPUT_SIZE),
        "native_complete_frames": sum(row["native_coverage"] == 1.0 for row in coverage),
        "fully_visible_frames": sum(row["visible_coverage"] == 1.0 for row in coverage),
        "coverage": coverage,
    }


def class_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode()).digest()
    # Bright BGR colours remain legible over the aerial imagery.
    return tuple(96 + channel % 160 for channel in digest[:3])


def draw_annotations(image: np.ndarray, annotations: list[dict]) -> None:
    scale_x = OUTPUT_SIZE[0] / SOURCE_WIDTH
    scale_y = OUTPUT_SIZE[1] / SOURCE_HEIGHT
    for annotation in annotations:
        label = annotation["object_id"]
        x1, y1, x2, y2 = annotation["bbox"]
        p1 = (round(x1 * scale_x), round(y1 * scale_y))
        p2 = (round(x2 * scale_x), round(y2 * scale_y))
        color = class_color(label)
        cv2.rectangle(image, p1, p2, color, 2, cv2.LINE_AA)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        tx = max(0, min(p1[0], OUTPUT_SIZE[0] - text_width - 8))
        ty = max(text_height + 8, p1[1])
        cv2.rectangle(
            image,
            (tx, ty - text_height - 7),
            (tx + text_width + 7, ty + baseline + 2),
            color,
            -1,
        )
        cv2.putText(image, label, (tx + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (10, 10, 10), 1, cv2.LINE_AA)


def compile_train(train_root: Path, output: Path, fps: float) -> dict:
    image_paths = sorted((train_root / "images").glob("frame_*.png"))
    if not image_paths:
        raise RuntimeError(f"No train images found under {train_root / 'images'}")

    writer = open_writer(output, fps, OUTPUT_SIZE)
    object_count = 0
    try:
        for position, image_path in enumerate(image_paths, start=1):
            frame_number = int(image_path.stem.split("_")[-1])
            annotation_path = train_root / "annotations" / f"frame_{frame_number:06d}.json"
            annotation_data = json.loads(annotation_path.read_text())
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")
            image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            annotations = annotation_data["annotations"]
            object_count += len(annotations)
            draw_annotations(image, annotations)
            add_banner(
                image,
                f"TRAIN  |  frame {position:03d}/{len(image_paths):03d}",
                f"{len(annotations)} labelled objects",
            )
            writer.write(image)
            print(f"train: {position}/{len(image_paths)} frames", flush=True)
    finally:
        writer.release()

    return {
        "video": str(output),
        "frames": len(image_paths),
        "fps": fps,
        "duration_seconds": len(image_paths) / fps,
        "resolution": list(OUTPUT_SIZE),
        "annotations_drawn": object_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-captures", type=Path, nargs="+", required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = compile_validation(
        args.validation_captures, args.output_dir / "drone-validation.mp4", args.fps
    )
    train = compile_train(
        args.train_root, args.output_dir / "drone-train-annotated.mp4", args.fps
    )
    manifest = {"validation": validation, "train": train}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "validation_video": validation["video"],
        "train_video": train["video"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render representative ground-object crops from the labelled training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    "large_launcher",
    "medium_launcher",
    "small_launcher",
    "tank",
    "jammer",
    "mine_roller",
    "ta-ta",
    "small_tower",
    "large_tower",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional class names to render instead of the default ground-object set.",
    )
    return parser.parse_args()


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in indices]


def make_cell(path: Path, width: int = 210, height: int = 170) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    available_height = height - 32
    scale = min((width - 12) / image.shape[1], available_height / image.shape[0], 3.0)
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    x = (width - resized.shape[1]) // 2
    y = (available_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    label = f"{path.parent.name}  {image.shape[1]}x{image.shape[0]}"
    cv2.putText(canvas, label, (7, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    cell_width, cell_height = 210, 170
    label_width = 190
    rows: list[np.ndarray] = []
    for class_name in args.classes or CLASSES:
        paths: list[Path] = []
        for level in ("L0", "L1", "L2"):
            level_paths = sorted((args.crops / class_name / level).glob("*.png"))
            if level_paths:
                paths.extend(evenly_spaced(level_paths, max(1, args.samples // 3)))
        paths = paths[: args.samples]
        if not paths:
            continue
        label = np.full((cell_height, label_width, 3), 8, dtype=np.uint8)
        cv2.putText(label, class_name, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cells = [label] + [make_cell(path, cell_width, cell_height) for path in paths]
        while len(cells) < args.samples + 1:
            cells.append(np.full((cell_height, cell_width, 3), 24, dtype=np.uint8))
        rows.append(cv2.hconcat(cells))
    output = cv2.vconcat(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f"Could not write {args.output}")
    print({"classes": len(rows), "output": str(args.output)})


if __name__ == "__main__":
    main()

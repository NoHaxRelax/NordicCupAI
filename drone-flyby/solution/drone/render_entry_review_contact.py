#!/usr/bin/env python3
"""Tile four native-detail entry review sheets per page without downscaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.glob("entry-f*.png"))
    if not paths:
        raise SystemExit("No entry review sheets found")
    args.output.mkdir(parents=True, exist_ok=True)
    pages, records = [], []
    for page_index, start in enumerate(range(0, len(paths), 4), start=1):
        loaded = [(path, cv2.imread(str(path), cv2.IMREAD_COLOR)) for path in paths[start : start + 4]]
        if any(image is None for _, image in loaded):
            raise SystemExit("Unable to read an entry review sheet")
        height, width = loaded[0][1].shape[:2]
        page = np.zeros((height * 2, width * 2, 3), np.uint8)
        for slot, (path, image) in enumerate(loaded):
            if image.shape[:2] != (height, width):
                raise ValueError(f"Inconsistent sheet size: {path}")
            x, y = (slot % 2) * width, (slot // 2) * height
            page[y : y + height, x : x + width] = image
            records.append({"file": path.name, "page": page_index, "slot": slot})
        filename = f"entry-review-contact-{page_index:02d}.jpg"
        cv2.imwrite(str(args.output / filename), page, [cv2.IMWRITE_JPEG_QUALITY, 94])
        pages.append(filename)
    (args.output / "manifest.json").write_text(json.dumps({"pages": pages, "sheets": records, "downscaled": False}, indent=2) + "\n")
    print(json.dumps({"sheets": len(paths), "pages": len(pages), "page_pixels": [width * 2, height * 2]}))


if __name__ == "__main__":
    main()

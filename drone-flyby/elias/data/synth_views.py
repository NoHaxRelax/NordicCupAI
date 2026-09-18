"""Compact look-at sheets for a synthetic dataset: real windows next to synthetic ones.

The contact sheets that synth.py writes are large (3000 px and more). This tool writes one small
JPEG per window scale with a few classes per row: 3 real centred windows on the left, 6 synthetic
ones on the right, every 64 px tile enlarged x3 with nearest neighbour so single pixels stay
visible. It reads two SPEC npz files and nothing else.

Usage (run from drone-flyby/):

  python elias/data/synth_views.py --real elias/out/real_helsinki.npz \
      --synthetic elias/out/synth_demo_helsinki.npz --out-prefix elias/out/synth_demo_helsinki_view

writes <prefix>_s1.jpg, <prefix>_s2.jpg and <prefix>_s4.jpg (scales without samples are skipped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from dtos import OBJECT_CLASSES  # noqa: E402

CLASSES = list(OBJECT_CLASSES)
TILE = 192


def pick(data, class_index, scale, count, rng, centred_only):
    rows = (data["label"] == class_index) & (data["scale"] == scale)
    if centred_only:
        rows &= np.abs(data["off"]).max(axis=1) == 0
        if "fill" in data.files:
            rows &= data["fill"] == 0
    index = np.nonzero(rows)[0]
    if len(index) > count:
        index = rng.choice(index, size=count, replace=False)
    return index


def sheet(real, synthetic, scale, classes, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for name in classes:
        class_index = CLASSES.index(name)
        real_index = pick(real, class_index, scale, 3, rng, centred_only=True)
        synthetic_index = pick(synthetic, class_index, scale, 6, rng, centred_only=False)
        if len(real_index) == 0 and len(synthetic_index) == 0:
            continue
        tiles = []
        for data, index, slots in ((real, real_index, 3), (synthetic, synthetic_index, 6)):
            for i in index:
                tiles.append(cv2.resize(data["x"][i], (TILE, TILE), interpolation=cv2.INTER_NEAREST))
            tiles.extend([np.full((TILE, TILE, 3), 32, np.uint8)] * (slots - len(index)))
            if slots == 3:
                tiles.append(np.full((TILE, 12, 3), 255, np.uint8))   # white bar: real | synthetic
        row = np.hstack(tiles)
        cv2.putText(row, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    return np.vstack(rows) if rows else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Small real-against-synthetic sheets per window scale.")
    parser.add_argument("--real", required=True, help="SPEC npz with real windows")
    parser.add_argument("--synthetic", required=True, help="SPEC npz written by synth.py")
    parser.add_argument("--out-prefix", required=True, help="output path without _s<scale>.jpg")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-rows", type=int, default=6, help="classes per sheet")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)

    real = np.load(args.real)
    synthetic = np.load(args.synthetic)
    for scale in args.scales:
        present = [CLASSES[c] for c in range(len(CLASSES))
                   if np.any((synthetic["label"] == c) & (synthetic["scale"] == scale))
                   and np.any((real["label"] == c) & (real["scale"] == scale))]
        if not present:
            continue
        step = max(1, len(present) // args.max_rows)
        image = sheet(real, synthetic, scale, present[::step][:args.max_rows], args.seed)
        if image is None:
            continue
        path = f"{args.out_prefix}_s{scale}.jpg"
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[synth_views] wrote {path} ({image.shape[1]}x{image.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

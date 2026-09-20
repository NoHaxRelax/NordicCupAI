#!/usr/bin/env python3
"""Certified-empty negative windows on validation frames (default: the dev and reserved blocks, 100-249).

Not mined from the detector: windows are sampled by position, kept only when they clear every v8 label box and every
hidden-object zone (Elias's validation_hidden*.json) by --margin source px, are not water or forest by Elias's terrain
rule, and mostly hold open ground. Half of the picks per frame are drawn uniformly, half from the most textured
candidates (roof stripes, kerbs, rail beds: the textures the verifier leaked on), and grass is capped so paved, sand
and mixed ground get their share. Output: a crop_picker export (kind negative, split from the frame) for cut_crops.py,
plus a contact sheet at the L1 look.

    python3 sample_negatives.py --frames /root/data/reconstructed-validation --v8 labels.json \
        --hidden validation_hidden.json validation_hidden2.json --first 100 --last 249 --per-frame 8 --out marks-dev-reserved.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrain import classify  # noqa: E402

W, H = 3840, 2160
SCENES = {'validation': {'dir_from_root': 'data/drone/reconstructed-validation', 'pattern': 'frame_%06d.png'}}


def split_of(frame):
    return 'train' if frame <= 90 else 'gap' if frame <= 99 else 'dev' if frame <= 180 else 'reserved'


def keepout_boxes(v8, hidden_files, frame):
    boxes = []
    for o in v8.get(str(frame), []):
        b = o['bbox']; boxes.append([b[0] * W, b[1] * H, b[2] * W, b[3] * H])
    for hf in hidden_files:
        z = hf.get(str(frame))
        if z is None:
            continue
        if isinstance(z[0], (int, float)):
            boxes.append(z)
        else:
            boxes.extend(z)
    return boxes


def clears(x, y, half, boxes, margin):
    x1, y1, x2, y2 = x - half - margin, y - half - margin, x + half + margin, y + half + margin
    return not any(b[0] < x2 and b[2] > x1 and b[1] < y2 and b[3] > y1 for b in boxes)


def open_ground(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32); s, v = hsv[:, :, 1] / 255, hsv[:, :, 2] / 255
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    tex = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (9, 9))
    hue = hsv[:, :, 0] * 2
    canopy = (hue > 60) & (hue < 170) & (s > 0.25) & (v < 0.45) & (tex > 12)
    shadow = v < 0.18
    return float(canopy.mean()), float((~canopy & ~shadow & (v > 0.30)).mean()), float(tex.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--v8', type=Path, required=True)
    ap.add_argument('--hidden', type=Path, nargs='+', required=True)
    ap.add_argument('--first', type=int, default=100); ap.add_argument('--last', type=int, default=249)
    ap.add_argument('--per-frame', type=int, default=8)
    ap.add_argument('--window', type=int, default=128); ap.add_argument('--margin', type=int, default=64)
    ap.add_argument('--grass-share', type=float, default=0.45)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sheet', type=Path, default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    v8 = json.loads(a.v8.read_text())['predictions_by_frame']
    hidden = [json.loads(h.read_text())['zones'] for h in a.hidden]
    half = a.window // 2
    marks, terrain_counts, tiles = [], Counter(), []
    for frame in range(a.first, a.last + 1):
        img = cv2.imread(str(a.frames / f'frame_{frame:06d}.png'), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        alpha = img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else None
        bgr = img[:, :, :3]
        boxes = keepout_boxes(v8, hidden, frame)
        cands = []
        for _ in range(400):
            x = rng.randint(half + 8, W - half - 8); y = rng.randint(half + 8, H - half - 8)
            if not clears(x, y, half, boxes, a.margin):
                continue
            if alpha is not None and (alpha[y - half:y + half, x - half:x + half] == 0).any():
                continue
            patch = bgr[y - half:y + half, x - half:x + half]
            t = classify(patch)
            if t in ('water', 'forest'):
                continue
            canopy, og, tex = open_ground(patch)
            if canopy > 0.30 or og < 0.45:
                continue
            cands.append((x, y, t, tex))
        if not cands:
            continue
        rng.shuffle(cands)
        n_tex = a.per_frame // 2
        by_tex = sorted(cands, key=lambda c: -c[3])
        chosen = []

        def take(pool, target):
            for x, y, t, tex in pool:
                if len(chosen) >= target:
                    break
                if any(abs(x - cx) < a.window and abs(y - cy) < a.window for cx, cy, _, _ in chosen):
                    continue
                if t == 'grass' and terrain_counts['grass'] >= a.grass_share * max(len(marks) + len(chosen), 20):
                    continue
                chosen.append((x, y, t, tex)); terrain_counts[t] += 1
        take(by_tex, n_tex)            # the most textured candidates first: roofs, kerbs, rails
        take(cands, a.per_frame)       # then uniform picks up to the per-frame budget
        for x, y, t, tex in chosen:
            # class tag round-robin over the four small classes, as the crop picker's auto negatives do: a terrain
            # negative serves every class (the trainer maps kind=negative to background whatever the tag)
            tag = ['small_launcher', 'medium_launcher', 'ta-ta', 'jammer'][len(marks) % 4]
            marks.append({'id': f'validation-{frame}-{x}-{y}-auto2', 'scene': 'validation', 'frame': frame, 'x': float(x), 'y': float(y),
                          'cls': tag, 'kind': 'negative', 'window_source_px': a.window, 'near_label': False, 'split': split_of(frame),
                          'terrain': t, 'texture': round(tex, 1), 'source': 'sample_negatives.py: clears v8 labels and hidden zones by %d px' % a.margin})
            if a.sheet is not None and len(tiles) < 400:
                tiles.append(cv2.resize(bgr[y - half:y + half, x - half:x + half], (64, 64), interpolation=cv2.INTER_AREA))
    a.out.write_text(json.dumps({'version': 1, 'tool': 'sample_negatives.py', 'crop_spec': {'window_source_px': a.window}, 'scenes': SCENES, 'marks': marks}))
    print(f'{len(marks)} negatives on frames {a.first}-{a.last}: terrain {dict(terrain_counts)}, splits {dict(Counter(m["split"] for m in marks))}')
    if a.sheet is not None and tiles:
        cols = 20; rows = (len(tiles) + cols - 1) // cols; cell = 96
        sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols); sheet[r * cell:r * cell + cell, c * cell:c * cell + cell] = cv2.resize(t, (cell, cell), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(a.sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()

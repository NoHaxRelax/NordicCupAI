"""Terrain under a box, by simple colour rules on BGR pixels (measured on 2026-09-19, elias/backdrop_study.py).

Objects in both known flights stand on open ground: bare soil, aprons, courts, grass, clearings. None stand on
water and 6 % have tree cover in their surroundings against a 24 % random prior. Two uses: the synthetic generator
refuses paste positions whose surroundings are water or forest (SYNTH_TERRAIN=1), and the detector hook can
down-weight boxes whose surroundings are water or forest (ELIAS_CONTEXT). Rules work on native and on delivered
pixels; the forest texture cue is weaker after downscaling, so forest also keys on darkness.
"""
from __future__ import annotations

import cv2
import numpy as np


def classify(pixels_bgr):
    """'water' | 'forest' | 'grass' | 'paved' | 'sand' | 'other' for an (N,1,3) or (H,W,3) uint8 BGR array."""
    arr = np.asarray(pixels_bgr, np.uint8).reshape(-1, 1, 3)
    if arr.shape[0] < 16:
        return 'other'
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    h, s, v = hsv[:, 0]*2, hsv[:, 1]/255, hsv[:, 2]/255
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).reshape(-1).astype(np.float32)
    tex = float(np.std(np.diff(gray))) if gray.size > 2 else 0.
    green = np.mean((h > 60) & (h < 170) & (s > 0.2) & (v > 0.1))
    blue = np.mean((h > 185) & (h < 250) & (s > 0.25))
    grey = np.mean((s < 0.18) & (v > 0.25))
    warm = np.mean((h < 60) & (s > 0.12) & (v > 0.45))
    mv, ms = float(np.mean(v)), float(np.mean(s))
    if blue > 0.45 or (mv < 0.22 and tex < 6):
        return 'water'
    if green > 0.5 and (mv < 0.42 or tex > 22):
        return 'forest'
    if green > 0.45:
        return 'grass'
    if grey > 0.5:
        return 'paved'
    if warm > 0.4 or (ms < 0.3 and mv > 0.55):
        return 'sand'
    return 'other'


def ring_pixels(img, box, grow=1.0):
    """Pixels of the ring around `box` (x1, y1, x2, y2) in `img`, the box itself blanked; None if empty."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]; w, h = max(1, x2-x1), max(1, y2-y1)
    X1, Y1 = max(0, int(x1-grow*w)), max(0, int(y1-grow*h)); X2, Y2 = min(img.shape[1], int(x2+grow*w)), min(img.shape[0], int(y2+grow*h))
    if X2-X1 < 4 or Y2-Y1 < 4:
        return None
    patch = img[Y1:Y2, X1:X2].copy()
    patch[max(0, y1-Y1):max(0, y2-Y1), max(0, x1-X1):max(0, x2-X1)] = 0
    pix = patch.reshape(-1, 3); pix = pix[pix.sum(1) > 0]
    return pix.reshape(-1, 1, 3) if len(pix) >= 16 else None


def implausible(img, box, grow=1.0):
    """True when the surroundings of `box` are water or forest."""
    pix = ring_pixels(img, box, grow)
    return pix is not None and classify(pix) in ('water', 'forest')

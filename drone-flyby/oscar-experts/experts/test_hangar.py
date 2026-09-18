"""Hangar expert on synthetic placements of its own training sprites. No dev data is read."""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from .common import iou, load_templates
from .hangar import HangarExpert, HangarSettings

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / 'data/drone/expert-bank-20260918-v1'


def apron(h, w, seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (150, 165, 175), np.uint8)
    noise = rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def place(canvas, template, angle, x, y, scale=1.):
    bgr, mask = template.posed(angle, scale)
    h, w = mask.shape
    region = canvas[y:y + h, x:x + w]
    region[mask] = bgr[mask]
    ys, xs = np.where(mask)
    return [x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1]


@unittest.skipUnless(BANK.exists(), 'expert bank not built')
class HangarExpertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert = HangarExpert(BANK)
        cls.templates = load_templates(BANK, 'hangar')

    def _detect_box(self, image, scale=1., zoom=2):
        rows, _ = self.expert.detect(image, scale, zoom)
        return rows

    def test_finds_rotated_sprite_at_native_scale(self):
        t = self.templates[1]
        for angle in (0., 35., 120., 250.):
            canvas = apron(540, 960, seed=int(angle))
            truth = place(canvas, t, angle, 380, 200)
            rows = self._detect_box(canvas)
            self.assertTrue(rows, f'no detection at {angle} deg')
            best = max(iou(r['mask_bbox'], truth) for r in rows)
            self.assertGreaterEqual(best, .6, f'mask box IoU {best:.2f} at {angle} deg')

    def test_finds_sprite_at_level1_scale(self):
        t = next(x for x in self.templates if x.zoom == 1)
        canvas = apron(540, 960, seed=7)
        truth = place(canvas, t, 60., 500, 300, scale=.5)
        rows = self._detect_box(canvas, .5, 1)
        self.assertTrue(rows)
        self.assertGreaterEqual(max(iou(r['mask_bbox'], truth) for r in rows), .6)

    def test_partial_at_top_edge_is_still_found(self):
        t = self.templates[1]
        canvas = apron(540, 960, seed=3)
        bgr, mask = t.posed(90., 1.)
        h, w = mask.shape
        cut = h // 2
        region = canvas[0:h - cut, 400:400 + w]
        region[mask[cut:]] = bgr[cut:][mask[cut:]]
        rows = self._detect_box(canvas)
        self.assertTrue(rows, 'half-visible hangar at the top edge was not found')
        self.assertTrue(rows[0]['partial'])
        self.assertLess(rows[0]['visible_fraction'], .75)

    def test_dark_blob_without_rim_is_a_low_scoring_candidate(self):
        canvas = np.full((540, 960, 3), 45, np.uint8)  # dim canopy: above the dark threshold, no rim contrast
        cv2.ellipse(canvas, (480, 270), (90, 50), 20, 0, 360, (6, 6, 6), -1)
        rows, _, candidates = self.expert.detect(canvas, 1., 2, explain=True)
        self.assertTrue(rows, 'the blob must be listed for the verifier')
        self.assertLess(rows[0]['rim_contrast'], 50)

    def test_rejects_wrong_size(self):
        t = self.templates[1]
        canvas = apron(540, 960, seed=11)
        place(canvas, t, 0., 300, 200, scale=.4)  # far too small for a native view
        self.assertFalse(self._detect_box(canvas))

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            HangarSettings(min_long_side=300, max_long_side=200)


if __name__ == '__main__':
    unittest.main()

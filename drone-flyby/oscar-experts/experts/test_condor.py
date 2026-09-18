"""Condor expert on synthetic placements of its own training sprites. No dev data is read."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from .common import iou, load_templates
from .condor import CondorExpert, CondorSettings

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / 'data/drone/expert-bank-20260918-v1'


def ground(h, w, seed=0, level=(120, 135, 140)):
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), level, np.uint8).astype(np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 1.5)[:, :, None]
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def warp_sprite(template, angle, scale=1., stretch=1., shear=0., gain=1., offset=0.):
    """Rotate, scale, stretch and shear the flat-filled sprite and its mask; returns (bgr, mask)."""
    bgr, mask = template.bgr, template.mask.astype(np.uint8) * 255
    h, w = mask.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    affine = np.array([[stretch, np.tan(np.radians(shear))], [0., 1.]], np.float32)
    matrix[:, :2] = affine @ matrix[:, :2]
    corners = cv2.transform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), matrix)[0]
    low, high = np.floor(corners.min(0)), np.ceil(corners.max(0))
    matrix[:, 2] -= low
    size = tuple((high - low).astype(int))
    out = cv2.warpAffine(bgr, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    return out, cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST) > 127


def place(canvas, bgr, mask, x, y):
    h, w = mask.shape
    region = canvas[y:y + h, x:x + w]
    region[mask] = bgr[mask]
    ys, xs = np.where(mask)
    return [x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1]


@unittest.skipUnless(BANK.exists(), 'expert bank not built')
class CondorExpertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert = CondorExpert(BANK)
        cls.templates = load_templates(BANK, 'condor')
        cls.native = next(t for t in cls.templates if t.zoom == 2)

    def _run(self, canvas, scale=1., zoom=2):
        families, candidates = self.expert.detect(canvas, scale, zoom, explain=True)
        return families['condor_expert'], families, candidates

    def _assert_found(self, rows, truth, what):
        self.assertTrue(rows, f'{what}: no detection')
        best = max(iou(r['outline_bbox'], truth) for r in rows)
        self.assertGreaterEqual(best, .5, f'{what}: outline IoU {best:.2f}')

    def test_any_heading(self):
        for angle in (0., 75., 190., 300.):
            canvas = ground(540, 960, seed=int(angle))
            bgr, mask = warp_sprite(self.native, angle)
            truth = place(canvas, bgr, mask, 400, 200)
            rows, _, _ = self._run(canvas)
            self._assert_found(rows, truth, f'heading {angle}')

    def test_perspective_like_distortion(self):
        canvas = ground(540, 960, seed=5)
        bgr, mask = warp_sprite(self.native, 40., scale=1.05, stretch=.86, shear=9.)
        truth = place(canvas, bgr, mask, 350, 180)
        rows, _, _ = self._run(canvas)
        self._assert_found(rows, truth, 'stretched and sheared')

    def test_lighting_change(self):
        canvas = ground(540, 960, seed=9, level=(110, 128, 150))
        bgr, mask = warp_sprite(self.native, 120., gain=.82, offset=12.)
        truth = place(canvas, bgr, mask, 420, 230)
        rows, _, _ = self._run(canvas)
        self._assert_found(rows, truth, 'darker, offset lighting')

    def test_level1_scale(self):
        t = next(x for x in self.templates if x.zoom == 1)
        canvas = ground(540, 960, seed=3)
        bgr, mask = warp_sprite(t, 250., scale=.5)
        truth = place(canvas, bgr, mask, 500, 300)
        rows, _, _ = self._run(canvas, .5, 1)
        self._assert_found(rows, truth, 'level 1')

    def test_partial_at_top_edge(self):
        canvas = ground(540, 960, seed=4)
        bgr, mask = warp_sprite(self.native, 30.)
        h, w = mask.shape
        cut = int(h * .45)
        region = canvas[0:h - cut, 400:400 + w]
        region[mask[cut:]] = bgr[cut:][mask[cut:]]
        rows, _, _ = self._run(canvas)
        self.assertTrue(rows, 'half-visible condor at the top edge was not found')
        self.assertTrue(rows[0]['partial'])

    def test_both_aircraft_become_candidates(self):
        """The expert lists candidates; it does not judge them. Both objects must be in the list."""
        other = max(load_templates(BANK, 'medium_plane'), key=lambda x: x.zoom)
        canvas = ground(540, 960, seed=8)
        bgr, mask = warp_sprite(other, 60.)
        plane = place(canvas, bgr, mask, 700, 380)
        bgr, mask = warp_sprite(self.native, 20.)
        truth = place(canvas, bgr, mask, 250, 150)
        rows, _, _ = self._run(canvas)
        self.assertTrue(any(iou(r['outline_bbox'], truth) >= .5 for r in rows), 'condor missing from the candidate list')
        self.assertTrue(all(r.get('rejected_by') is None for r in rows))

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            CondorSettings(min_pattern=1.5)


if __name__ == '__main__':
    unittest.main()

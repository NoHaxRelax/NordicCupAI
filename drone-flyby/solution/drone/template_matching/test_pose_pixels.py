import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .detector import iou, sha
from .pose_pixels import PosePixelDetector


class PosePixelTests(unittest.TestCase):
    def test_occluded_asset_and_native_launcher_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory)
            rng = np.random.default_rng(813)
            patch = rng.integers(20, 230, (40, 32, 3), dtype=np.uint8)
            mask = np.full((40, 32), 255, np.uint8)
            cv2.imwrite(str(bank/'sprite.png'), patch)
            cv2.imwrite(str(bank/'mask.png'), mask)
            rows = [dict(id=label, file='sprite.png', sha256=sha(bank/'sprite.png'),
                         mask_file='mask.png', mask_sha256=sha(bank/'mask.png'),
                         calibration=True, **{'class': label})
                    for label in ['jet_plane', 'medium_launcher']]
            (bank/'manifest.json').write_text(json.dumps(dict(templates=rows)))
            model = PosePixelDetector(bank)
            image = np.full((100, 150, 3), 100, np.uint8)
            image[30:70, 60:92] = patch
            image[30:41, 60:92] = 100  # upper part occluded
            predictions = model.detect(image)
            self.assertTrue(any(r['class'] == 'jet_plane' and
                                r['template_id'].endswith('-lower') and
                                iou(r['bbox'], [60, 30, 92, 70]) > .99
                                for r in predictions))
            self.assertEqual(predictions, model.detect(image))
            variants = model.variants(1.)
            self.assertEqual({r[1] for r in variants if r[0]['class'] == 'medium_launcher'},
                             {.9, 1., 1.1})
            self.assertEqual(model.detect(np.full_like(image, 100)), [])
            with self.assertRaises(ValueError):
                model.detect(image, float('nan'))


if __name__ == '__main__':
    unittest.main()

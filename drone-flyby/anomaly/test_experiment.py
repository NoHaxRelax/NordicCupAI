import unittest

import numpy as np

from anomaly.experiment import iou, proposals, render


class ExperimentTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_render_centers_and_scales_target(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        box = [1900, 1000, 1940, 1080]
        for zoom, expected_width in ((0, 10), (1, 20), (2, 40)):
            view, local, region = render(image, box, zoom)
            self.assertEqual(view.shape, (540, 960, 3))
            self.assertAlmostEqual(local[2] - local[0], expected_width)
            self.assertEqual(region[2] - region[0], 3840 // 2**zoom)

    def test_proposals_are_class_free_and_bounded(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)
        found = proposals(image, limit=12)
        self.assertLessEqual(len(found), 12)
        for score, box in found:
            self.assertIsInstance(score, float)
            self.assertEqual(len(box), 4)
            self.assertTrue(0 <= box[0] < box[2] <= 960)
            self.assertTrue(0 <= box[1] < box[3] <= 540)


if __name__ == '__main__':
    unittest.main()

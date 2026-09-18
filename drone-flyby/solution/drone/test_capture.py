"""Check geometry against the official camera, and lossless pixel reconstruction."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'artifacts/drone-source-2026-09-17'))
from local_evaluator import Camera, build_request
from dtos import DroneFlybyPredictResponseDto
from capture import next_view, save_request
from reconstruct import reconstruct


class CaptureTests(unittest.TestCase):
    def test_all_sixteen_tiles_reachable_on_third_request(self):
        for y in (270, 810, 1350, 1890):
            for x in (480, 1440, 2400, 3360):
                camera = Camera()
                for i in range(2):
                    req = build_request(i, i, camera, '', None)
                    camera.apply(**next_view(req, (x, y)))
                self.assertEqual((camera.resolution_level, camera.center_x, camera.center_y), (2, x, y))

    def test_capture_and_reconstruct_exact_pixels(self):
        source = np.random.default_rng(2026).integers(0, 256, (2160, 3840, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for y in (270, 810, 1350, 1890):
                for x in (480, 1440, 2400, 3360):
                    camera = Camera(2, x, y)
                    x1, y1, x2, y2 = camera.source_region
                    stream = io.BytesIO()
                    Image.fromarray(source[y1:y2, x1:x2]).save(stream, format='PNG')
                    req = build_request(42, 41, camera, base64.b64encode(stream.getvalue()).decode(), None)
                    response = save_request(req, root, (x, y))
                    DroneFlybyPredictResponseDto.model_validate(response)
            records = [(p, json.loads(p.read_text())) for p in root.rglob('*.json')]
            rgb, mask = reconstruct(records)
            self.assertTrue(mask.all())
            np.testing.assert_array_equal(rgb, source)
            _, partial = reconstruct(records[:1])
            self.assertEqual(partial.mean(), 1/16)
            bad_path, bad_record = records[0]
            damaged = dict(bad_record, image_file='damaged.png')
            Image.new('RGB', (960, 540)).save(bad_path.parent / 'damaged.png')
            with self.assertRaisesRegex(ValueError, 'Conflicting pixels'):
                reconstruct(records + [(bad_path, damaged)])


if __name__ == '__main__':
    unittest.main()

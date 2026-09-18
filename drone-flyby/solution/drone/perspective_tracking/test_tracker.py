"""Noise-free geometry, runtime and integration regression tests."""
import json
from pathlib import Path
import unittest

import numpy as np

from . import (CalibrationError, DeterministicTracker, MotionModel,
               ProjectionError, ViewGeometry, calibrate_images)


ROOT = Path(__file__).resolve().parents[2]


class MotionTests(unittest.TestCase):
    def geometry(self, velocity):
        intrinsic = np.array([[1200., 0, 1920.], [0, 1200., 1080.], [0, 0, 1.]])
        normal, distance = np.array([.1, -.15, 1.]), 1000.
        xy = np.array([[x, y] for x in np.linspace(-500, 500, 7) for y in np.linspace(-350, 350, 5)])
        world = np.c_[xy, (distance-xy@normal[:2])/normal[2]]
        def pixels(tick):
            homogeneous = (world-tick*velocity)@intrinsic.T
            return homogeneous[:, :2]/homogeneous[:, 2, None]
        return pixels

    def test_fit_matches_exact_3d_with_arbitrary_origin_and_interval(self):
        # Includes parallel image flow with a vanishing point at infinity.
        for velocity in (np.array([3., 2., 1.]), np.array([3., 1., -.15]), np.array([3., 2., 0.])):
            with self.subTest(velocity=velocity):
                pixels = self.geometry(velocity)
                model = MotionModel.from_matches(pixels(7), pixels(9.5), first_tick=7, second_tick=9.5)
                np.testing.assert_allclose(model.points(pixels(12), 12, 25), pixels(25), atol=1e-6)
                combined = model.mapping(19, 25)@model.mapping(12, 19)
                np.testing.assert_allclose(combined, model.mapping(12, 25), atol=1e-10)

    def test_parallel_translation_in_image(self):
        points = np.array([[x, y] for x in (100., 800., 2200.) for y in (100., 600., 1200., 1900.)])
        model = MotionModel.from_matches(points, points+[2, 50])
        np.testing.assert_allclose(model.points(points, 1, 11), points+[20, 500], atol=1e-6)

    def test_zero_motion_and_line_of_features_do_not_fake_calibration(self):
        points = np.array([[x, y] for x in (100., 800., 2200.) for y in (100., 600., 1200., 1900.)])
        with self.assertRaises(CalibrationError):
            MotionModel.from_matches(points, points)
        line = np.c_[np.arange(15)*100., np.ones(15)*500.]
        with self.assertRaises(CalibrationError):
            MotionModel.from_matches(line, line+[0, 50])
        with self.assertRaises(CalibrationError):
            MotionModel.from_matches(points, points+[0, 50], first_tick=2, second_tick=2)

    def test_invalid_models_and_projective_horizon(self):
        with self.assertRaises(ValueError):
            MotionModel(np.eye(3), (3840, 2160), 0, 1)
        model = MotionModel(np.diag([0., 0., -.1]), (3840, 2160), 0, 1)
        with self.assertRaises(ProjectionError):
            model.box([10, 20, 40, 60], 1, 10)
        with self.assertRaises(ValueError):
            model.mapping(1, float('nan'))

    def test_blank_images_fail_explicitly(self):
        blank = np.zeros((540, 960), np.uint8)
        view = ViewGeometry((3840, 2160), (0, 0, 3840, 2160), (960, 540))
        with self.assertRaises(CalibrationError):
            calibrate_images(blank, blank, view, view)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        matrix = np.zeros((3, 3)); matrix[:2, 2] = [2., 50.]
        self.model = MotionModel(matrix, (3840, 2160), 0, 1)
        self.tracker = DeterministicTracker(self.model, "example")
        self.tracker.add("tank-1", "tank", [100, 100, 140, 140], 1, confidence=.8)

    def request(self, index):
        return {"sequence_id": "example", "request_id": "request-42", "frame": 100+index,
                "frame_index": index, "original_width": 3840, "original_height": 2160,
                "view": {"width": 960, "height": 540, "source_region_xyxy": [1920, 0, 3840, 1080]}}

    def test_skipped_frames_and_global_answer_outside_current_crop(self):
        answer = self.tracker.response(self.request(6))
        np.testing.assert_allclose(answer['annotations'][0]['bbox'],
                                   np.array([110, 350, 150, 390])/[3840, 2160, 3840, 2160])
        self.assertEqual(answer['frame'], 106)
        self.assertEqual(answer['request_id'], 'request-42')
        self.assertEqual(set(answer['annotations'][0]), {'object_id', 'bbox', 'confidence'})
        self.assertEqual(self.tracker.tracks[0].tick, 1)

    def test_queries_do_not_change_geometry_or_grow_boxes(self):
        matrix = np.outer([.3, 1., .0001], [.001, -.003, 20.])
        tracker = DeterministicTracker(MotionModel(matrix, (3840, 2160), 0, 1), "x")
        tracker.add("a", "tank", [400, 400, 440, 440], 1)
        expected = tracker.predict("a", 20)
        for tick in range(2, 20):
            tracker.predict("a", tick)
        self.assertEqual(expected, tracker.predict("a", 20))
        self.assertEqual(tracker.tracks[0].box, (400, 400, 440, 440))

    def test_crop_coordinates_and_pixel_centres(self):
        view = ViewGeometry((3840, 2160), (1920, 0, 3840, 1080), (960, 540))
        box = [100, 200, 120, 240]
        source = view.box_to_source(box)
        np.testing.assert_allclose(source, [2120, 400, 2160, 480])
        np.testing.assert_allclose(view.box_from_source(source), box)
        np.testing.assert_allclose(view.box_to_source(np.array(box)/[960, 540, 960, 540], normalized=True), source)
        np.testing.assert_allclose(view.points_to_source([[0, 0]]), [[1920.5, .5]])
        self.tracker.add_view_detection("tank-2", "tank", box, 1, view)
        self.assertEqual(len(self.tracker.response(self.request(2))['annotations']), 2)

    def test_explicit_fractional_motion_clock(self):
        unchanged = self.tracker.response(self.request(2), tick=1)
        np.testing.assert_allclose(unchanged['annotations'][0]['bbox'],
                                   np.array([100, 100, 140, 140])/[3840, 2160, 3840, 2160])
        halfway = self.tracker.predict("tank-1", 1.5)
        np.testing.assert_allclose(halfway['bbox_source_xyxy'], [101, 125, 141, 165])

    def test_first_frame_detection_can_be_retained_during_calibration(self):
        self.tracker.add('early', 'tank', [500, 500, 540, 540], 0)
        np.testing.assert_allclose(self.tracker.predict('early', 1)['bbox_source_xyxy'], [502, 550, 542, 590])
        with self.assertRaises(ValueError):
            self.tracker.predict('early', 0)

    def test_clipping_exit_and_optional_refresh(self):
        partial = self.tracker.predict('tank-1', 42)
        self.assertEqual(partial['bbox_source_xyxy'][3], 2160)
        self.assertIsNone(self.tracker.predict('tank-1', 43))
        self.tracker.refresh('tank-1', [300, 300, 340, 340], 2, confidence=.9)
        np.testing.assert_allclose(self.tracker.predict('tank-1', 3)['bbox_source_xyxy'], [302, 350, 342, 390])
        self.tracker.remove('tank-1')
        self.assertEqual(self.tracker.predictions(4), [])

    def test_input_and_sequence_guards(self):
        with self.assertRaises(ValueError):
            self.tracker.add('tank-1', 'tank', [0, 0, 1, 1], 1)
        with self.assertRaises(ValueError):
            self.tracker.add('bad', 'unknown', [0, 0, 1, 1], 1)
        with self.assertRaises(ValueError):
            self.tracker.add('early', 'tank', [0, 0, 1, 1], -1)
        with self.assertRaises(ValueError):
            self.tracker.add('bad', 'tank', [0, 0, float('nan'), 1], 1)
        with self.assertRaises(ValueError):
            self.tracker.response({**self.request(3), 'sequence_id': 'different'})
        with self.assertRaises(ValueError):
            self.tracker.response({**self.request(3), 'original_width': 1000})
        with self.assertRaises(ValueError):
            self.tracker.refresh('tank-1', [0, 0, 10, 10], 1)

    def test_json_state_roundtrip_preserves_predictions(self):
        saved = json.loads(json.dumps(self.tracker.to_dict(), allow_nan=False))
        restored = DeterministicTracker.from_dict(saved)
        self.assertEqual(self.tracker.response(self.request(20)), restored.response(self.request(20)))


class ReferenceRegressionTests(unittest.TestCase):
    def test_reproduces_saved_one_observation_experiment(self):
        matches = ROOT/'artifacts/drone-scene-analysis/matches-reference-000-001.npz'
        report = ROOT/'artifacts/drone-shared-motion/reference-measurements.json'
        if not matches.exists() or not report.exists():
            self.skipTest('Local reference calibration artifacts are not present')
        data = np.load(matches)
        model = MotionModel.from_matches(data['x'][data['train']], data['y'][data['train']])
        primary = json.loads(report.read_text())['primary_unmasked_one_box']
        old_model = MotionModel(primary['model']['A'], (3840, 2160), 0, 1)
        # Test unseen times and boxes, without using future labels to calibrate.
        for anchor in (1, 5, 12):
            box = [1700., 100., 1740., 140.]
            np.testing.assert_allclose(model.box(box, anchor, 24), old_model.box(box, anchor, 24), atol=.05)


if __name__ == '__main__':
    unittest.main()

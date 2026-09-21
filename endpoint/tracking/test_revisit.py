"""Revisit integration: crops, association, timing, camera legality and state."""
import base64
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

from . import (Detection, DroneTrackingWorkflow, LevelOneSweep, MotionModel,
               RevisitConfig, RevisitTracker, ViewGeometry)
from .workflow import observed_step


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_evaluator import Camera, CameraRejection
from dtos import DroneFlybyPredictResponseDto
from utils import validate_response


def request(frame, camera, sequence='test'):
    return {'sequence_id': sequence, 'frame': frame+100, 'frame_index': frame,
            'request_id': f'{sequence}-{frame}', 'original_width': 3840, 'original_height': 2160,
            'view': {'resolution_level': camera.resolution_level, 'center_x': camera.center_x,
                     'center_y': camera.center_y, 'width': 960, 'height': 540,
                     'source_region_xyxy': list(camera.source_region)},
            'camera_constraints': camera.constraints()}


class RevisitTests(unittest.TestCase):
    def setUp(self):
        a = np.zeros((3, 3)); a[1, 2] = 10
        self.model = MotionModel(a, (3840, 2160), 0., 1.)
        self.tracker = RevisitTracker(self.model, 'test')
        self.full = ViewGeometry((3840, 2160), (0, 0, 3840, 2160), (960, 540))
        self.left = ViewGeometry((3840, 2160), (0, 0, 1920, 1080), (960, 540))
        self.right = ViewGeometry((3840, 2160), (1920, 0, 3840, 1080), (960, 540))

    def detection(self, source, view=None, label='tank', confidence=.9, complete=True):
        return Detection(label, tuple((view or self.full).box_from_source(source)), confidence, complete)

    def test_revisit_refreshes_same_id_across_zoom_with_no_external_ids(self):
        self.tracker.update([self.detection([100, 100, 140, 140])], self.full, 1, 1)
        self.tracker.update([], self.right, 2, 2)
        result = self.tracker.update([self.detection([102, 122, 142, 159], self.left)], self.left, 3, 3)
        self.assertEqual(len(result), 1); self.assertEqual(result[0]['track_id'], 'track-00001')
        np.testing.assert_allclose(result[0]['bbox_source_xyxy'], [102, 122, 142, 159])
        np.testing.assert_allclose(self.tracker.predictions(4)[0]['bbox_source_xyxy'], [102, 132, 142, 169])

    def test_objects_outside_crop_survive_many_frames_and_visible_misses_retire(self):
        self.tracker.update([self.detection([100, 100, 140, 140])], self.full, 1, 1)
        for frame in range(2, 15): self.tracker.update([], self.right, frame, frame)
        self.assertEqual(len(self.tracker.tracks), 1)
        for frame in range(15, 18): self.tracker.update([], self.left, frame, frame)
        self.assertEqual(len(self.tracker.tracks), 0)

    def test_long_confident_track_is_not_overwritten_by_one_unsure_sighting(self):
        tracker = RevisitTracker(self.model, 'test', RevisitConfig(birth_confidence=.25, update_confidence=.15,
                                                                   confidence_memory=8, strong_confidence=.5))
        box = [100, 100, 140, 140]
        for frame in range(1, 11):
            shifted = [box[0], box[1]+10*(frame-1), box[2], box[3]+10*(frame-1)]
            rows = tracker.update([self.detection(shifted, confidence=.9)], self.full, frame, frame)
        self.assertEqual(len(tracker.tracks), 1); self.assertAlmostEqual(rows[0]['confidence'], .9)
        self.assertEqual(tracker.tracks['track-00001'].strong_sightings, 8)
        rows = tracker.update([self.detection([100, 200, 140, 240], confidence=.15)], self.full, 11, 11)
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['track_id'], 'track-00001')
        # One weak sighting moves 1/9 of the way down, not straight to 0.15.
        self.assertAlmostEqual(rows[0]['confidence'], .9-(.9-.15)/9)
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [100, 200, 140, 240])
        # Repeated weak sightings erode it one remembered strong sighting at a time.
        confidences = [rows[0]['confidence']]
        for frame in range(12, 22):
            rows = tracker.update([self.detection([100, 100+10*(frame-1), 140, 140+10*(frame-1)], confidence=.15)],
                                  self.full, frame, frame)
            confidences.append(rows[0]['confidence'])
        self.assertTrue(all(a >= b for a, b in zip(confidences, confidences[1:])))
        self.assertTrue(all(a > b for a, b in zip(confidences[:8], confidences[1:9])))
        self.assertAlmostEqual(confidences[-1], .15)
        self.assertEqual(tracker.tracks['track-00001'].strong_sightings, 0)
        # Once forgotten, a weak sighting overwrites at once and a strong one rises at once.
        rows = tracker.update([self.detection([100, 310, 140, 350], confidence=.15)], self.full, 22, 22)
        self.assertAlmostEqual(rows[0]['confidence'], .15)
        rows = tracker.update([self.detection([100, 320, 140, 360], confidence=.8)], self.full, 23, 23)
        self.assertAlmostEqual(rows[0]['confidence'], .8)

    def test_ema_and_peak_confidence_modes(self):
        for mode, expected in (('ema', .9-(.9-.15)*.25), ('peak', .9*.8)):
            tracker = RevisitTracker(self.model, 'test', RevisitConfig(birth_confidence=.25, update_confidence=.15,
                confidence_memory=8, confidence_mode=mode, confidence_alpha=.25, confidence_decay=.8))
            tracker.update([self.detection([100, 100, 140, 140], confidence=.9)], self.full, 1, 1)
            rows = tracker.update([self.detection([100, 110, 140, 150], confidence=.15)], self.full, 2, 2)
            self.assertAlmostEqual(rows[0]['confidence'], expected, msg=mode)
            rows = tracker.update([self.detection([100, 120, 140, 160], confidence=.95)], self.full, 3, 3)
            self.assertAlmostEqual(rows[0]['confidence'], .95, msg=mode)
        with self.assertRaises(ValueError):
            RevisitConfig(confidence_mode='mean')

    def test_confidence_memory_is_off_by_default_and_survives_state_round_trip(self):
        self.tracker.update([self.detection([100, 100, 140, 140], confidence=.9)], self.full, 1, 1)
        self.tracker.update([self.detection([100, 110, 140, 150], confidence=.9)], self.full, 2, 2)
        rows = self.tracker.update([self.detection([100, 120, 140, 160], confidence=.45)], self.full, 3, 3)
        self.assertAlmostEqual(rows[0]['confidence'], .45)
        tracker = RevisitTracker(self.model, 'test', RevisitConfig(confidence_memory=4, strong_confidence=.5))
        tracker.update([self.detection([100, 100, 140, 140], confidence=.9)], self.full, 1, 1)
        tracker.update([self.detection([100, 110, 140, 150], confidence=.9)], self.full, 2, 2)
        self.assertEqual(tracker.tracks['track-00001'].strong_sightings, 2)
        restored = RevisitTracker.from_dict(json.loads(json.dumps(tracker.to_dict())))
        self.assertEqual(restored.tracks['track-00001'].strong_sightings, 2)
        rows = restored.update([self.detection([100, 120, 140, 160], confidence=.45)], self.full, 3, 3)
        self.assertAlmostEqual(rows[0]['confidence'], .9-(.9-.45)/3)
        with self.assertRaises(ValueError):
            RevisitConfig(confidence_memory=-1)
        with self.assertRaises(ValueError):
            RevisitConfig(strong_confidence=1.5)

    def test_skipped_detector_does_not_count_as_negative_evidence(self):
        self.tracker.update([self.detection([100, 100, 140, 140])], self.full, 1, 1)
        for frame in range(2, 10): self.tracker.update([], self.full, frame, frame, detector_ran=False)
        self.assertEqual(len(self.tracker.tracks), 1)

    def test_partial_box_does_not_shrink_stored_extent_or_create_new_track(self):
        self.tracker.update([self.detection([1900, 100, 1940, 140])], self.full, 1, 1)
        partial = self.detection([1900, 110, 1920, 150], self.left, complete=False)
        result = self.tracker.update([partial], self.left, 2, 2)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]['bbox_source_xyxy'], [1900, 110, 1940, 150])
        self.assertEqual(result[0]['observations'], 1)
        unknown = RevisitTracker(self.model, 'test')
        self.assertFalse(unknown.update([partial], self.left, 2, 2))

    def test_two_same_class_objects_duplicates_and_ambiguous_association(self):
        initial = [self.detection([100, 100, 140, 140]), self.detection([200, 100, 240, 140])]
        self.tracker.update(initial, self.full, 1, 1)
        later = [self.detection([200, 120, 240, 160]), self.detection([100, 120, 140, 160])]
        self.tracker.update(later+[later[0]], self.full, 3, 3)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertEqual(self.tracker.tracks['track-00001'].history[-1][1][0], 100)
        mid = self.detection([150, 130, 190, 170])
        self.tracker.update([mid], self.full, 4, 4)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertIn('ambiguous_detection', [e['event'] for e in self.tracker.events])

    def test_distinct_neighbor_of_matched_track_gets_own_identity_and_survives_crop_exit(self):
        # Like the two nearby aircraft: one exists, another appears nearby.
        lower = [200, 180, 240, 220]
        self.tracker.update([self.detection(lower, label='medium_plane')], self.full, 1, 1)
        upper = self.detection([200, 140, 240, 180], label='medium_plane', confidence=.76)
        lower = self.detection([200, 190, 240, 230], label='medium_plane', confidence=.73)
        result = self.tracker.update([upper, lower], self.full, 2, 2)
        self.assertEqual(len(result), 2)
        self.assertIn('distinct_neighbor', [e['event'] for e in self.tracker.events])
        self.assertEqual({r['track_id'] for r in self.tracker.update([], self.right, 3, 3)},
                         {'track-00001', 'track-00002'})

    def test_overlapping_unmatched_candidate_does_not_create_duplicate(self):
        self.tracker.update([self.detection([100,100,140,140])], self.full, 1, 1)
        self.tracker.update([self.detection([100,110,140,150]),
                             self.detection([115,110,155,150],confidence=.75)], self.full, 2, 2)
        self.assertEqual(len(self.tracker.tracks), 1)

    def test_frozen_and_skipped_frames_and_state_roundtrip(self):
        box = [100, 100, 140, 140]
        self.tracker.update([self.detection(box)], self.full, 1, 1)
        self.tracker.update([self.detection(box)], self.full, 1, 2)
        self.assertEqual(len(self.tracker.tracks['track-00001'].history), 1)
        restored = RevisitTracker.from_dict(json.loads(json.dumps(self.tracker.to_dict())))
        rows = restored.update([], self.right, 5, 7)
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [100, 140, 140, 180])
        with self.assertRaises(ValueError): restored.update([], self.right, 5, 7)
        with self.assertRaises(ValueError): restored.update([], self.right, 4, 8)

    def test_source_exit_retirement_and_invalid_input_is_atomic(self):
        self.tracker.update([self.detection([100, 2100, 140, 2140])], self.full, 1, 1)
        before = self.tracker.to_dict()
        with self.assertRaises(ValueError): self.tracker.update([Detection('tank', (-1, 1, 10, 20), .9)], self.full, 2, 2)
        self.assertEqual(before, self.tracker.to_dict())
        self.assertFalse(self.tracker.update([], self.right, 10, 10))

    def test_optional_adaptation_uses_past_prediction_check(self):
        a = np.outer([1920., -8000., 1.], [0., -7e-7, -.006])
        model = MotionModel(a, (3840, 2160), 0, 1)
        tracker = RevisitTracker(model, 'test', RevisitConfig(adapt_edges=True))
        near = MotionModel(a*1.03, (3840, 2160), 0, 1)
        initial = np.array([1100., 100., 1170., 200.])
        for frame in (1, 3, 5, 7):
            box = near.box(initial, 1, frame)
            tracker.update([self.detection(box)], self.full, frame, frame)
        self.assertTrue(tracker.tracks['track-00001'].adaptation_checked)
        # Regardless of whether the optional model qualifies, observed boxes
        # remain exact anchors and forecasting stays finite.
        np.testing.assert_allclose(tracker.predictions(7)[0]['bbox_source_xyxy'], box)
        self.assertTrue(np.all(np.isfinite(tracker.predictions(9)[0]['bbox_source_xyxy'])))


class CameraAndWorkflowTests(unittest.TestCase):
    def test_overview_between_sides_cycle_rejection_and_level_two_return(self):
        camera = Camera()
        planner = LevelOneSweep(overview_between_sides=True)
        expected = [(1, 960, 540), (0, 1920, 1080), (1, 2880, 540), (0, 1920, 1080)]
        for frame in range(12):
            command = planner.next_view(request(frame, camera))
            # If a camera command is not applied, repeat it on the same actual view.
            self.assertEqual(command, planner.next_view(request(frame, camera)))
            camera.apply(**command)
            self.assertEqual((camera.resolution_level, camera.center_x, camera.center_y), expected[frame % 4])
        levels = []
        for frame in range(12, 40):
            focus = [3400, 1200, 3450, 1250] if frame in (14, 15, 16) else None
            command = planner.next_view(request(frame, camera), focus_box=focus)
            if command: camera.apply(**command)
            levels.append(camera.resolution_level)
        self.assertIn(2, levels)
        self.assertEqual(sorted(levels[-8:]), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_sweep_and_level_two_detours_obey_actual_camera_constraints(self):
        camera, planner = Camera(), LevelOneSweep()
        centres = []
        for frame in range(40):
            focus = [3400, 1200, 3450, 1250] if frame in (6, 7, 8) else None
            command = planner.next_view(request(frame, camera), focus_box=focus)
            if command: camera.apply(**command)
            centres.append((camera.resolution_level, camera.center_x, camera.center_y))
        self.assertEqual([c[1] for c in centres[:5]], [960, 1920, 2880, 1920, 960])
        self.assertTrue(any(c[0] == 2 for c in centres))
        self.assertEqual(centres[-1][0], 1)
        camera = Camera(1, 960, 540)
        with self.assertRaises(CameraRejection): camera.apply(1, 2880, 540)

    def test_real_image_bootstrap_response_validation_retry_and_resume(self):
        base = ROOT/'src/helsinki/images'
        camera = Camera(); workflow = DroneTrackingWorkflow(observe_motion=False)
        for frame in (0, 1):
            image = cv2.resize(cv2.imread(str(base/f'frame_{frame:06d}.png'), 0), (960, 540))
            req = request(frame, camera)
            result = workflow.process(req, [Detection('tank', (100, 100+frame*12, 110, 110+frame*12), .9)], image=image)
            parsed = DroneFlybyPredictResponseDto.model_validate(result)
            validate_response(parsed)
            self.assertEqual(workflow.process(req, [], image=image), result)
            if result['requested_view']: camera.apply(**result['requested_view'])
        self.assertIsNotNone(workflow.tracker)
        restored = DroneTrackingWorkflow.from_dict(json.loads(json.dumps(workflow.to_dict())))
        req = request(3, camera)
        self.assertEqual(workflow.process(req, [], detector_ran=False), restored.process(req, [], detector_ran=False))
        with self.assertRaises(ValueError): restored.process(request(4, camera, 'other'), [])
        workflow.camera.overview_between_sides = True
        restored = DroneTrackingWorkflow.from_dict(json.loads(json.dumps(workflow.to_dict())))
        self.assertTrue(restored.camera.overview_between_sides)

    def test_observed_clock_handles_freeze_and_doubled_motion_without_labels(self):
        # Physical image displacement is generated by a known constant image
        # translation. This tests clock mechanics, not reference accuracy.
        rng = np.random.default_rng(123)
        first = rng.integers(0, 256, (540, 960), dtype=np.uint8)
        first = cv2.GaussianBlur(first, (5, 5), 0)
        view = ViewGeometry((3840, 2160), (0, 0, 3840, 2160), (960, 540))
        a = np.zeros((3, 3)); a[1, 2] = 20
        model = MotionModel(a, view.source_size, 0, 1)
        for steps in (0, 1, 2):
            second = cv2.warpAffine(first, np.float32([[1, 0, 0], [0, 1, 5*steps]]), (960, 540))
            value, info = observed_step(first, view, second, view, model, 1, 1)
            self.assertEqual(value, steps); self.assertEqual(info['mode'], 'observed_background')


if __name__ == '__main__':
    unittest.main()

"""Placement rules: extent policies, partial completion, clipping, transients, decay."""
import unittest

import numpy as np

from . import Detection, MotionModel, RevisitConfig, RevisitTracker, ViewGeometry
from .placement import SizePrior, clip_box, complete_partial, normalize_extent
from .revisit import frame_rows, overlap


def prior():
    return SizePrior({'tank': {'w': {'intercept': 40., 'slope': 0., 'minimum': 36., 'maximum': 44.},
                               'h': {'intercept': 60., 'slope': 0., 'minimum': 54., 'maximum': 66.}}})


class ExtentTests(unittest.TestCase):
    def test_policies_keep_centre_and_follow_prior(self):
        box = [100, 100, 120, 130]  # 20 x 30 silhouette-like detection
        np.testing.assert_allclose(normalize_extent('tank', box, prior(), 'detector'), box)
        p = normalize_extent('tank', box, prior(), 'prior')
        np.testing.assert_allclose(p, [90, 85, 130, 145])
        b = normalize_extent('tank', box, prior(), 'blend')
        size = b[2:]-b[:2]
        np.testing.assert_allclose(size, [np.sqrt(20*40), np.sqrt(30*60)])
        np.testing.assert_allclose((b[:2]+b[2:])/2, [110, 115])
        # A factor-two disagreement per axis is exactly the IoU 0.50 boundary for the blend.
        self.assertGreaterEqual(overlap(b, p), .5-1e-9)
        self.assertGreaterEqual(overlap(b, np.array(box, float)), .5-1e-9)

    def test_unknown_class_or_missing_prior_is_unchanged(self):
        box = [10, 10, 30, 40]
        np.testing.assert_allclose(normalize_extent('jammer', box, prior(), 'prior'), box)
        np.testing.assert_allclose(normalize_extent('tank', box, None, 'prior'), box)
        with self.assertRaises(ValueError):
            normalize_extent('tank', box, prior(), 'nonsense')

    def test_prior_size_is_clamped(self):
        p = SizePrior({'tank': {'w': {'intercept': 10., 'slope': 1., 'minimum': 36., 'maximum': 44.},
                                'h': {'intercept': 60., 'slope': 0., 'minimum': 54., 'maximum': 66.}}})
        np.testing.assert_allclose(p.size('tank', 0), [.9*36, 60])
        np.testing.assert_allclose(p.size('tank', 2000), [1.1*44, 60])


class PartialTests(unittest.TestCase):
    def test_crop_edge_sides_extend_but_frame_edge_sides_stay(self):
        source = (3840, 2160)
        # Right side cut by a level-1 crop ending at x=1920: extend right to the prior width.
        box = complete_partial('tank', [1900, 500, 1920, 560], (0, 0, 1920, 1080), source, prior())
        np.testing.assert_allclose(box, [1900, 500, 1940, 560])
        # Top side at the source edge: the organizer box is clipped there too, so nothing changes.
        box = complete_partial('tank', [500, 0, 540, 30], (0, 0, 1920, 1080), source, prior())
        np.testing.assert_allclose(box, [500, 0, 540, 30])
        # Both sides cut by a narrow crop: grow symmetrically.
        box = complete_partial('tank', [1000, 1000, 1030, 1080], (1000, 540, 1030, 1620), source, prior())
        np.testing.assert_allclose(box, [995, 1000, 1035, 1080])
        # Left side cut only: grow left.
        box = complete_partial('tank', [1000, 1000, 1030, 1080], (1000, 540, 2920, 1620), source, prior())
        np.testing.assert_allclose(box, [990, 1000, 1030, 1080])

    def test_clip_uses_last_pixel_index(self):
        np.testing.assert_allclose(clip_box([3800, 2100, 3900, 2300], (3840, 2160)), [3800, 2100, 3839, 2159])
        np.testing.assert_allclose(clip_box([3800, 2100, 3900, 2300], (3840, 2160), last_index=False), [3800, 2100, 3840, 2160])
        self.assertIsNone(clip_box([3839, 100, 3900, 200], (3840, 2160)))


class TrackerPlacementTests(unittest.TestCase):
    def setUp(self):
        a = np.zeros((3, 3)); a[1, 2] = 10
        self.model = MotionModel(a, (3840, 2160), 0., 1.)
        self.full = ViewGeometry((3840, 2160), (0, 0, 3840, 2160), (960, 540))
        self.left = ViewGeometry((3840, 2160), (0, 0, 1920, 1080), (960, 540))

    def detection(self, source, view, label='tank', confidence=.9):
        return Detection(label, tuple(view.box_from_source(source)), confidence)

    def test_defaults_do_not_emit_partials_and_clip_to_size(self):
        tracker = RevisitTracker(self.model, 's')
        rows = tracker.update([self.detection([1900, 500, 1920, 560], self.left)], self.left, 1, 1)
        self.assertEqual(rows, [])
        tracker.update([self.detection([3700, 2100, 3800, 2150], self.full)], self.full, 2, 2)
        rows = tracker.predictions(4)
        self.assertEqual(rows[0]['bbox_source_xyxy'][3], 2160.)

    def test_transient_for_crop_cut_box_creates_no_track_and_lasts_one_frame(self):
        config = RevisitConfig(emit_partials=True, size_prior='bundled')
        tracker = RevisitTracker(self.model, 's', config)
        rows = tracker.update([self.detection([1900, 500, 1920, 560], self.left)], self.left, 1, 1)
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['observations'], 0)
        self.assertGreater(rows[0]['bbox_source_xyxy'][2], 1920)
        self.assertAlmostEqual(rows[0]['confidence'], .9*.8)
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(tracker.predictions(2), [])  # not carried into the next tick

    def test_low_confidence_complete_box_is_reported_once_without_a_track(self):
        config = RevisitConfig(emit_partials=True)
        tracker = RevisitTracker(self.model, 's', config)
        rows = tracker.update([self.detection([100, 100, 140, 160], self.full, confidence=.5)], self.full, 1, 1)
        self.assertEqual(len(rows), 1); self.assertEqual(len(tracker.tracks), 0)
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [100, 100, 140, 160])

    def test_extent_policy_applies_to_birth_and_refresh(self):
        config = RevisitConfig(extent_policy='prior')
        tracker = RevisitTracker(self.model, 's', config)
        tracker.prior = prior()
        rows = tracker.update([self.detection([100, 100, 120, 130], self.full)], self.full, 1, 1)
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [90, 85, 130, 145])
        rows = tracker.update([self.detection([100, 110, 120, 140], self.full)], self.full, 2, 2)
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [90, 95, 130, 155])
        self.assertEqual(rows[0]['observations'], 2)

    def test_clip_last_index_and_forecast_decay(self):
        config = RevisitConfig(clip_last_index=True, forecast_decay=.1)
        tracker = RevisitTracker(self.model, 's', config)
        tracker.update([self.detection([3700, 2100, 3800, 2150], self.full)], self.full, 1, 1)
        rows = tracker.predictions(3)
        self.assertEqual(rows[0]['bbox_source_xyxy'][3], 2159.)
        self.assertAlmostEqual(rows[0]['confidence'], .9*.81)

    def test_frame_rows_for_warmup(self):
        config = RevisitConfig(emit_partials=True, extent_policy='prior')
        detections = [self.detection([100, 100, 120, 130], self.left), self.detection([1900, 500, 1920, 560], self.left),
                      self.detection([300, 300, 320, 330], self.left, confidence=.3)]
        rows = frame_rows(detections, self.left, config, prior())
        self.assertEqual([r['complete'] for r in rows], [True, False])
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [90, 85, 130, 145])
        self.assertGreater(rows[1]['bbox_source_xyxy'][2], 1920)

    def test_place_partial_normalizes_only_free_axes(self):
        from .placement import place_partial
        # Top cut by the source edge, both horizontal sides visible: the width follows the prior,
        # the visible height stays because the organizer box is clipped there too.
        box = place_partial('tank', [500, 0, 520, 30], (0, 0, 1920, 1080), (3840, 2160), prior(), 'prior')
        np.testing.assert_allclose(box, [490, 0, 530, 30])

    def test_entry_track_from_top_edge_partial(self):
        from .placement import entry_box
        full = entry_box('tank', [500, 0, 540, 30], (0, 0, 3840, 2160), (3840, 2160), prior())
        np.testing.assert_allclose(full, [500, -30, 540, 30])
        self.assertIsNone(entry_box('tank', [500, 0, 540, 30], (0, 540, 1920, 1620), (3840, 2160), prior()))
        self.assertIsNone(entry_box('tank', [1900, 0, 1920, 30], (0, 0, 1920, 1080), (3840, 2160), prior()))
        config = RevisitConfig(entry_tracks=True, emit_partials=True)
        tracker = RevisitTracker(self.model, 's', config); tracker.prior = prior()
        rows = tracker.update([self.detection([500, 0, 540, 30], self.full)], self.full, 1, 1)
        self.assertEqual(len(rows), 1); self.assertTrue(rows[0]['provisional'])
        np.testing.assert_allclose(rows[0]['bbox_source_xyxy'], [500, 0, 540, 30])
        self.assertAlmostEqual(rows[0]['confidence'], .9*.8)
        # The forecast keeps the visible extent growing downwards at the scene motion.
        np.testing.assert_allclose(tracker.predictions(2)[0]['bbox_source_xyxy'], [500, 0, 540, 40])
        np.testing.assert_allclose(tracker.predictions(5)[0]['bbox_source_xyxy'], [500, 10, 540, 70])
        # A complete observation turns it into an ordinary track at full confidence.
        rows = tracker.update([self.detection([500, 12, 540, 72], self.full)], self.full, 5, 5)
        self.assertFalse(rows[0]['provisional']); self.assertAlmostEqual(rows[0]['confidence'], .9)
        self.assertEqual(rows[0]['observations'], 2)

    def test_entry_track_retires_after_one_clear_miss(self):
        config = RevisitConfig(entry_tracks=True)
        tracker = RevisitTracker(self.model, 's', config); tracker.prior = prior()
        tracker.update([self.detection([500, 0, 540, 30], self.full)], self.full, 1, 1)
        tracker.update([], self.full, 2, 2)  # still cut at the top: not a full opportunity, kept
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([], self.full, 6, 6)  # now fully inside the view and unseen: retired
        self.assertEqual(len(tracker.tracks), 0)

    def test_state_round_trip_keeps_placement_config(self):
        config = RevisitConfig(extent_policy='blend', emit_partials=True, clip_last_index=True)
        tracker = RevisitTracker(self.model, 's', config)
        tracker.update([self.detection([100, 100, 140, 160], self.full)], self.full, 1, 1)
        restored = RevisitTracker.from_dict(tracker.to_dict())
        self.assertEqual(restored.config, config)
        self.assertIsNotNone(restored.prior)


class CameraModeTests(unittest.TestCase):
    def test_l2_top_sweep_is_legal_and_covers_the_row(self):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for candidate in (root/'artifacts/drone-source-2026-09-17', Path(__file__).resolve().parents[1]):
            if (candidate/'local_evaluator.py').exists():
                sys.path.insert(0, str(candidate)); break
        from local_evaluator import Camera
        from .workflow import LevelOneSweep
        camera = Camera(); sweep = LevelOneSweep(mode='l2_top')
        levels, centres = [], []
        for frame in range(40):
            request = {'original_width': 3840, 'original_height': 2160,
                       'view': {'resolution_level': camera.resolution_level, 'center_x': camera.center_x, 'center_y': camera.center_y},
                       'camera_constraints': camera.constraints()}
            requested = sweep.next_view(request, overview=frame < 2)
            self.assertIsNotNone(requested)
            camera.apply(**requested)  # raises CameraRejection on any illegal move
            levels.append(camera.resolution_level); centres.append((camera.center_x, camera.center_y))
        self.assertEqual(levels[:2], [0, 0]); self.assertEqual(levels[2], 1)
        self.assertTrue(all(level == 2 for level in levels[3:]))
        xs = sorted({x for (x, y), level in zip(centres, levels) if level == 2})
        self.assertEqual(xs[0], 480); self.assertEqual(xs[-1], 3360); self.assertGreaterEqual(len(xs), 7)
        self.assertTrue(all(y == 270 for (x, y), level in zip(centres, levels) if level == 2))
        hops = [abs(a[0]-b[0]) for a, b in zip(centres[3:], centres[4:])]
        self.assertTrue(all(h <= 551 for h in hops))
        self.assertEqual(max(hops), 480)

    def test_stale_track_revisit_picks_oldest_reachable_track(self):
        from .workflow import DroneTrackingWorkflow
        a = np.zeros((3, 3)); a[1, 2] = 10
        model = MotionModel(a, (3840, 2160), 0., 1.)
        workflow = DroneTrackingWorkflow(camera_mode='l2_top', revisit_every=1, revisit_min_age=3)
        workflow.tracker = RevisitTracker(model, 's')
        full = ViewGeometry((3840, 2160), (0, 0, 3840, 2160), (960, 540))
        workflow.tracker.update([Detection('tank', tuple(full.box_from_source([1000, 300, 1040, 360])), .9)], full, 1, 1)
        workflow.tracker.update([Detection('jammer', tuple(full.box_from_source([3000, 300, 3030, 340])), .9)], full, 5, 5)
        request = {'original_width': 3840, 'original_height': 2160,
                   'view': {'resolution_level': 2, 'center_x': 960, 'center_y': 270},
                   'camera_constraints': {'maximum_center_delta': 551., 'allowed_resolution_levels': [1, 2],
                                          'center_bounds': [{'resolution_level': 2, 'minimum_center_x': 480, 'maximum_center_x': 3360,
                                                             'minimum_center_y': 270, 'maximum_center_y': 1890}]}}
        # The tank is 9 ticks old and within one L2 move; the jammer is younger and far away.
        box = workflow.stale_track(request, 10)
        self.assertIsNotNone(box); self.assertAlmostEqual(box[0], 1000)
        workflow.revisit_min_age = 20
        self.assertIsNone(workflow.stale_track(request, 10))  # nothing old enough yet
        workflow.revisit_min_age = 3
        requested = workflow.camera.next_view(request, focus_box=box)
        self.assertEqual(requested['resolution_level'], 2)
        self.assertLessEqual(abs(requested['center_x']-960)+abs(requested['center_y']-270), 551)


if __name__ == '__main__':
    unittest.main()

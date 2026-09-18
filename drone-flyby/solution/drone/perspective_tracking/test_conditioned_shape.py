"""Structural generalization and non-regression guards, not field accuracy."""
from dataclasses import replace
import json
import unittest

import numpy as np

from .conditioned_shape import (ConditionedShapeModel, axis_from_crop, features,
                                fit_residual_factors)
from .motion import MotionModel


class ConditionedShapeTests(unittest.TestCase):
    def setUp(self):
        self.model = MotionModel(np.outer([1920., -8000., 1.], [0., -7e-7, -.006]),
                                 (3840, 2160), 0., 1.)
        self.box = np.array([1100., 100., 1170., 200.])
        self.samples = []
        for identity in range(3):
            for offset in (0., 120., 240.):
                box = self.box+np.array([identity*100., offset, identity*100., offset])
                even, odd = features(self.model, box, 1., .6, .8)
                self.samples.append({'label': 'tower', 'instance_id': f'train-{identity}',
                    'sequence_id': 'training-flight', 'even': even.tolist(), 'odd': odd.tolist(),
                    'factors': [1.01, 1.025, 1.02, .997]})
        self.bank = ConditionedShapeModel.fit(self.samples, reference_step_m=1., ridge=.1)

    def qualified_records(self):
        return [{'label': 'tower', 'instance_id': f'test-{i}', 'sequence_id': 'new-flight',
                 'label_source': 'official', 'frames': 10, 'candidate_passes': 10,
                 'candidate_mean_iou': .85, 'baseline_mean_iou': .8, 'candidate_min_iou': .7}
                for i in range(2)]

    def predict(self, bank, **kwargs):
        return bank.predict('tower', self.model, self.box, 1., distance_m=12.,
                            orientation=.6, axis_score=.8, **kwargs)

    def test_reflecting_camera_object_and_axis_reflects_prediction(self):
        width = self.model.source_size[0]
        reflection = np.array([[-1., 0., width], [0., 1., 0.], [0., 0., 1.]])
        mirrored_model = MotionModel(reflection@self.model.matrix@reflection,
                                     self.model.source_size, 0., 1.)
        mirrored_box = self.box.copy(); mirrored_box[[0, 2]] = width-self.box[[2, 0]]
        original = self.predict(self.bank, diagnostic=True)['box']
        reflected = self.bank.predict('tower', mirrored_model, mirrored_box, 1., distance_m=12.,
                                      orientation=np.pi-.6, axis_score=.8, diagnostic=True)['box']
        expected = original.copy(); expected[[0, 2]] = width-original[[2, 0]]
        np.testing.assert_allclose(reflected, expected, atol=1e-9)

    def test_zero_residual_is_exactly_shared_geometry(self):
        zero = replace(self.bank, even_weights=np.zeros_like(self.bank.even_weights),
                       odd_weights=np.zeros_like(self.bank.odd_weights))
        result = self.predict(zero, diagnostic=True)
        np.testing.assert_array_equal(result['box'], self.model.box(self.box, 1., 13.))
        observations = [(f, self.model.box(self.box, 0., f)) for f in range(6)]
        np.testing.assert_allclose(fit_residual_factors(self.model, observations), np.ones(4), atol=1e-10)

    def test_duplicate_frames_do_not_increase_an_instances_training_weight(self):
        extra = self.samples+[s for s in self.samples if s['instance_id'] == 'train-0']*9
        repeated = ConditionedShapeModel.fit(extra, reference_step_m=1., ridge=.1)
        np.testing.assert_allclose(repeated.even_weights, self.bank.even_weights, atol=1e-12)
        np.testing.assert_allclose(repeated.odd_weights, self.bank.odd_weights, atol=1e-12)

    def test_one_instance_cannot_fit_a_class_position_orientation_surface(self):
        one = ConditionedShapeModel.fit(self.samples[:3], reference_step_m=1.)
        np.testing.assert_array_equal(one.even_weights[7:], 0.)
        np.testing.assert_array_equal(one.odd_weights[2:], 0.)
        self.assertEqual(one.instance_counts['tower'], 1)
        qualified, reasons = one.qualify(self.qualified_records())
        self.assertFalse(qualified.validated_classes)
        self.assertIn('fewer_than_two_training_instances', reasons['tower'])

    def test_unvalidated_class_keeps_existing_prediction(self):
        result = self.predict(self.bank)
        self.assertEqual(result['mode'], 'shared_fallback')
        np.testing.assert_array_equal(result['box'], self.model.box(self.box, 1., 13.))

    def test_qualification_rejects_pseudo_labels_overlap_and_regressions(self):
        for field, value, reason in [
            ('label_source', 'tracker_assisted_pseudo_labels', 'evaluation_uses_tracker_pseudo_labels'),
            ('instance_id', 'train-0', 'training_instance_or_flight_overlap'),
            ('sequence_id', 'training-flight', 'training_instance_or_flight_overlap'),
            ('candidate_passes', 9, 'held_out_regression_or_insufficient_margin'),
            ('candidate_min_iou', .55, 'held_out_regression_or_insufficient_margin'),
            ('candidate_mean_iou', .75, 'held_out_regression_or_insufficient_margin')]:
            with self.subTest(field=field):
                records = self.qualified_records(); records[0][field] = value
                qualified, reasons = self.bank.qualify(records)
                self.assertFalse(qualified.validated_classes)
                self.assertIn(reason, reasons['tower'])

    def test_qualified_model_still_rejects_uncertainty_and_outside_coverage(self):
        qualified, reasons = self.bank.qualify(self.qualified_records())
        self.assertEqual(qualified.validated_classes, ('tower',)); self.assertFalse(reasons)
        self.assertEqual(self.predict(qualified)['mode'], 'conditioned_profile')
        self.assertEqual(self.predict(qualified, class_confidence=.4)['reason'], 'uncertain_class')
        no_axis = qualified.predict('tower', self.model, self.box, 1., distance_m=12.)
        self.assertEqual(no_axis['reason'], 'uncertain_image_axis')
        outside = qualified.predict('tower', self.model, self.box+[0, 1400, 0, 1400], 1.,
                                    distance_m=1., orientation=.6, axis_score=.8)
        self.assertEqual(outside['reason'], 'outside_observed_shape_context')

    def test_factor_limits_and_json_roundtrip(self):
        extreme = replace(self.bank, even_weights=self.bank.even_weights*1e6,
                          odd_weights=self.bank.odd_weights*1e6)
        factors = self.predict(extreme, diagnostic=True).get('factors')
        if factors is None:  # Crossing projected edges is an explicit fallback.
            self.assertEqual(self.predict(extreme, diagnostic=True)['reason'], 'invalid_profile_projection')
        else:
            self.assertTrue(np.all(np.array(factors) <= 1.15+1e-12))
            self.assertTrue(np.all(np.array(factors) >= 1/1.15-1e-12))
        restored = ConditionedShapeModel.from_dict(json.loads(json.dumps(self.bank.to_dict())))
        np.testing.assert_array_equal(self.predict(restored, diagnostic=True)['box'],
                                      self.predict(self.bank, diagnostic=True)['box'])

    def test_image_axis_is_undirected_and_reflects_consistently(self):
        import cv2
        image = np.zeros((100, 100), np.uint8)
        vertices = cv2.boxPoints(((50, 50), (65, 17), 25)).astype(np.int32)
        cv2.fillConvexPoly(image, vertices, 200)
        angle, score = axis_from_crop(image)
        reflected, mirrored_score = axis_from_crop(image[:, ::-1].copy())
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(score, mirrored_score, places=12)
        self.assertAlmostEqual(np.cos(2*angle), np.cos(2*reflected), places=12)
        self.assertAlmostEqual(np.sin(2*angle), -np.sin(2*reflected), places=12)
        self.assertIsNone(axis_from_crop(np.zeros((30, 30), np.uint8))[0])


if __name__ == '__main__':
    unittest.main()

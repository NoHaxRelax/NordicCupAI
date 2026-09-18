"""Physical projection checks for the experimental learned edge profile."""
import json
import unittest

import numpy as np

from .edge_profile import EdgeMotionProfile
from .motion import MotionModel, ProjectionError


class EdgeProfileTests(unittest.TestCase):
    def setUp(self):
        # Camera moves one metre/tick towards a plane 1000 metres away.
        # This simple geometry has independently calculable image projections.
        self.model = MotionModel(np.outer([1920., 1080., 1.], [0., 0., -.001]),
                                 (3840, 2160), 0, 1)

    @staticmethod
    def projected_box(depth, tick):
        xyz = np.array([[-20., -10., depth], [20., 10., depth]])
        xyz[:, 2] -= tick
        intrinsic = np.array([[1200., 0., 1920.], [0., 1200., 1080.], [0., 0., 1.]])
        homogeneous = xyz @ intrinsic.T
        return (homogeneous[:, :2]/homogeneous[:, 2, None]).reshape(4)

    def test_known_depth_predicts_independent_3d_projection_from_one_box(self):
        # At tick 1 the reference plane is 999 m away, object plane 499 m.
        profile = EdgeMotionProfile('nearer-plane', (999/499,)*4, 1.)
        initial = self.projected_box(500, 1)
        for distance in (0., 12.5, 50., 125.):
            predicted = profile.predict(self.model, initial, 1, distance_m=distance)
            np.testing.assert_allclose(predicted, self.projected_box(500, 1+distance), atol=1e-10)

    def test_fitting_recovers_reference_depth_and_forecasts_unused_views(self):
        training = {'training-object': [(f, self.projected_box(1000, f)) for f in range(1, 9)]}
        profile = EdgeMotionProfile.fit(self.model, training, label='plane', reference_step_m=1.)
        np.testing.assert_allclose(profile.factors, np.ones(4), atol=1e-10)
        predicted = profile.predict(self.model, self.projected_box(1000, 9), 9, distance_m=71.)
        np.testing.assert_allclose(predicted, self.projected_box(1000, 80), atol=1e-10)
        self.assertEqual(profile.training['instance_views'], {'training-object': 8})

    def test_distance_clock_and_serialization(self):
        profile = EdgeMotionProfile('plane', (1.,)*4, 1.)
        restored = EdgeMotionProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
        initial = self.projected_box(1000, 1)
        slow = profile.predict(self.model, initial, 1, distance_m=5.*10.)
        fast = restored.predict(self.model, initial, 1, distance_m=10.*5.)
        np.testing.assert_array_equal(slow, fast)
        np.testing.assert_allclose(slow, self.projected_box(1000, 51), atol=1e-10)

    def test_profile_horizon_is_checked_before_reference_plane_horizon(self):
        profile = EdgeMotionProfile('nearer-plane', (999/499,)*4, 1.)
        with self.assertRaises(ProjectionError):
            profile.predict(self.model, self.projected_box(500, 1), 1, distance_m=499.)

    def test_unsupported_geometry_and_invalid_inputs_fail_explicitly(self):
        profile = EdgeMotionProfile('plane', (1.,)*4, 1.)
        for distance in (-1., float('nan')):
            with self.assertRaises(ValueError):
                profile.predict(self.model, self.projected_box(1000, 1), 1, distance_m=distance)
        translation = np.zeros((3, 3)); translation[1, 2] = 50.
        with self.assertRaises(ValueError):
            profile.predict(MotionModel(translation, (3840, 2160), 0, 1),
                            [100, 100, 140, 140], 1, distance_m=1.)
        for factors, step in (((1., 1., 1., 0.), 1.), ((1.,)*4, 0.)):
            with self.assertRaises(ValueError):
                EdgeMotionProfile('invalid', factors, step)

    def test_training_needs_information_not_just_repeated_identical_boxes(self):
        fixed = self.projected_box(1000, 1)
        for track in ([(1, fixed)], [(f, fixed) for f in range(8)]):
            with self.assertRaises(ValueError):
                EdgeMotionProfile.fit(self.model, {'stationary': track},
                                      label='stationary', reference_step_m=1.)


if __name__ == '__main__':
    unittest.main()

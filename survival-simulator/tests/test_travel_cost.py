import math
import unittest

import numpy as np

from src.utils.controllers.travel_cost import PublicBiomeTravel


def layer(labels, confidence=None, bounds=(0., 0., 100., 20.)):
    return dict(bounds=list(bounds), palette=["forest", "river"], labels=labels,
                confidence=confidence if confidence is not None else np.ones(np.shape(labels)).tolist(),
                updated_at=0., fingerprint="fixture")


class PublicBiomeTravelTests(unittest.TestCase):
    def setUp(self):
        self.sampler = PublicBiomeTravel()

    def test_uniform_forest_preserves_actual_polyline_distance(self):
        self.sampler.update(layer([[0, 0]]))
        self.assertAlmostEqual(self.sampler.requested_distance([(0., 10.), (100., 10.)], 1.), 100.)

    def test_half_forest_half_river_accounts_for_slow_section(self):
        self.sampler.update(layer([[0, 1]]))
        requested = self.sampler.requested_distance([(0., 10.), (100., 10.)], 1.)
        self.assertAlmostEqual(requested, 50. + 50. / .3)
        # Ten stored energy can fund the forest trip, but not the mixed trip.
        forest_cost = 100. * (.05 + .1 * 1.2 / 10.)
        mixed_cost = requested * (.05 + .1 * 1.2 / 10.)
        self.assertLess(forest_cost, 10.)
        self.assertGreater(mixed_cost, 10.)

    def test_confidence_blends_energy_cost_not_average_speed(self):
        self.sampler.update(layer([[1]], [[.5]]))
        self.assertAlmostEqual(self.sampler.requested_distance([(0., 10.), (100., 10.)], 1.),
                               100. * (.5 / .3 + .5))

    def test_unknown_labels_and_outside_segments_use_current_biome(self):
        self.sampler.update(layer([[-1, 1]]))
        # Outside50px + unknown50px use swamp(.5); known river uses.3.
        requested = self.sampler.requested_distance([(-25., 10.), (125., 10.)], .5)
        self.assertAlmostEqual(requested, 100. / .5 + 50. / .3)

    def test_route_entirely_outside_never_uses_a_clipped_edge_cell(self):
        self.sampler.update(layer([[1]]))
        self.assertAlmostEqual(self.sampler.requested_distance([(0., 25.), (100., 25.)], 1.), 100.)

    def test_diagonal_boundary_crossing_is_exact_in_both_directions(self):
        self.sampler.update(layer([[0, 1]], bounds=(0., 0., 100., 100.)))
        points = [(0., 0.), (100., 100.)]
        expected = math.sqrt(2.) * (50. + 50. / .3)
        self.assertAlmostEqual(self.sampler.requested_distance(points, 1.), expected)
        self.assertAlmostEqual(self.sampler.requested_distance(points[::-1], 1.), expected)

    def test_detour_and_pickup_endpoint_are_not_replaced_by_center_distance(self):
        self.sampler.update(layer([[0, 0]]))
        # Route ends at the supplied pickup point, not a hypothetical fruit center.
        requested = self.sampler.requested_distance([(0., 10.), (0., 0.), (90., 0.), (90., 10.)], 1.)
        self.assertAlmostEqual(requested, 110.)

    def test_empty_and_zero_length_paths_cost_nothing(self):
        self.sampler.update(layer([[1]]))
        for points in ([], [(10., 10.)], [(10., 10.), (10., 10.)]):
            self.assertEqual(self.sampler.requested_distance(points, .5), 0.)

    def test_nonfinite_routes_are_unaffordable(self):
        for coordinate in (math.inf, -math.inf, math.nan):
            self.assertEqual(self.sampler.requested_distance([(0., 0.), (coordinate, 10.)], 1.), math.inf)

    def test_missing_or_malformed_layer_falls_back(self):
        for estimate in (None, {}, layer([[0]], bounds=(0., 0., 0., 20.)),
                         layer([[0]], confidence=[[1., 1.]])):
            self.sampler.update(estimate)
            self.assertEqual(self.sampler.requested_distance([(0., 0.), (100., 0.)], .5), 200.)

    def test_unchanged_layer_reuses_arrays_and_new_fit_replaces_them(self):
        estimate = layer([[0, 1]])
        self.sampler.update(estimate)
        prepared = self.sampler.inverse
        self.sampler.update(estimate)
        self.assertIs(self.sampler.inverse, prepared)
        estimate["labels"] = [[0, 0]]
        estimate["updated_at"] = 5.
        self.sampler.update(estimate)
        self.assertIsNot(self.sampler.inverse, prepared)
        self.assertEqual(self.sampler.requested_distance([(0., 10.), (100., 10.)], 1.), 100.)

    def test_invalid_fallback_factor_is_rejected(self):
        for factor in (0., -1., math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.sampler.requested_distance([(0., 0.), (1., 1.)], factor)


if __name__ == "__main__":
    unittest.main()

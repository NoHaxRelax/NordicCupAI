"""Independent geometry and recurrence checks for the spot analyzer."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().with_name('predator_stuck_cpp')))
from spot_features import (cluster_spots, endpoint_clearances, expanded_rectangles,
                           geometry_features, period_features)
from spot_report import rank_rules


BOUNDARIES = [[0, 0, 1600, 30], [0, 1170, 1600, 30],
              [0, 0, 30, 1200], [1570, 0, 30, 1200]]


class GeometryTests(unittest.TestCase):
    def test_deep_overlap_blocks_every_heading(self):
        geo = geometry_features(150, 150, 0, 11, 0, BOUNDARIES+[[100, 100, 100, 100]])
        self.assertEqual(geo['free_step_directions_36'], 0)
        self.assertEqual(geo['sampled_free_direction_fraction'], 0)
        self.assertEqual(geo['step_circle_inside_one_obstacle'], 1)

    def test_shallow_overlap_has_escape_directions(self):
        geo = geometry_features(95, 150, 0, 11, 0, BOUNDARIES+[[100, 100, 100, 100]])
        self.assertEqual(geo['overlap_count'], 1)
        self.assertGreater(geo['free_step_directions_36'], 0)
        self.assertEqual(geo['step_circle_inside_one_obstacle'], 0)

    def test_single_wall_is_not_a_corner(self):
        geo = geometry_features(500, 41, 0, 11, 0, BOUNDARIES)
        self.assertEqual(geo['perpendicular_faces_15'], 0)
        self.assertEqual(geo['boundary_corner_15'], 0)
        self.assertGreater(geo['free_step_directions_36'], 0)

    def test_world_corner_still_has_free_directions(self):
        geo = geometry_features(41, 41, 0, 11, 0, BOUNDARIES)
        self.assertEqual(geo['perpendicular_faces_2'], 1)
        self.assertEqual(geo['boundary_corner_15'], 1)
        self.assertGreater(geo['free_step_directions_36'], 0)

    def test_touching_collision_boundary_is_legal(self):
        rects = expanded_rectangles([[100, 100, 100, 100]], 10)
        clearance = endpoint_clearances(79, 150, 11, np.array([0., math.pi]), rects)
        self.assertEqual(clearance[0], 0)
        self.assertGreater(clearance[1], 0)


class RecurrenceTests(unittest.TestCase):
    def test_position_recurrence_does_not_imply_heading_recurrence(self):
        a = np.zeros((128, 34))
        a[:, 1] = np.arange(128) % 2
        a[:, 3] = np.arange(128)*math.pi/2
        result = period_features(a)
        self.assertEqual(result['position_period'], 2)
        self.assertEqual(result['pose_period'], 4)

    def test_small_drift_is_not_a_closed_cycle(self):
        a = np.zeros((128, 34))
        a[:, 1] = np.arange(128)*.01
        result = period_features(a)
        self.assertEqual(result['position_period'], 0)
        self.assertEqual(result['pose_period'], 0)

    def test_spots_are_explicitly_connected_components(self):
        rows = [dict(center_x=x, center_y=0, seed=1, predator_id=i,
                     classification='moving_confinement', trace=str(i), shard='test')
                for i, x in enumerate((0, 4, 8, 20))]
        spots = cluster_spots(rows, 5)
        self.assertEqual([s['predators'] for s in spots], [3, 1])
        self.assertEqual(spots[0]['center_span_x'], 8)


class RuleValidationTests(unittest.TestCase):
    def test_held_out_maps_do_not_choose_the_rule(self):
        cases = np.array([True, False, True, False])
        controls = ~cases
        d = {'seed': np.array([1, 1, 0, 0])}
        hyp = dict(training_winner=dict(label='A', geometry_only=True,
                       mask=np.array([True, False, False, True])),
                   held_out_winner=dict(label='B', geometry_only=True,
                       mask=np.array([False, False, True, False])))
        rules = rank_rules(hyp, d, cases, controls, cases.astype(float))
        self.assertEqual(rules[0]['id'], 'training_winner')
        self.assertEqual(rules[0]['held_out']['case_coverage'], 0)
        self.assertEqual(rules[0]['held_out']['control_specificity'], 0)


if __name__ == '__main__':
    unittest.main()

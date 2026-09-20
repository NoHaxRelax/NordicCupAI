"""Entry evidence, direction-independent blockage and exit-witness tests."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().with_name('predator_stuck_cpp')))
from diagnostics import DETAIL_COLUMNS, INDEX as IX
from entrance_features import (find_exit, observe_entry, prove_no_first_step,
                               segment_is_clear, validate_route)


def entering_trace(escaped=False):
    rows = np.zeros((602, len(DETAIL_COLUMNS)))
    rows[:, 0] = np.arange(1, 603)
    rows[:, 1] = 20; rows[:, IX['before_x']] = 20
    rows[:, IX['accepted_candidate']] = -1
    rows[0, 1] = 31; rows[0, IX['before_x']] = 42
    rows[1, IX['before_x']] = 31
    rows[:2, IX['mode']] = 1
    rows[:2, IX['accepted_candidate']] = 0
    return dict(diagnostic_rows=rows.tolist(), anchor=[20, 0], radius=15,
        initial_position=[42, 0], start_seconds=.2, born_seconds=0,
        detected_seconds=60.2, follow_up=dict(escaped_after_detection=escaped,
            first_exit_seconds=80 if escaped else None, observation_end_seconds=600))


class EntryTests(unittest.TestCase):
    def test_observed_entry_requires_a_legal_crossing_and_sixty_seconds(self):
        result, _, index = observe_entry(entering_trace(), np.empty((0, 4)))
        self.assertEqual(index, 0)
        self.assertEqual(result['observed_legal_entry'], 1)
        self.assertEqual(result['entered_no_observed_exit'], 1)
        self.assertAlmostEqual(result['observed_residence_seconds'], 599.9)

    def test_later_escape_is_not_called_no_exit(self):
        result, _, _ = observe_entry(entering_trace(True), np.empty((0, 4)))
        self.assertEqual(result['entered_then_escaped'], 1)
        self.assertEqual(result['entered_no_observed_exit'], 0)

    def test_spawn_inside_is_not_an_entry(self):
        trace = entering_trace()
        for row in trace['diagnostic_rows']:
            row[1] = row[IX['before_x']] = 20
        trace.update(start_seconds=0, initial_position=[20, 0])
        result, _, index = observe_entry(trace, np.empty((0, 4)))
        self.assertIsNone(index)
        self.assertEqual(result['observed_legal_entry'], 0)
        self.assertEqual(result['entry_evidence'], 'confinement_since_birth_without_observed_entry')

    def test_crossing_from_overlap_is_not_legal_entry(self):
        result, _, index = observe_entry(entering_trace(), np.array([[41, -1, 43, 1]]))
        self.assertIsNone(index)
        self.assertEqual(result['entry_evidence'], 'crossing_has_an_overlapping_endpoint')

    def test_segment_clipping_is_distinguished_from_native_endpoint_legality(self):
        rects = np.array([[34, -1, 38, 1]])
        result, _, _ = observe_entry(entering_trace(), rects)
        self.assertEqual(result['observed_legal_entry'], 1)
        self.assertEqual(result['entry_segment_clear'], 0)
        self.assertTrue(segment_is_clear([34, 1], [38, 1], rects))


class EscapeTests(unittest.TestCase):
    def test_combined_rectangles_can_block_all_headings_from_a_legal_pose(self):
        rects = np.array([[-2,-2,-.5,2], [.5,-2,2,2], [-2,-2,2,-.5], [-2,.5,2,2]])
        self.assertTrue(prove_no_first_step([0, 0], 1, rects))
        self.assertFalse(prove_no_first_step([0, 0], .4, rects))

    def test_isolated_legal_tangency_is_not_a_blocking_proof(self):
        rects = np.array([[-2,-2,-.5,2], [.5,-2,2,2], [-2,-2,2,-.5], [-2,.5,2,2]])
        self.assertFalse(prove_no_first_step([0, 0], math.sqrt(.5), rects))

    def test_reverse_move_can_fail_when_the_biome_step_changes(self):
        biomes = np.zeros((100, 100), dtype=np.uint8)
        path = np.array([[31, 20], [20, 20]])
        self.assertTrue(validate_route(path, [31, 20], 5, biomes, np.empty((0, 4))))
        biomes[31, 20] = 4
        self.assertFalse(validate_route(path, [31, 20], 5, biomes, np.empty((0, 4))))

    def test_search_constructs_a_full_exit_using_short_steps(self):
        biomes = np.full((100, 100), 4, dtype=np.uint8)
        path, _ = find_exit([50, 50], [50, 50], 15, biomes, np.empty((0, 4)))
        self.assertIsNotNone(path)
        self.assertTrue(validate_route(path, [50, 50], 15, biomes, np.empty((0, 4))))
        self.assertGreaterEqual(len(path), 6)

    def test_search_budget_failure_does_not_prove_impossibility(self):
        biomes = np.zeros((100, 100), dtype=np.uint8)
        path, expanded = find_exit([50, 50], [50, 50], 15, biomes, np.empty((0, 4)), max_nodes=1)
        self.assertIsNone(path)
        self.assertEqual(expanded, 1)
        self.assertFalse(prove_no_first_step([50, 50], 11, np.empty((0, 4))))


if __name__ == '__main__':
    unittest.main()

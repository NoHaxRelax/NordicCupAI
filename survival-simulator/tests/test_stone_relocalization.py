"""Collision recovery must not turn a misplaced stone copy into evidence."""

import unittest

import numpy as np

from src.utils.controllers.world_estimator import WorldEstimator
from test_expert_policy import agent_state
from test_global_planner import action, planner_config, tree
from test_mapping import BOUNDARIES, seen_edges


STONE = [((453.4953602283688, 41.42712522472003), (548.6204557783245, 41.42712522472003)),
         ((453.4953602283688, 41.42712522472003), (453.4953602283688, 99.712))]


def observed(position, segments, extra=()):
    return agent_state(seen_edges(np.array(position, dtype=float), 0, segments) + list(extra))


class StoneRelocalizationTests(unittest.TestCase):
    def mapping(self, segments=STONE, position=(520, 35)):
        mapping = WorldEstimator(planner_config(estimator={"relocalize_stones": True}).estimator)
        mapping.update([observed(position, BOUNDARIES + list(segments))], 0)
        self.assertTrue(mapping.groups[7].anchored)
        return mapping

    def test_two_nonparallel_segments_recover_forty_unit_collision_before_storage(self):
        mapping = self.mapping()
        mapping.remember_actions([action(distance=20)])
        # Collision redirects the actual step west; odometry predicts east.
        mapping.update([observed((500, 35), STONE)], 1)
        np.testing.assert_allclose(mapping.poses[7].position, (500, 35), atol=1e-8)
        np.testing.assert_allclose(mapping.poses[7].origin, (520, 35), atol=1e-8)
        self.assertEqual(mapping.poses[7].uncertainty, 1)
        self.assertEqual(len(mapping.groups[7].edges), 4)

    def test_one_far_shape_blocks_all_new_layers_without_guessing_opposite_face(self):
        mapping = self.mapping(STONE[:1])
        mapping.remember_actions([action(distance=20)])
        unknown = ((600, 100), (679, 100))
        mapping.update([observed((500, 35), STONE[:1] + [unknown], [tree(25, 0)])], 1)
        group = mapping.groups[7]
        self.assertGreater(mapping.poses[7].uncertainty, mapping.config.max_position_uncertainty)
        self.assertEqual(len(group.edges), 3)
        self.assertEqual(group.trees, [])
        self.assertTrue(all(sample.last_seen == 0 for sample in group.biomes.values()))
        self.assertTrue(all(sample[2] == 0 for sample in group.visited.values()))

    def test_unseen_opposite_faces_preserve_correct_pose_and_extend_map(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                first = ((200., 200.), (200., 263.173)) if vertical else ((200., 200.), (263.173, 200.))
                shift = np.array((60., 0.)) if vertical else np.array((0., 60.))
                opposite = tuple(tuple(np.array(point) + shift) for point in first)
                position = np.array((180., 100.)) if vertical else np.array((100., 180.))
                mapping = self.mapping([first], position=position)
                # The estimate is accurate when the previously unseen face appears.
                position += 100 * shift / np.linalg.norm(shift)
                mapping.poses[7].position = position.copy()
                mapping.update([observed(position, [opposite], [tree(25, 0)])], 1)
                np.testing.assert_allclose(mapping.poses[7].position, position, atol=1e-8)
                self.assertLessEqual(mapping.poses[7].uncertainty, mapping.config.max_position_uncertainty)
                self.assertEqual(len(mapping.groups[7].edges), 4)
                self.assertEqual(len(mapping.groups[7].trees), 1)
                self.assertTrue(any(sample.last_seen == 1 for sample in mapping.groups[7].biomes.values()))

    def test_compatible_repeated_shape_does_not_make_an_opposite_face_contradictory(self):
        horizontal = ((200., 200.), (263.173, 200.))
        repeated = ((500., 200.), (563.173, 200.))
        opposite = ((200., 260.), (263.173, 260.))
        mapping = self.mapping([horizontal, repeated], position=(100, 180))
        mapping.poses[7].position = np.array((100., 280.))
        mapping.update([observed((100, 280), [opposite])], 1)
        np.testing.assert_allclose(mapping.poses[7].position, (100, 280), atol=1e-8)
        self.assertLessEqual(mapping.poses[7].uncertainty, mapping.config.max_position_uncertainty)
        self.assertEqual(len(mapping.groups[7].edges), 5)

    def test_repeated_rectangle_shapes_reject_ambiguous_global_placements(self):
        repeated = [(tuple(np.array(start) + (-300, 150)), tuple(np.array(end) + (-300, 150)))
                    for start, end in STONE]
        mapping = self.mapping(STONE + repeated)
        mapping.poses[7].position = np.array((700., 500.))
        mapping.update([observed((500, 35), STONE)], 1)
        np.testing.assert_array_equal(mapping.poses[7].position, (700, 500))
        self.assertGreater(mapping.poses[7].uncertainty, mapping.config.max_position_uncertainty)
        self.assertEqual(len(mapping.groups[7].edges), 6)

    def test_known_shapes_with_conflicting_translations_cannot_vote_independently(self):
        mapping = self.mapping()
        mapping.poses[7].position = np.array((650., 35.))
        changed = [STONE[0], (tuple(np.array(STONE[1][0]) + (25, 0)),
                             tuple(np.array(STONE[1][1]) + (25, 0)))]
        mapping.update([observed((500, 35), changed)], 1)
        np.testing.assert_array_equal(mapping.poses[7].position, (650, 35))
        self.assertGreater(mapping.poses[7].uncertainty, mapping.config.max_position_uncertainty)
        self.assertEqual(len(mapping.groups[7].edges), 4)

    def test_similar_length_edge_is_not_used_or_coalesced_as_an_exact_shape(self):
        mapping = self.mapping(STONE[:1])
        changed = [(STONE[0][0], tuple(np.array(STONE[0][1]) + (1, 0)))]
        mapping.update([observed((520, 35), changed)], 1)
        np.testing.assert_allclose(mapping.poses[7].position, (520, 35), atol=1e-8)
        self.assertEqual(len(mapping.groups[7].edges), 4)

    def test_exact_shape_beats_nearer_wrong_length_edge_after_collision(self):
        # Seed 23 at 42.2s, translated 700 units upward to fit this fixture.
        # The old three-unit shape tolerance chose the nearer, longer edge
        # and introduced a confident 16.15-unit error.
        exact = np.array(((625.3969353780183, 155.7878841482517),
                          (625.3969353780184, 202.9252891955069)))
        wrong = np.array(((618.0306336460667, 140.2653591067175),
                          (618.0306336460668, 189.7111790837629)))
        actual = np.array((618.8923412644176, 195.6296832901705))
        predicted = np.array((614.558900458662, 186.61739774304033))
        mapping = self.mapping([exact, wrong], position=actual)
        mapping.poses[7].position = predicted.copy()
        projected = exact - actual + predicted
        self.assertLess(np.linalg.norm(wrong - projected, axis=1).max(),
                        np.linalg.norm(exact - projected, axis=1).max())
        shape_difference = np.linalg.norm((wrong[1] - wrong[0]) - (exact[1] - exact[0]))
        self.assertAlmostEqual(shape_difference, 2.3084149297901604)
        self.assertLess(shape_difference, mapping.config.landmark_consensus_tolerance)

        mapping.update([observed(actual, [exact])], 1)

        np.testing.assert_allclose(mapping.poses[7].position, actual, atol=1e-8)
        self.assertEqual(mapping.poses[7].uncertainty, mapping.config.sighting_uncertainty)
        self.assertEqual(len(mapping.groups[7].edges), 4)
        wrong_landmark = next(edge for edge in mapping.groups[7].edges if np.allclose(edge.start, wrong[0]))
        self.assertEqual(wrong_landmark.last_seen, 0)
        self.assertEqual(wrong_landmark.sightings, 1)

    def test_radius_includes_exact_twenty_unit_collision_with_roundoff(self):
        mapping = self.mapping(STONE[:1])
        mapping.poses[7].position[0] += 1e-10
        mapping.remember_actions([action(distance=10)])
        mapping.update([observed((510, 35), STONE[:1])], 1)
        np.testing.assert_allclose(mapping.poses[7].position, (510, 35), atol=1e-8)
        self.assertEqual(mapping.poses[7].uncertainty, 1)
        self.assertEqual(len(mapping.groups[7].edges), 3)

    def test_far_parallel_faces_alone_do_not_relocalize(self):
        parallel = [STONE[0], (tuple(np.array(STONE[0][0]) + (0, 60)),
                              tuple(np.array(STONE[0][1]) + (0, 60)))]
        mapping = self.mapping(parallel)
        mapping.poses[7].position = np.array((700., 500.))
        mapping.update([observed((500, 35), parallel)], 1)
        np.testing.assert_array_equal(mapping.poses[7].position, (700, 500))
        self.assertGreater(mapping.poses[7].uncertainty, mapping.config.max_position_uncertainty)


if __name__ == "__main__":
    unittest.main()

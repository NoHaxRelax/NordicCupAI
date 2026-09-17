"""Shared static geometry connects maps without inventing agent sightings."""

import unittest
from unittest.mock import patch

import numpy as np

from src.utils.controllers.world_estimator import WorldEstimator, rotate, wrap
from test_expert_policy import agent_state
from test_global_planner import planner_config, tree
from test_mapping import BOUNDARIES, seen_edges


CORNER = [((200, 200), (263.173, 200)), ((200, 200), (200, 247.612))]
SECOND = [((400, 300), (482.134, 300)), ((400, 300), (400, 353.927))]


def state(agent_id, position, heading, segments, extra=()):
    return agent_state(seen_edges(np.array(position, dtype=float), heading, segments) + list(extra),
                       agent_id=agent_id)


def estimator(**overrides):
    return WorldEstimator(planner_config(estimator={
        "shared_landmark_merges": True, "anchor_boundaries": False, **overrides,
    }).estimator)


class LandmarkMergeTests(unittest.TestCase):
    def test_shared_corner_merges_positions_headings_origins_and_map_layers(self):
        mapping = estimator()
        first, second = np.array((120., 130.)), np.array((400., 300.))
        first_heading, second_heading = 0.4, -0.7
        mapping.update([state(1, first, first_heading, CORNER),
                        state(2, second, second_heading, CORNER, [tree(25, 0)])], 0)
        self.assertEqual(set(mapping.groups), {1})
        expected = rotate(second - first, -first_heading)
        np.testing.assert_allclose(mapping.poses[2].position, expected, atol=1e-8)
        np.testing.assert_allclose(mapping.poses[2].origin, expected, atol=1e-8)
        self.assertAlmostEqual(mapping.poses[2].heading, wrap(second_heading - first_heading))
        group = mapping.groups[1]
        self.assertEqual(len(group.edges), 2)
        self.assertTrue(all(edge.sightings == 2 for edge in group.edges))
        self.assertEqual(len(group.biomes), 2)
        self.assertEqual(len(group.visited), 2)
        np.testing.assert_allclose(group.trees[0].position,
                                   expected + rotate((25, 0), second_heading - first_heading), atol=1e-8)
        self.assertEqual(mapping.links, {})
        self.assertFalse(group.anchored)

    def test_disabled_option_preserves_disconnected_frames(self):
        mapping = estimator(shared_landmark_merges=False)
        mapping.update([state(1, (100, 100), 0, CORNER), state(2, (300, 300), 1, CORNER)], 0)
        self.assertEqual(len(mapping.groups), 2)

    def test_single_or_parallel_segments_cannot_merge(self):
        for segments in (CORNER[:1], [CORNER[0], ((200, 247.612), (263.173, 247.612))]):
            with self.subTest(segments=segments):
                mapping = estimator()
                mapping.update([state(1, (100, 100), 0, segments),
                                state(2, (300, 300), 1, segments)], 0)
                self.assertEqual(len(mapping.groups), 2)

    def test_opposite_rectangle_faces_do_not_agree_on_a_transform(self):
        opposite = [((200, 247.612), (263.173, 247.612)),
                    ((263.173, 200), (263.173, 247.612))]
        mapping = estimator()
        mapping.update([state(1, (100, 100), 0, CORNER),
                        state(2, (300, 300), 1, opposite)], 0)
        self.assertEqual(len(mapping.groups), 2)

    def test_repeated_shape_with_two_supported_placements_is_ambiguous(self):
        repeated = [(tuple(np.array(start) + (300, 100)), tuple(np.array(end) + (300, 100)))
                    for start, end in CORNER]
        mapping = estimator()
        mapping.update([state(1, (100, 100), 0, CORNER + repeated),
                        state(2, (300, 300), 1, CORNER)], 0)
        self.assertEqual(len(mapping.groups), 2)

    def test_nearly_equal_lengths_are_not_geometric_identifiers(self):
        altered = [((200, 200), (263.183, 200)), CORNER[1]]
        mapping = estimator()
        mapping.update([state(1, (100, 100), 0, CORNER),
                        state(2, (300, 300), 1, altered)], 0)
        self.assertEqual(len(mapping.groups), 2)

    def test_matching_lengths_require_consistent_endpoint_positions(self):
        altered = [CORNER[0], ((210, 200), (210, 247.612))]
        mapping = estimator()
        mapping.update([state(1, (100, 100), 0, CORNER),
                        state(2, (300, 300), 1, altered)], 0)
        self.assertEqual(len(mapping.groups), 2)

    def test_boundary_faces_and_thickness_caps_are_not_stone_identities(self):
        for segments in (BOUNDARIES, [((0, 0), (30, 0)), ((0, 0), (0, 30))]):
            with self.subTest(segments=segments):
                mapping = estimator()
                mapping.update([state(1, (100, 100), 0, segments),
                                state(2, (300, 300), 1, segments)], 0)
                self.assertEqual(len(mapping.groups), 2)

    def test_anchored_reference_survives_even_with_a_higher_id(self):
        mapping = estimator(anchor_boundaries=True)
        mapping.update([state(8, (100, 150), 0.7, BOUNDARIES + CORNER)], 0)
        revision = mapping.groups[8].frame_revision
        mapping.update([state(1, (350, 400), -0.4, CORNER), agent_state(agent_id=8)], 1)
        self.assertEqual(set(mapping.groups), {8})
        np.testing.assert_allclose(mapping.poses[1].position, (350, 400), atol=1e-8)
        np.testing.assert_allclose(mapping.poses[1].origin, (350, 400), atol=1e-8)
        self.assertAlmostEqual(mapping.poses[1].heading, -0.4)
        self.assertEqual(mapping.groups[8].frame_revision, revision)
        self.assertEqual(mapping.links, {})

    def test_transform_contradicting_an_anchored_boundary_is_rejected(self):
        mapping = estimator(anchor_boundaries=True)
        mapping.update([state(8, (100, 150), 0.7, BOUNDARIES + CORNER)], 0)
        mapping.update([state(1, (350, 400), -0.4, CORNER + [((0, 90), (800, 90))]),
                        agent_state(agent_id=8)], 1)
        self.assertEqual(len(mapping.groups), 2)
        self.assertTrue(mapping.groups[8].anchored)

    def test_bridge_map_can_connect_three_groups_in_one_pass(self):
        mapping = estimator()
        mapping.update([state(1, (100, 100), 0, CORNER),
                        state(2, (300, 300), 1, CORNER + SECOND),
                        state(3, (600, 400), -0.6, SECOND)], 0)
        self.assertEqual(set(mapping.groups), {1})
        np.testing.assert_allclose(mapping.poses[3].position, (500, 300), atol=1e-8)
        self.assertAlmostEqual(mapping.poses[3].heading, -0.6)
        self.assertEqual(len(mapping.groups[1].edges), 4)

    def test_attempts_are_throttled_and_unchanged_geometry_is_not_retested(self):
        mapping = estimator()
        single = [state(1, (100, 100), 0, CORNER[:1]), state(2, (300, 300), 1, CORNER[:1])]
        both = [state(1, (100, 100), 0, CORNER), state(2, (300, 300), 1, CORNER)]
        original = mapping._shared_landmark_transform
        with patch.object(mapping, "_shared_landmark_transform", wraps=original) as matching:
            mapping.update(single, 0)
            mapping.update(single, 1)
            self.assertEqual(matching.call_count, 1)
            mapping.update(both, 1.2)
            self.assertEqual(matching.call_count, 1)
            self.assertEqual(len(mapping.groups), 2)
            mapping.update(both, 2)
            self.assertEqual(matching.call_count, 2)
        self.assertEqual(len(mapping.groups), 1)

    def test_candidate_overflow_rejects_pair_instead_of_accepting_partial_evidence(self):
        mapping = estimator(shared_landmark_max_candidates=2)
        repeated = [(tuple(np.array(start) + (300, 100)), tuple(np.array(end) + (300, 100)))
                    for start, end in CORNER]
        mapping.update([state(1, (100, 100), 0, CORNER + repeated),
                        state(2, (300, 300), 1, CORNER)], 0)
        self.assertEqual(len(mapping.groups), 2)


if __name__ == "__main__":
    unittest.main()

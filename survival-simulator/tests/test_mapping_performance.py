"""Regression coverage for reusable geometry, frame identity and spawn origins."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from src.utils.controllers import world_estimator as mapping
from src.utils.controllers.world_estimator import EstimatedPose, WorldEstimator
from test_expert_policy import agent_state
from test_global_planner import action, planner_config, sighting
from test_mapping import BOUNDARIES, edge, seen_edges


class MappingPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.estimator = WorldEstimator(planner_config().estimator)

    def test_origin_is_independent_of_movement_and_transforms_with_the_map(self):
        pose = EstimatedPose(7, 7, np.array((4., 8.)))
        pose.position += (10, 0)
        np.testing.assert_array_equal(pose.origin, (4, 8))

        self.estimator.update([agent_state(agent_id=7)], 0)
        self.estimator.remember_actions([action(distance=10)])
        position = np.array((100., 150.))
        self.estimator.update([agent_state(seen_edges(position, 0, BOUNDARIES), agent_id=7)], 1)
        pose = self.estimator.poses[7]
        np.testing.assert_allclose(pose.position, position, atol=1e-8)
        np.testing.assert_allclose(pose.origin, (90, 150), atol=1e-8)
        self.assertEqual(self.estimator.snapshot()["agents"][0]["origin"], pose.origin.tolist())

    def test_incoming_agent_does_not_change_reference_frame_revision_or_origin(self):
        self.estimator.update([agent_state(seen_edges(np.array((100., 150.)), 0, BOUNDARIES),
                                          agent_id=1)], 0)
        group = self.estimator.groups[1]
        frame_revision, map_revision = group.frame_revision, group.revision
        self.estimator.update([agent_state([sighting(2, 50, 0, 0.4)], agent_id=1),
                               agent_state(agent_id=2)], 1)
        self.assertEqual(group.frame_revision, frame_revision)
        self.assertGreater(group.revision, map_revision)
        np.testing.assert_allclose(self.estimator.poses[1].origin, (100, 150), atol=1e-8)
        np.testing.assert_allclose(self.estimator.poses[2].origin, (150, 150), atol=1e-8)

    def test_moving_group_origin_survives_rotated_merge(self):
        self.estimator.update([agent_state(agent_id=1), agent_state(agent_id=2)], 0)
        self.estimator.remember_actions([action(agent_id=2, distance=10)])
        self.estimator.update([agent_state([sighting(2, 50, 20, math.pi / 2)], agent_id=1),
                               agent_state(agent_id=2)], 1)
        np.testing.assert_allclose(self.estimator.poses[2].position, (50, 20), atol=1e-8)
        np.testing.assert_allclose(self.estimator.poses[2].origin, (50, 10), atol=1e-8)

    def test_repeated_rays_parse_once_per_observer_and_keep_edge_counts(self):
        observation = edge((40, -10), (40, 60))
        state = agent_state([observation] * 40)
        original = mapping.edge_sightings
        with patch.object(mapping, "edge_sightings", wraps=original) as parsing:
            self.estimator.update([state], 0)
            self.assertEqual(parsing.call_count, 1)
        self.estimator.update([state], 1)
        self.assertEqual(len(self.estimator.groups[7].edges), 1)
        self.assertEqual(self.estimator.groups[7].edges[0].sightings, 2)

    def test_new_evidence_from_earlier_agent_keeps_required_second_correction(self):
        self.estimator.update([agent_state([sighting(2, 50, 0)], agent_id=1),
                               agent_state(agent_id=2)], 0)
        self.estimator.remember_actions([action(agent_id=2, distance=10)])
        self.estimator.update([
            agent_state([edge((80, 10), (140, 10))], agent_id=1),
            agent_state([edge((30, 10), (90, 10))], agent_id=2),
        ], 1)
        np.testing.assert_allclose(self.estimator.poses[2].position, (50, 0), atol=1e-8)
        self.assertEqual(len(self.estimator.groups[1].edges), 1)

    def test_landmark_arrays_follow_recency_reordering_and_eviction(self):
        first, second = edge((30, 10), (70, 10)), edge((0, 40), (0, 80))
        self.estimator.update([agent_state([first, second])], 0)
        group = self.estimator.groups[7]
        group.edge_positions()
        self.estimator.update([agent_state([second])], 1)
        self.estimator.update([agent_state([first])], 2)
        stored = group.edge_positions()
        np.testing.assert_array_equal(stored, [[e.start, e.end] for e in group.edges])
        self.estimator._retain_landmarks(group, "edges", "_edge_positions", 1)
        np.testing.assert_array_equal(group.edge_positions(), [[group.edges[0].start, group.edges[0].end]])
        self.assertEqual(group.edges[0].last_seen, 2)

    def test_failed_anchor_is_retried_only_after_edge_geometry_changes(self):
        horizontal = edge((0, 30), (800, 30))
        original = mapping.infer_boundary_frame
        with patch.object(mapping, "infer_boundary_frame", wraps=original) as inference:
            self.estimator.update([agent_state([horizontal])], 0)
            self.estimator.update([agent_state([horizontal])], 1)
            self.estimator.update([agent_state([horizontal])], 2)
            self.assertEqual(inference.call_count, 1)
            self.estimator.update([agent_state([horizontal, edge((30, 0), (30, 600))])], 3)
            self.assertEqual(inference.call_count, 2)
        self.assertTrue(self.estimator.groups[7].anchored)


if __name__ == "__main__":
    unittest.main()

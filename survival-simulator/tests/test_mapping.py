"""Coordinate geometry and evidence lifetime, independent of true simulator state."""

import json
import math
import unittest

import numpy as np

from src.utils.controllers.world_estimator import WorldEstimator, rotate
from test_expert_policy import agent_state
from test_global_planner import action, planner_config, sighting, tree


def edge(start, end):
    return {"type": "Edge", "coords": [list(start), list(end)]}


def seen_edges(position, heading, segments):
    return [edge(rotate(np.array(start) - position, -heading),
                 rotate(np.array(end) - position, -heading)) for start, end in segments]


BOUNDARIES = [((0, 30), (800, 30)), ((30, 0), (30, 600))]


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.estimator = WorldEstimator(planner_config().estimator)

    def test_boundary_anchor_transforms_agents_trees_edges_and_biome_samples(self):
        position, heading = np.array([123., 234.]), 1.17
        observations = seen_edges(position, heading, BOUNDARIES) + [tree(20, 0)]
        self.estimator.update([agent_state(observations, agent_id=1, biome="swamp")], 0)
        pose, group = self.estimator.poses[1], self.estimator.groups[1]
        self.assertTrue(group.anchored)
        np.testing.assert_allclose(group.world_size, [800, 600], atol=1e-8)
        np.testing.assert_allclose(pose.position, position, atol=1e-8)
        self.assertAlmostEqual(pose.heading, heading)
        np.testing.assert_allclose(group.trees[0].position, position + rotate((20, 0), heading))
        sample = next(iter(group.biomes.values()))
        self.assertEqual(sample.biome, "swamp")
        np.testing.assert_allclose(sample.position, position)
        for boundary in BOUNDARIES:
            self.assertTrue(any(np.allclose(e.start, boundary[0]) and np.allclose(e.end, boundary[1])
                                for e in group.edges))
        self.assertGreater(group.frame_revision, 0)

    def test_two_independently_anchored_groups_merge_without_agent_sightings(self):
        first, second = np.array([123., 234.]), np.array([500., 400.])
        self.estimator.update([
            agent_state(seen_edges(first, 0.7, BOUNDARIES), agent_id=1),
            agent_state(seen_edges(second, -0.5, BOUNDARIES), agent_id=2),
        ], 0)
        self.assertEqual(set(self.estimator.groups), {1})
        np.testing.assert_allclose(self.estimator.poses[1].position, first, atol=1e-8)
        np.testing.assert_allclose(self.estimator.poses[2].position, second, atol=1e-8)
        self.assertEqual(len(self.estimator.groups[1].edges), 2)
        self.assertEqual(self.estimator.links, {})

    def test_single_wall_or_ordinary_stone_does_not_invent_absolute_coordinates(self):
        self.estimator.update([
            agent_state([edge((0, 40), (800, 40))], agent_id=1),
            agent_state([edge((0, 30), (80, 30)), edge((30, 0), (30, 60))], agent_id=2),
        ], 0)
        self.assertEqual(len(self.estimator.groups), 2)
        self.assertTrue(all(not group.anchored for group in self.estimator.groups.values()))

    def test_overlapping_newborn_records_contact_without_inventing_a_heading(self):
        # atan2(0, 0) is zero in BOTH directions, invalidating the usual +pi.
        observation = {"type": "Agent", "id": 2, "distance": 0., "angle": -0.8, "rel_dir": -1.2}
        self.estimator.update([agent_state([observation], agent_id=1), agent_state(agent_id=2)], 0)
        self.assertEqual(len(self.estimator.groups), 2)
        self.assertIsNone(self.estimator.links[(1, 2)].relative_heading)
        self.estimator.update([agent_state([sighting(2, 5, 0, 0.4)], agent_id=1),
                               agent_state(agent_id=2)], 0.1)
        self.assertEqual(len(self.estimator.groups), 1)
        self.assertAlmostEqual(self.estimator.poses[2].heading, 0.4)

    def test_stale_observations_do_not_undo_motion_or_add_map_evidence(self):
        estimator = WorldEstimator(planner_config(estimator={"detect_stale_observations": True}).estimator)
        observation = edge((40, -10), (40, 60))
        state = agent_state([observation], age=5)
        estimator.update([state], 0)
        estimator.remember_actions([action(distance=10)])
        estimator.update([state], 0.1)  # Action ran, but cached vision/age did not refresh.
        np.testing.assert_allclose(estimator.poses[7].position, [10, 0], atol=1e-9)
        self.assertEqual(estimator.groups[7].edges[0].sightings, 1)
        self.assertEqual(next(iter(estimator.groups[7].biomes.values())).last_seen, 0)
        estimator.update([agent_state([edge((20, -10), (20, 60))], age=5.1)], 0.2)
        np.testing.assert_allclose(estimator.poses[7].position, [20, 0], atol=1e-9)
        self.assertEqual(estimator.groups[7].edges[0].sightings, 2)

    def test_anchored_group_survives_lower_id_unanchored_observer(self):
        absolute = np.array([123., 234.])
        self.estimator.update([
            agent_state(agent_id=1),
            agent_state(seen_edges(absolute, 0, BOUNDARIES), agent_id=8),
        ], 0)
        self.estimator.update([
            agent_state([sighting(8, 50, 20)], agent_id=1), agent_state(agent_id=8),
        ], 0.1)
        self.assertEqual(set(self.estimator.groups), {8})
        self.assertTrue(self.estimator.groups[8].anchored)
        np.testing.assert_allclose(self.estimator.poses[8].position, absolute, atol=1e-8)
        np.testing.assert_allclose(self.estimator.poses[1].position, absolute - [50, 20], atol=1e-8)

    def test_link_history_survives_separation_and_bridge_death_without_reintegrating_retry(self):
        self.estimator.update([
            agent_state([sighting(2, 100, 0)], agent_id=1),
            agent_state([sighting(3, 100, 0)], agent_id=2), agent_state(agent_id=3),
        ], 0)
        self.estimator.update([agent_state(agent_id=1), agent_state(agent_id=3)], 1)
        self.assertEqual(len(self.estimator.groups), 1)
        self.assertEqual(len(self.estimator.links), 2)
        self.assertEqual(self.estimator.links[(1, 2)].last_seen, 0)
        self.estimator.update([agent_state([sighting(3, 200, 0)], agent_id=1), agent_state(agent_id=3)], 2)
        snapshot = self.estimator.snapshot()
        self.estimator.update([agent_state(agent_id=1), agent_state(agent_id=3)], 2)
        self.assertEqual(snapshot, self.estimator.snapshot())
        self.assertEqual(self.estimator.links[(1, 3)].count, 1)

    def test_whole_static_segment_corrects_blocked_motion_and_deduplicates_rays(self):
        observation = edge((40, -10), (40, 60))
        self.estimator.update([agent_state([observation] * 5)], 0)
        self.estimator.remember_actions([action(distance=10)])
        self.estimator.update([agent_state([observation] * 5)], 0.1)
        group = self.estimator.groups[7]
        np.testing.assert_allclose(self.estimator.poses[7].position, [0, 0], atol=1e-9)
        self.assertEqual(len(group.edges), 1)
        self.assertEqual(group.edges[0].sightings, 2)
        self.assertEqual(self.estimator.poses[7].uncertainty, 1)

    def test_full_boundary_recovers_large_collision_without_creating_ghost_wall(self):
        position = np.array([700., 450.])
        right = ((770, 0), (770, 600))
        self.estimator.update([agent_state(seen_edges(position, 0.7, BOUNDARIES + [right]))], 0)
        self.estimator.poses[7].position += [15, 23]
        self.estimator.update([agent_state(seen_edges(position, 0.7, [right]))], 0.1)
        np.testing.assert_allclose(self.estimator.poses[7].position, position, atol=1e-8)
        self.assertEqual(len(self.estimator.groups[7].edges), 3)
        self.assertEqual(self.estimator.poses[7].uncertainty, 1)

    def test_conflicting_boundary_marks_pose_uncertain_instead_of_creating_ghost_geometry(self):
        position = np.array([700., 450.])
        right = ((770, 0), (770, 600))
        self.estimator.update([agent_state(seen_edges(position, 0, BOUNDARIES + [right]))], 0)
        self.estimator.poses[7].position += [50, 23]
        self.estimator.update([agent_state(seen_edges(position, 0, [right]) + [tree(10, 0)])], 0.1)
        pose, group = self.estimator.poses[7], self.estimator.groups[7]
        self.assertAlmostEqual(pose.position[1], position[1])
        self.assertGreater(pose.uncertainty, self.estimator.config.max_position_uncertainty)
        self.assertEqual(len(group.edges), 3)
        self.assertEqual(group.trees, [])
        self.assertEqual(next(iter(group.biomes.values())).last_seen, 0)

    def test_merge_rotates_stones_and_biomes_and_keeps_identity(self):
        self.estimator.update([
            agent_state(agent_id=1, biome="forest"),
            agent_state([edge((30, 0), (60, 0))], agent_id=2, biome="desert"),
        ], 0)
        self.estimator.update([
            agent_state([sighting(2, 100, 20, math.pi / 2)], agent_id=1, biome="forest"),
            agent_state(agent_id=2, biome="desert"),
        ], 1)
        group = self.estimator.groups[1]
        np.testing.assert_allclose(group.edges[0].start, [100, 50], atol=1e-8)
        np.testing.assert_allclose(group.edges[0].end, [100, 80], atol=1e-8)
        samples = [sample for sample in group.biomes.values() if sample.biome == "desert"]
        self.assertEqual(len(samples), 1)
        np.testing.assert_allclose(samples[0].position, [100, 20], atol=1e-8)

    def test_stones_persist_when_trees_expire_and_memory_is_bounded(self):
        estimator = WorldEstimator(planner_config(estimator={
            "max_edges_per_group": 2, "max_sighting_links": 1,
            "max_visited_cells_per_group": 1,
        }).estimator)
        estimator.update([
            agent_state([edge((10, 0), (10, 30)), edge((50, 0), (50, 30)),
                         edge((90, 0), (90, 30)), tree(10, 20),
                         sighting(2, 100, 0), sighting(3, 200, 0)], agent_id=1),
            agent_state(agent_id=2), agent_state(agent_id=3),
        ], 0)
        estimator.update([agent_state(agent_id=1)], 61)
        group = estimator.groups[1]
        self.assertEqual(len(group.edges), 2)
        self.assertEqual(group.trees, [])
        self.assertEqual(len(group.biomes), 1)
        self.assertEqual(len(estimator.links), 1)

    def test_snapshot_serializes_evidence_and_reset_clears_all_layers(self):
        self.estimator.update([agent_state([edge((10, 0), (10, 30))])], 10)
        snapshot = json.loads(json.dumps(self.estimator.snapshot(), allow_nan=False))
        self.assertEqual(snapshot["groups"][0]["biomes"][0]["biome"], "grassland")
        self.assertEqual(len(snapshot["groups"][0]["edges"]), 1)
        self.estimator.update([agent_state(agent_id=8)], 0)
        self.assertEqual(set(self.estimator.groups), {8})
        self.assertEqual(self.estimator.groups[8].edges, [])
        self.estimator.update([], 1)
        self.assertEqual(self.estimator.snapshot()["groups"], [])

    def test_estimate_is_independent_of_observation_and_agent_order(self):
        states = [agent_state(seen_edges(np.array([123., 234.]), 0.3, BOUNDARIES), agent_id=1),
                  agent_state([edge((20, 30), (20, 70)), sighting(1, 100, 0)], agent_id=2)]
        other = WorldEstimator(planner_config().estimator)
        self.estimator.update(states, 0)
        other.update([dict(s, observations=list(reversed(s["observations"])))
                      for s in reversed(states)], 0)
        self.assertEqual(self.estimator.snapshot(), other.snapshot())


if __name__ == "__main__":
    unittest.main()

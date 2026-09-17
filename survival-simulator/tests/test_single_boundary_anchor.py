"""Absolute origins and observed dimensions need not be learned together."""

import math
import unittest

import numpy as np

from src.utils.controllers.boundary_anchor import infer_directed_boundary_axes, infer_single_boundary_frame
from src.utils.controllers.world_estimator import WorldEstimator, rotate
from src.utils.sensing import compute_visibility
from test_boundary_anchor import local_edges, rotation
from test_expert_policy import agent_state
from test_global_planner import planner_config, tree
from test_mapping import seen_edges


TOP = ((0, 30), (800, 30))
BOTTOM = ((0, 570), (800, 570))
LEFT = ((30, 0), (30, 600))
RIGHT = ((770, 0), (770, 600))
HORIZONTAL = ((150, 180), (213.173, 180))
VERTICAL = ((150, 180), (150, 251.273))


def state(position, heading, segments, agent_id=7, **kwargs):
    return agent_state(seen_edges(np.array(position, dtype=float), heading, segments), agent_id=agent_id, **kwargs)


def mapping(**changes):
    return WorldEstimator(planner_config(estimator={"anchor_single_boundary": True, **changes}).estimator)


class SingleBoundaryGeometryTests(unittest.TestCase):
    def test_top_and_left_anchor_under_arbitrary_local_frames(self):
        rng = np.random.default_rng(427)
        for wall, stone, dimensions in ((TOP, VERTICAL, (800, None)), (LEFT, HORIZONTAL, (None, 600))):
            for _ in range(15):
                angle = rng.uniform(-math.pi, math.pi)
                translation = rng.uniform(-2000, 2000, size=2)
                edges = local_edges([wall, stone], angle, translation)
                observer = rotation(angle) @ (100, 100) + translation
                frame = infer_single_boundary_frame(edges, observer)
                self.assertIsNotNone(frame)
                self.assertIsNone(frame.world_size)
                for actual, expected in zip((frame.known_width, frame.known_height), dimensions):
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        self.assertAlmostEqual(actual, expected)
                for point in ((0, 0), (800, 600), (100, 100)):
                    local = rotation(angle) @ point + translation
                    np.testing.assert_allclose(rotation(frame.angle) @ local + frame.offset, point, atol=1e-8)

    def test_bottom_and_right_require_the_perpendicular_dimension(self):
        for wall, stone, observer in ((BOTTOM, VERTICAL, (100, 450)), (RIGHT, HORIZONTAL, (650, 100))):
            edges = local_edges([wall, stone], 0.7, (-200, 900))
            observer = rotation(0.7) @ observer + (-200, 900)
            self.assertIsNone(infer_single_boundary_frame(edges, observer))
            frame = infer_single_boundary_frame(edges, observer, known_width=800, known_height=600)
            self.assertIsNotNone(frame)
            np.testing.assert_allclose(frame.world_size, (800, 600), atol=1e-8)
            np.testing.assert_allclose(rotation(frame.angle) @ (-200, 900) + frame.offset, (0, 0), atol=1e-8)

    def test_single_direction_does_not_identify_the_world_axes(self):
        for segments in ([TOP], [TOP, HORIZONTAL], [VERTICAL, HORIZONTAL]):
            self.assertIsNone(infer_single_boundary_frame(local_edges(segments), (100, 100)))
        self.assertIsNone(infer_directed_boundary_axes(local_edges([TOP, HORIZONTAL])))

    def test_inconsistent_axes_dimensions_and_retained_positions_are_rejected(self):
        for extra in (((0, 80), (900, 80)), ((15, 30), (815, 30)),
                      ((150, 180), (155, 250)), ((200, 200), (130, 200))):
            with self.subTest(extra=extra):
                self.assertIsNone(infer_single_boundary_frame(
                    local_edges([TOP, VERTICAL]), (100, 100), orientation_edges=local_edges([extra])))
        self.assertIsNone(infer_single_boundary_frame(local_edges([TOP, VERTICAL]), (100, 100), known_width=900))

    def test_close_on_wall_and_inside_wall_outer_faces_are_ambiguous(self):
        for observer in ((100, 30), (100, 50), (100, 63)):
            self.assertIsNone(infer_single_boundary_frame(local_edges([TOP, VERTICAL]), observer))
        outer = ((0, 0), (800, 0))
        self.assertIsNone(infer_single_boundary_frame(local_edges([outer, VERTICAL]), (100, 10)))
        self.assertIsNone(infer_single_boundary_frame(local_edges([LEFT, HORIZONTAL]), (10, 100), known_width=800))

    def test_raycast_from_free_interior_returns_inner_face_before_outer(self):
        outer = ((0, 0), (800, 0))
        _, hits = compute_visibility(100, 100, -math.pi / 2, math.pi / 2, 200, [outer, TOP])
        self.assertTrue(hits)
        self.assertTrue(all(edge == TOP for edge in hits))


class SingleBoundaryEstimatorTests(unittest.TestCase):
    def test_optional_feature_is_disabled_by_default(self):
        estimator = WorldEstimator(planner_config().estimator)
        estimator.update([state((100, 100), 0.7, [TOP, VERTICAL])], 0)
        self.assertFalse(estimator.groups[7].anchored)

    def test_partial_frame_transforms_all_layers_and_exports_known_dimension(self):
        estimator = mapping()
        observation = state((100, 100), 0.7, [TOP, VERTICAL], biome="swamp")
        observation["observations"].append(tree(20, 0))
        estimator.update([observation], 0)
        group, pose = estimator.groups[7], estimator.poses[7]
        self.assertTrue(group.anchored)
        self.assertIsNone(group.world_size)
        self.assertAlmostEqual(group.known_width, 800)
        self.assertIsNone(group.known_height)
        np.testing.assert_allclose(pose.position, (100, 100), atol=1e-8)
        np.testing.assert_allclose(pose.origin, (100, 100), atol=1e-8)
        self.assertAlmostEqual(pose.heading, 0.7)
        np.testing.assert_allclose(group.trees[0].position, (100, 100) + rotate((20, 0), 0.7), atol=1e-8)
        np.testing.assert_allclose(next(iter(group.biomes.values())).position, (100, 100), atol=1e-8)
        exported = estimator.snapshot()["groups"][0]
        self.assertAlmostEqual(exported["known_width"], 800)
        self.assertIsNone(exported["known_height"])
        self.assertIsNone(exported["world_size"])
        # Correction/validation must safely handle anchored frames of unknown size.
        estimator.update([state((100, 100), 0.7, [TOP, VERTICAL], biome="swamp")], 1)
        np.testing.assert_allclose(pose.position, (100, 100), atol=1e-8)

    def test_disconnected_near_or_far_wall_groups_pool_dimensions_and_merge(self):
        for horizontal, vertical, first, second in (
            (TOP, LEFT, (100, 100), (200, 300)),
            (BOTTOM, RIGHT, (100, 450), (650, 300)),
        ):
            with self.subTest(horizontal=horizontal):
                estimator = mapping()
                estimator.update([state(first, 0.7, [horizontal, VERTICAL], 1),
                                  state(second, -1.2, [vertical, HORIZONTAL], 2)], 0)
                self.assertEqual(set(estimator.groups), {1})
                group = estimator.groups[1]
                self.assertTrue(group.anchored)
                np.testing.assert_allclose(group.world_size, (800, 600), atol=1e-8)
                np.testing.assert_allclose(estimator.poses[1].position, first, atol=1e-8)
                np.testing.assert_allclose(estimator.poses[2].position, second, atol=1e-8)
                self.assertEqual(estimator.links, {})

    def test_two_partial_frames_merge_before_height_is_known(self):
        estimator = mapping()
        estimator.update([state((100, 100), 0.7, [TOP, VERTICAL], 1),
                          state((600, 150), -1.2, [TOP, VERTICAL], 2)], 0)
        self.assertEqual(set(estimator.groups), {1})
        self.assertTrue(estimator.groups[1].anchored)
        self.assertIsNone(estimator.groups[1].world_size)
        np.testing.assert_allclose(estimator.poses[2].position, (600, 150), atol=1e-8)

    def test_later_height_completes_bounds_without_changing_absolute_frame(self):
        estimator = mapping()
        estimator.update([state((100, 100), 0.7, [TOP, VERTICAL], 1)], 0)
        revision = estimator.groups[1].frame_revision
        estimator.update([state((100, 100), 0.7, [], 1),
                          state((200, 300), -1.2, [LEFT, HORIZONTAL], 2)], 1)
        group = estimator.groups[1]
        np.testing.assert_allclose(group.world_size, (800, 600), atol=1e-8)
        self.assertEqual(group.frame_revision, revision)
        np.testing.assert_allclose(estimator.poses[1].origin, (100, 100), atol=1e-8)
        np.testing.assert_allclose(estimator.poses[2].origin, (200, 300), atol=1e-8)

    def test_axis_basis_can_come_from_earlier_stone_observations(self):
        estimator = mapping()
        estimator.update([state((100, 100), 0.7, [VERTICAL])], 0)
        self.assertFalse(estimator.groups[7].anchored)
        estimator.update([state((100, 100), 0.7, [TOP])], 1)
        self.assertTrue(estimator.groups[7].anchored)
        np.testing.assert_allclose(estimator.poses[7].position, (100, 100), atol=1e-8)

    def test_conflicting_dimensions_never_merge_absolute_maps(self):
        estimator = mapping()
        estimator.update([state((100, 100), 0.7, [TOP, VERTICAL], 1)], 0)
        wider_top = ((0, 30), (900, 30))
        estimator.update([state((100, 100), 0.7, [], 1),
                          state((200, 300), -1.2, [wider_top, LEFT], 2)], 1)
        self.assertEqual(set(estimator.groups), {1, 2})
        self.assertTrue(estimator._dimension_conflict)
        self.assertIsNone(estimator.groups[1].world_size)
        self.assertTrue(estimator.groups[2].anchored)

    def test_partial_map_rejects_conflicting_new_boundary(self):
        estimator = mapping()
        estimator.update([state((100, 100), 0, [TOP, VERTICAL])], 0)
        estimator.update([state((100, 100), 0, [((0, 30), (900, 30))])], 1)
        self.assertGreater(estimator.poses[7].uncertainty, estimator.config.max_position_uncertainty)
        self.assertEqual(len(estimator.groups[7].edges), 2)

    def test_reset_forgets_global_dimension_facts(self):
        estimator = mapping()
        estimator.update([state((100, 100), 0, [TOP, VERTICAL])], 0)
        self.assertAlmostEqual(estimator.known_width, 800)
        estimator.reset()
        self.assertIsNone(estimator.known_width)
        self.assertIsNone(estimator.known_height)
        self.assertFalse(estimator._dimension_conflict)


if __name__ == "__main__":
    unittest.main()

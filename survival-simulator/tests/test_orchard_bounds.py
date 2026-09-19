"""Synthetic public-observation checks; no engine imports or world lookups."""
import math
import unittest

from models.survival import oscar_orchard as orchard
from models.observed_bounds import ObservedBoundsOrchard


def state_with_edges(edges, position=(100., 110.), heading=.37):
    def local(point):
        return orchard.rot(orchard.sub(point, position), -heading)
    return {
        "agent_id": 1, "energy": 150., "age": 0., "biome": "forest",
        "speed": 10., "sprint_speed": 15., "hearing_radius": 50.,
        "vision_angle": math.pi / 3, "vision_range": 200., "max_energy": 500.,
        "observations": [{"type": "Edge", "coords": [local(a), local(b)]}
                         for a, b in edges],
    }


def observe(policy, state):
    if 1 not in policy.minds:
        policy._register([1], {1: state})
    policy._observe(policy.minds[1], state)
    return policy.minds[1]


class ObservedBoundsTests(unittest.TestCase):
    def setUp(self):
        self.original_bounds = orchard.W, orchard.H

    def tearDown(self):
        orchard.W, orchard.H = self.original_bounds

    def test_single_edge_does_not_invent_other_dimension(self):
        policy = ObservedBoundsOrchard()
        mind = observe(policy, state_with_edges([((0., 25.), (1800., 25.))]))
        self.assertFalse(policy.bounds_audit["inferred"])
        self.assertIsNone(policy.bounds_audit["width"])
        self.assertFalse(policy.groups[mind.group].anchored)
        self.assertEqual(mind.pose.p, (0., 0.))
        self.assertTrue(math.isinf(orchard.W) and math.isinf(orchard.H))

    def test_rotated_corner_infers_arbitrary_dimensions_and_inset(self):
        for width, height, inset, position in [
            (1800., 1300., 25., (100., 110.)),
            (2400., 1700., 42., (180., 140.)),
        ]:
            with self.subTest(width=width):
                policy = ObservedBoundsOrchard()
                edges = [((0., inset), (width, inset)),
                         ((inset, 0.), (inset, height))]
                mind = observe(policy, state_with_edges(edges, position))
                self.assertTrue(policy.groups[mind.group].anchored)
                for key, value in [("width", width), ("height", height), ("inset", inset)]:
                    self.assertAlmostEqual(policy.bounds_audit[key], value)
                self.assertAlmostEqual(mind.pose.p[0], position[0])
                self.assertAlmostEqual(mind.pose.p[1], position[1])
                self.assertAlmostEqual(mind.pose.theta, .37)
                self.assertTrue(policy.bounds_audit["evidence"]["same_observation"])

    def test_bottom_right_corner_and_observed_landmark_use_same_frame(self):
        policy = ObservedBoundsOrchard()
        width, height, inset = 1800., 1300., 25.
        position = (width - 100., height - 110.)
        edges = [((0., height - inset), (width, height - inset)),
                 ((width - inset, 0.), (width - inset, height))]
        state = state_with_edges(edges, position)
        state["observations"].append({"type": "Tree", "distance": 20., "angle": 0.})
        mind = observe(policy, state)
        self.assertTrue(policy.groups[mind.group].anchored)
        tree = next(iter(policy.groups[mind.group].trees.values()))
        self.assertAlmostEqual(tree.p[0], position[0] + 20 * math.cos(.37))
        self.assertAlmostEqual(tree.p[1], position[1] + 20 * math.sin(.37))

    def test_tall_world_uses_longer_axis_without_reflecting_pose(self):
        policy = ObservedBoundsOrchard()
        edges = [((0., 25.), (1300., 25.)), ((25., 0.), (25., 1800.))]
        mind = observe(policy, state_with_edges(edges))
        self.assertAlmostEqual(policy.bounds_audit["width"], 1800.)
        self.assertAlmostEqual(policy.bounds_audit["height"], 1300.)
        self.assertAlmostEqual(mind.pose.p[0], 110.)
        self.assertAlmostEqual(mind.pose.p[1], 1200.)
        self.assertAlmostEqual(mind.pose.theta, .37 - math.pi / 2)

    def test_inconsistent_or_nonperpendicular_edges_do_not_anchor(self):
        for edges in [
            [((0., 25.), (1800., 25.)), ((40., 0.), (40., 1300.))],
            [((0., 25.), (1800., 25.)), ((25., 0.), (35., 1300.))],
            [((0., 25.), (1800., 25.)), ((25., 0.), (25., 1800.))],
        ]:
            with self.subTest(edges=edges):
                policy = ObservedBoundsOrchard()
                mind = observe(policy, state_with_edges(edges))
                self.assertFalse(policy.bounds_audit["inferred"])
                self.assertFalse(policy.groups[mind.group].anchored)


if __name__ == "__main__":
    unittest.main()

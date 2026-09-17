from dataclasses import dataclass
import math
import unittest

import numpy as np

from src.utils.controllers.boundary_anchor import infer_boundary_frame


@dataclass
class Edge:
    start: np.ndarray
    end: np.ndarray
    last_seen: float = 0.0
    sightings: int = 1


def rotation(angle):
    return np.array(((math.cos(angle), -math.sin(angle)),
                     (math.sin(angle), math.cos(angle))))


def local_edges(segments, angle=0.0, translation=(0, 0)):
    transform = rotation(angle)
    return [Edge(transform @ np.array(start) + translation,
                 transform @ np.array(end) + translation)
            for start, end in segments]


class BoundaryAnchorTests(unittest.TestCase):
    def assert_frame(self, frame, angle, translation, size):
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame.world_size, size, atol=1e-8)
        for point in ((0, 0), size, (73, 165)):
            local = rotation(angle) @ point + translation
            recovered = rotation(frame.angle) @ local + frame.offset
            np.testing.assert_allclose(recovered, point, atol=1e-8)

    def test_all_inner_corners_under_random_rotations_and_translations(self):
        rng = np.random.default_rng(721)
        for width, height in ((1300, 850), (1000, 1000)):
            for x in (30, width - 30):
                for y in (30, height - 30):
                    for _ in range(8):
                        angle = rng.uniform(-math.pi, math.pi)
                        translation = rng.uniform(-2000, 2000, size=2)
                        segments = [((0, y), (width, y)), ((x, 0), (x, height))]
                        edges = local_edges(segments, angle, translation)
                        rng.shuffle(edges)
                        with self.subTest(size=(width, height), corner=(x, y), angle=angle):
                            self.assert_frame(infer_boundary_frame(edges), angle, translation, (width, height))

    def test_all_inner_and_outer_faces_can_agree(self):
        width, height = 1500, 900
        segments = [((0, y), (width, y)) for y in (0, 30, height - 30, height)]
        segments += [((x, 0), (x, height)) for x in (0, 30, width - 30, width)]
        self.assert_frame(infer_boundary_frame(local_edges(segments, 2.8, (-800, 450))),
                          2.8, (-800, 450), (width, height))

    def test_one_wall_and_parallel_faces_are_insufficient(self):
        self.assertIsNone(infer_boundary_frame([]))
        self.assertIsNone(infer_boundary_frame(local_edges([((0, 30), (1000, 30))])))
        self.assertIsNone(infer_boundary_frame(local_edges([
            ((0, 0), (1000, 0)), ((0, 30), (1000, 30)), ((0, 970), (1000, 970)),
        ])))

    def test_ordinary_stones_do_not_anchor_a_map(self):
        self.assertIsNone(infer_boundary_frame(local_edges([
            ((100, 100), (170, 100)), ((100, 100), (100, 180)),
        ])))
        self.assertIsNone(infer_boundary_frame(local_edges([
            ((100, 400), (1100, 400)), ((700, 100), (700, 900)),
        ])))

    def test_short_stones_and_invalid_edges_do_not_block_a_valid_anchor(self):
        edges = local_edges([
            ((0, 30), (1000, 30)), ((30, 0), (30, 800)),
            ((100, 150), (180, 150)), ((100, 150), (100, 180)),
            ((5, 5), (5, 5)),
        ])
        edges += [Edge(np.array((math.nan, 0)), np.array((1000, 0))),
                  Edge(np.array((math.inf, 10)), np.array((math.inf, 800)))]
        self.assert_frame(infer_boundary_frame(edges), 0, (0, 0), (1000, 800))

    def test_contradictory_lengths_or_positions_prevent_an_anchor(self):
        boundary = [((0, 30), (1000, 30)), ((30, 0), (30, 800))]
        for conflicting in (
            ((0, 770), (1200, 770)),  # incompatible width
            ((400, 0), (400, 800)),  # unsupported vertical wall position
            ((0, 770), (1000, 775)),  # incompatible orientation
            ((15, 30), (1015, 30)),  # stale displaced observation
        ):
            with self.subTest(conflicting=conflicting):
                self.assertIsNone(infer_boundary_frame(local_edges(boundary + [conflicting])))

    def test_nearly_perpendicular_pair_must_fit_full_endpoints(self):
        self.assertIsNone(infer_boundary_frame(local_edges([
            ((0, 30), (1000, 30)), ((30, 0), (40, 800)),
        ])))
        self.assertIsNotNone(infer_boundary_frame(local_edges([
            ((0, 30), (1000, 30)), ((30, 0), (31, 800)),
        ])))

    def test_tolerance_is_applied_to_all_observations(self):
        edges = local_edges([
            ((0, 30), (1000, 30)), ((30, 0), (30, 800)), ((1, 770), (1001, 770)),
        ])
        self.assertIsNotNone(infer_boundary_frame(edges, tolerance=3))
        self.assertIsNone(infer_boundary_frame(edges, tolerance=0.1))

    def test_configurable_map_format(self):
        edges = local_edges([((0, 10), (160, 10)), ((150, 0), (150, 120))])
        self.assertIsNone(infer_boundary_frame(edges))
        self.assert_frame(infer_boundary_frame(edges, minimum_length=100, wall_thickness=10),
                          0, (0, 0), (160, 120))

    def test_invalid_configuration_is_rejected(self):
        for arguments in ({"minimum_length": 0}, {"minimum_length": math.inf},
                          {"wall_thickness": -1}, {"wall_thickness": math.nan},
                          {"tolerance": -1}, {"tolerance": math.inf}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                infer_boundary_frame([], **arguments)


if __name__ == "__main__":
    unittest.main()

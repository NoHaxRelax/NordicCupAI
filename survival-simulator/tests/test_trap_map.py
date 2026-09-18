import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.trap_map import TrapInferenceConfig, TrapMapper, observed_rectangles
from src.utils.controllers.world_estimator import EdgeLandmark, MapGroup
from src.utils.trap_sites import find_trap_sites
from test_expert_policy import agent_state
from test_global_planner import planner_config


def faces(x, y, width, height):
    return np.array([[(x, y), (x + width, y)], [(x + width, y), (x + width, y + height)],
                     [(x, y + height), (x + width, y + height)], [(x, y), (x, y + height)]], dtype=float)


def group_with(*rectangles, group_id=1, size=(400., 320.)):
    group = MapGroup(group_id, anchored=True, world_size=size)
    add_faces(group, np.concatenate([faces(*rect) for rect in rectangles]) if rectangles else [])
    return group


def add_faces(group, segments):
    group.edges += [EdgeLandmark(np.array(a), np.array(b), 0.) for a, b in segments]
    group._edge_revision += 1


class RectangleInferenceTests(unittest.TestCase):
    def test_requires_two_adjacent_faces_and_reports_observed_support(self):
        edges = faces(120, 80, 30, 100)
        self.assertEqual(observed_rectangles(edges[:1]), [])
        self.assertEqual(observed_rectangles(edges[[0, 2]]), [])
        for count in (2, 3, 4):
            inferred = observed_rectangles(edges[:count])
            self.assertEqual(len(inferred), 1)
            self.assertEqual(inferred[0]["rectangle"], [120, 80, 30, 100])
            self.assertEqual(inferred[0]["observed_faces"], count)

    def test_deduplicates_reversed_faces_and_handles_small_pose_error(self):
        edges = faces(120, 80, 30, 100)
        edges[1] += .1
        inferred = observed_rectangles(np.concatenate([edges, edges[:, ::-1]]))
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["observed_faces"], 4)

    def test_unmatched_and_rotated_faces_do_not_fill_in_rocks(self):
        edges = faces(120, 80, 30, 100)[:2]
        edges[1] += (10, 10)
        self.assertEqual(observed_rectangles(edges), [])
        rotation = np.array([[.8, -.6], [.6, .8]])
        self.assertEqual(observed_rectangles(faces(120, 80, 30, 100) @ rotation), [])
        self.assertEqual(observed_rectangles([[(float("nan"), 0), (10, 0)]]), [])


class TrapMapperTests(unittest.TestCase):
    def mapper(self, **kwargs):
        return TrapMapper(TrapInferenceConfig(enabled=True, **kwargs))

    def test_known_wall_gets_absolute_candidate_without_claiming_safety(self):
        mapper = self.mapper()
        group = group_with((120, 80, 30, 100))
        before = copy.deepcopy(group.edges)
        mapper.update({1: group}, 0.)
        snapshot = json.loads(json.dumps(mapper.snapshot(1), allow_nan=False))
        wall = next(site for site in snapshot["sites"] if site["kind"] == "wall")
        self.assertEqual(wall["bait"], [155.01, 130.])
        self.assertEqual(wall["predator_side"], [109.99, 130.])
        self.assertFalse(wall["guaranteed"])
        self.assertFalse(wall["approach_verified"])
        self.assertEqual(wall["source"], "observed edges")
        snapshot["sites"].clear()
        self.assertTrue(mapper.snapshot(1)["sites"])
        for old, new in zip(before, group.edges):
            np.testing.assert_array_equal(old.start, new.start)
            np.testing.assert_array_equal(old.end, new.end)

    def test_slot_and_boundary_slot_use_observed_geometry(self):
        for rocks in (((140, 80, 60, 150), (214, 80, 60, 150)), ((44, 80, 60, 150),)):
            with self.subTest(rocks=rocks):
                mapper = self.mapper()
                mapper.update({1: group_with(*rocks)}, 0.)
                slots = [row for row in mapper.snapshot(1)["sites"] if row["kind"] == "slot"]
                self.assertTrue(slots)
                self.assertTrue(all(row["geometry_guaranteed"] for row in slots))
                self.assertTrue(all(not row["guaranteed"] for row in slots))

    def test_unanchored_or_incomplete_bounds_never_use_default_map_dimensions(self):
        mapper = self.mapper()
        group = group_with((120, 80, 30, 100))
        group.anchored = False
        with patch("src.utils.controllers.trap_map.find_trap_sites") as scan:
            mapper.update({1: group}, 0.)
            self.assertFalse(mapper.snapshot(1)["sites"])
            group.anchored = True
            group.world_size = None
            group.known_width = 400
            mapper.update({1: group}, 1.)
            self.assertEqual(mapper.snapshot(1)["status"], "waiting for observed world bounds")
            scan.assert_not_called()
        group.known_height = 320
        mapper.update({1: group}, 2.)
        self.assertTrue(mapper.snapshot(1)["sites"])

    def test_unchanged_geometry_and_repeat_sightings_do_not_repeat_scans(self):
        mapper = self.mapper()
        group = group_with((120, 80, 30, 100))
        with patch("src.utils.controllers.trap_map.find_trap_sites", wraps=find_trap_sites) as scan:
            mapper.update({1: group}, 0.)
            for now in (1, 10, 100, 1000):
                group.edges[0].last_seen = now
                group.edges[0].sightings += 1
                mapper.update({1: group}, now)
            self.assertEqual(scan.call_count, 1)
            # A duplicate edge can arrive during a merge; rectangle geometry is unchanged.
            add_faces(group, faces(120, 80, 30, 100))
            mapper.update({1: group}, 1001.)
            self.assertEqual(scan.call_count, 1)

    def test_new_rock_invalidates_cached_wall_after_throttled_refresh(self):
        mapper = self.mapper()
        group = group_with((120, 80, 30, 100))
        mapper.update({1: group}, 0.)
        add_faces(group, faces(145, 80, 45, 100))
        mapper.update({1: group}, 1.)
        self.assertTrue(mapper.snapshot(1)["stale"])
        self.assertEqual(mapper.snapshot(1)["scan_count"], 1)
        mapper.update({1: group}, 10.)
        self.assertFalse(mapper.snapshot(1)["stale"])
        self.assertFalse(mapper.snapshot(1)["sites"])
        self.assertEqual(mapper.snapshot(1)["scan_count"], 2)

    def test_incomplete_new_face_blocks_bait_without_a_new_raster_scan(self):
        mapper = self.mapper()
        group = group_with((120, 80, 30, 100))
        mapper.update({1: group}, 0.)
        add_faces(group, [[(155, 120), (155, 140)]])
        mapper.update({1: group}, 10.)
        self.assertFalse(mapper.snapshot(1)["sites"])
        self.assertEqual(mapper.snapshot(1)["scan_count"], 1)

    def test_frame_change_recomputes_positions_and_removed_groups_are_pruned(self):
        mapper = self.mapper()
        group = group_with((120, 80, 30, 100))
        mapper.update({1: group}, 0.)
        original = mapper.snapshot(1)["sites"][0]["bait"]
        for edge in group.edges:
            edge.start += (50, 0)
            edge.end += (50, 0)
        group.frame_revision += 1
        group._edge_revision += 1
        mapper.update({1: group}, .1)
        moved = mapper.snapshot(1)["sites"][0]["bait"]
        np.testing.assert_allclose(moved, np.array(original) + (50, 0))
        mapper.update({}, .2)
        self.assertIsNone(mapper.snapshot(1))

    def test_disabled_and_oversize_maps_do_not_allocate_rasters(self):
        group = group_with((120, 80, 30, 100))
        with patch("src.utils.controllers.trap_map.find_trap_sites") as scan:
            disabled = TrapMapper(TrapInferenceConfig())
            disabled.update({1: group}, 0.)
            self.assertIsNone(disabled.snapshot(1))
            group.world_size = (100000., 100000.)
            mapper = self.mapper()
            mapper.update({1: group}, 0.)
            self.assertEqual(mapper.snapshot(1)["status"], "world bounds exceed scan budget")
            scan.assert_not_called()

    def test_planner_builds_traps_from_public_observations_and_resets(self):
        planner = GlobalPlanner(planner_config(enabled=False, mapping_enabled=True,
            exploration={"enabled": False}, biome_inference={"enabled": False},
            trap_inference={"enabled": True}, estimator={"anchor_boundaries": True}))
        segments = [np.array([(0, 30), (400, 30)]), np.array([(30, 0), (30, 320)])]
        segments += list(faces(120, 80, 30, 100))
        observations = [{"type": "Edge", "coords": (segment - (60, 60)).tolist()} for segment in segments]
        planner.instructions([agent_state(observations)], 0.)
        group = planner.snapshot()["groups"][0]
        self.assertTrue(group["anchored"])
        self.assertTrue(group["trap_estimate"]["sites"])
        self.assertEqual(group["trap_estimate"]["sites"][0]["bait"], (155.01, 130.))
        # Rendering/snapshot reads never run the detector a second time.
        with patch("src.utils.controllers.trap_map.find_trap_sites") as scan:
            planner.snapshot()
            planner.snapshot()
            scan.assert_not_called()
        planner.reset()
        self.assertEqual(planner.trap_mapper.layers, {})


if __name__ == "__main__":
    unittest.main()

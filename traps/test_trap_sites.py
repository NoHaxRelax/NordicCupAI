"""Run with: python -m unittest test_trap_sites -v.

Pins the trap geometry against the engine's own numbers and against the
reference scene in predator_trap.py.
"""

import math
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from trap_sites import (AGENT_RADIUS, PREDATOR_RADIUS, PREDATOR_SMELL,
                        SEPARATION_LIMIT, TOUCH_DISTANCE, TrapSite,
                        find_trap_sites)

W, H = 1600, 1200
BOUNDS = [(0, 0, W, 30), (0, H - 30, W, 30), (0, 0, 30, H), (W - 30, 0, 30, H)]


def sites(*rocks, safety="measured"):
    return find_trap_sites(BOUNDS + list(rocks), W, H, safety=safety)


class Geometry(unittest.TestCase):
    def test_matches_predator_trap_reference(self):
        """The 30-wide rock from predator_trap.py, at its 45-unit separation."""
        found = sites((800, 500, 30, 100))
        self.assertEqual(len(found), 1)
        site = found[0]
        self.assertEqual(site.axis, "x")
        self.assertAlmostEqual(site.thickness, 30.0, delta=0.05)
        # thickness + agent radius + predator radius, plus a hair of clearance.
        self.assertAlmostEqual(site.separation, 45.0, delta=0.05)
        # predator_trap.py observes the predator swinging out to 52.75.
        self.assertGreater(site.worst_case_distance, 52.0)
        self.assertLess(site.worst_case_distance, PREDATOR_SMELL)

    def test_separation_is_thickness_plus_both_radii(self):
        for thickness in (30, 33, 36):
            site = sites((800, 500, thickness, 120))[0]
            self.assertAlmostEqual(
                site.separation, thickness + AGENT_RADIUS + PREDATOR_RADIUS,
                delta=0.05)

    def test_rejects_walls_that_are_too_thick(self):
        self.assertEqual(len(sites((800, 500, 37, 120))), 1)
        self.assertEqual(len(sites((800, 500, 40, 120))), 0)
        # Anything past the separation limit is out by definition.
        self.assertEqual(
            len(sites((800, 500, SEPARATION_LIMIT, 120))), 0)

    def test_rejects_walls_that_are_too_short(self):
        self.assertEqual(len(sites((800, 500, 30, 60))), 0)
        self.assertEqual(len(sites((800, 500, 30, 70))), 1)

    def test_finds_traps_on_both_axes(self):
        self.assertEqual(sites((800, 500, 30, 100))[0].axis, "x")
        self.assertEqual(sites((800, 500, 100, 30))[0].axis, "y")
        # A square is neither thin-and-long nor short-and-fat.
        self.assertEqual(len(sites((800, 500, 100, 100))), 0)

    def test_bare_map_has_no_sites(self):
        """Boundary walls are unusable: the predator would have to stand
        outside the world."""
        self.assertEqual(len(sites()), 0)


class Overlap(unittest.TestCase):
    def test_overlap_that_thickens_a_wall_is_rejected(self):
        self.assertEqual(len(sites((800, 500, 30, 100))), 1)
        self.assertEqual(len(sites((800, 500, 30, 100), (820, 500, 30, 100))), 0)

    def test_overlap_that_lengthens_a_wall_is_kept(self):
        self.assertEqual(len(sites((800, 500, 30, 100), (800, 595, 30, 100))), 1)

    def test_rock_blocking_the_bait_position_is_rejected(self):
        self.assertEqual(len(sites((800, 500, 30, 100), (836, 500, 30, 100))), 0)

    def test_independent_rocks_are_reported_separately(self):
        self.assertEqual(len(sites((800, 500, 30, 100), (400, 300, 30, 100))), 2)


class Slots(unittest.TestCase):
    """Two rocks with a gap an agent fits into but a predator cannot."""

    @staticmethod
    def slots(*rocks, safety="measured"):
        found = find_trap_sites(BOUNDS + list(rocks), W, H, safety=safety)
        return [s for s in found if s.kind == "slot"]

    def corridor(self, gap, length=200, y=400):
        """Two rocks facing each other across `gap`."""
        return [(700.0, y, 60.0, length), (760.0 + gap, y, 60.0, length)]

    def test_gap_must_admit_the_agent(self):
        # Agent radius 5, so a gap under 10 has nowhere legal to stand.
        self.assertEqual(len(self.slots(*self.corridor(8))), 0)
        self.assertEqual(len(self.slots(*self.corridor(14))), 1)

    def test_gap_must_exclude_the_predator(self):
        # Predator radius 10: from 20 up it walks in and the slot is worthless.
        self.assertEqual(len(self.slots(*self.corridor(18))), 1)
        self.assertEqual(len(self.slots(*self.corridor(24))), 0)

    def test_corridor_must_be_deep_enough(self):
        # Clearance below TOUCH_DISTANCE means the predator reaches in.
        self.assertEqual(len(self.slots(*self.corridor(14, length=8))), 0)
        self.assertEqual(len(self.slots(*self.corridor(14, length=200))), 1)

    def test_slot_clearance_beats_touch_distance(self):
        site = self.slots(*self.corridor(14))[0]
        self.assertGreaterEqual(site.clearance, TOUCH_DISTANCE)
        self.assertTrue(site.guaranteed)
        self.assertEqual(site.worst_case_distance, site.clearance)

    def test_slot_against_the_boundary_wall(self):
        """The gap between a rock and the map edge works the same way."""
        self.assertEqual(len(self.slots((44.0, 400.0, 60.0, 200.0))), 1)

    def test_reported_gap_matches_the_geometry(self):
        for gap in (12.0, 14.0, 18.0):
            site = self.slots(*self.corridor(gap))[0]
            self.assertAlmostEqual(site.thickness, gap, delta=1.0)

    def test_slots_outrank_walls(self):
        rocks = self.corridor(14) + [(300.0, 200.0, 30.0, 120.0)]
        found = find_trap_sites(BOUNDS + rocks, W, H, safety="measured")
        self.assertEqual(found[0].kind, "slot")
        self.assertTrue(any(s.kind == "wall" for s in found))

    def test_kinds_filter(self):
        rocks = self.corridor(14) + [(300.0, 200.0, 30.0, 120.0)]
        walls = find_trap_sites(BOUNDS + rocks, W, H, safety="measured",
                                kinds=("wall",))
        only = find_trap_sites(BOUNDS + rocks, W, H, safety="measured",
                               kinds=("slot",))
        self.assertTrue(all(s.kind == "wall" for s in walls))
        self.assertTrue(all(s.kind == "slot" for s in only))

    def test_walls_are_not_marked_guaranteed(self):
        """A wall trap depends on predator AI, so it carries no guarantee."""
        wall = sites((800, 500, 30, 100))[0]
        self.assertEqual(wall.kind, "wall")
        self.assertFalse(wall.guaranteed)


class Presets(unittest.TestCase):
    def test_presets_are_progressively_stricter(self):
        rocks = [(800, 500, 36, 70), (400, 300, 30, 120), (200, 700, 32, 90)]
        counts = [len(find_trap_sites(BOUNDS + rocks, W, H, safety=s))
                  for s in ("measured", "safe", "paranoid")]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreater(counts[0], 0)

    def test_stricter_presets_never_report_more_sites(self):
        """On real map geometry, not just handcrafted rocks.

        Regression: pockets used to be labelled on the thresholded mask, so a
        tighter preset split one pocket into several components and reported the
        same place repeatedly -- making 'paranoid' look richer than 'safe'.
        """
        from trap_sites import _sampled_map
        for seed in range(6):
            rects = _sampled_map(seed, W, H)
            counts = [len(find_trap_sites(rects, W, H, safety=s))
                      for s in ("measured", "safe", "paranoid")]
            self.assertEqual(counts, sorted(counts, reverse=True),
                             f"seed {seed}: {counts}")

    def test_unknown_preset_rejected(self):
        with self.assertRaises(ValueError):
            find_trap_sites(BOUNDS, W, H, safety="reckless")

    def test_results_are_sorted_by_score(self):
        found = sites((800, 500, 30, 200), (400, 300, 36, 70))
        self.assertEqual([s.score for s in found],
                         sorted((s.score for s in found), reverse=True))


class Interop(unittest.TestCase):
    def test_accepts_obstacle_objects_and_empty_input(self):
        """env.obstacles can be passed straight in, not just tuples."""
        from _bootstrap import ensure_on_path
        if ensure_on_path() is None:
            self.skipTest("not inside a survival-simulator checkout")
        from src.elements.obstacle import Obstacle
        objs = [Obstacle(x, y, w, h) for (x, y, w, h) in BOUNDS]
        objs.append(Obstacle(800, 500, 30, 100))
        found = find_trap_sites(objs, W, H, safety="measured")
        self.assertEqual(len(found), 1)
        self.assertEqual(find_trap_sites([], W, H), [])

    def test_site_fields_are_self_consistent(self):
        site = sites((800, 500, 30, 100))[0]
        self.assertIsInstance(site, TrapSite)
        dx = site.bait[0] - site.predator_side[0]
        dy = site.bait[1] - site.predator_side[1]
        self.assertAlmostEqual(math.hypot(dx, dy), site.separation, places=2)
        self.assertIn("worst_case_distance", site.as_dict())

    def test_limit_truncates(self):
        rocks = [(800, 500, 30, 100), (400, 300, 30, 100), (200, 700, 30, 100)]
        self.assertEqual(
            len(find_trap_sites(BOUNDS + rocks, W, H, safety="measured", limit=2)), 2)


if __name__ == "__main__":
    unittest.main()

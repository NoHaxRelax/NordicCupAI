import unittest
from unittest.mock import patch

import numpy as np

from src.utils.controllers.navigation import Navigator
from src.utils.controllers.world_estimator import EdgeLandmark, MapGroup


def edge(a, b):
    return EdgeLandmark(np.array(a, dtype=float), np.array(b, dtype=float), 0.)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.group = MapGroup(1, anchored=True, world_size=(400., 400.))
        self.navigator = Navigator()

    def navigate(self, start, goal, walls):
        self.group.edges = walls
        self.navigator.update(self.group, 0.)
        result = self.navigator.steer(1, start, goal, 0.)
        self.assertFalse(result.blocked)
        path = [np.array(start)] + self.navigator.routes[1].points
        self.assertTrue(all(self.navigator._clear(a, b) for a, b in zip(path, path[1:])))
        self.assertTrue(np.allclose(path[-1], goal))
        return result, path

    def test_routes_around_known_rectangle(self):
        corners = [(150, 100), (250, 100), (250, 300), (150, 300)]
        result, path = self.navigate((80, 200), (320, 200),
                                    [edge(a, b) for a, b in zip(corners, corners[1:] + corners[:1])])
        self.assertGreater(result.remaining, 300.)
        self.assertTrue(any(p[1] < 100 or p[1] > 300 for p in path))

    def test_exits_u_barrier_before_heading_to_goal(self):
        result, path = self.navigate((200, 200), (200, 80),
                                    [edge((140, 100), (260, 100)),
                                     edge((140, 100), (140, 300)),
                                     edge((260, 100), (260, 300))])
        self.assertGreater(result.waypoint[1], 300)
        self.assertTrue(any(p[1] > 300 for p in path))

    def test_stationary_agent_replans_once_then_reports_blocked(self):
        self.navigator.update(self.group, 0.)
        first = self.navigator.steer(1, (80, 200), (320, 200), 0.)
        self.assertEqual(first.status, "following")
        self.assertEqual(self.navigator.steer(1, (80, 200), (320, 200), 5.).status, "replanned")
        result = self.navigator.steer(1, (80, 200), (320, 200), 10.)
        self.assertTrue(result.blocked)
        self.assertIsNone(result.waypoint)

    def test_sideways_oscillation_does_not_count_as_progress(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        for now in range(1, 12):
            result = self.navigator.steer(1, (80, 200 + (-1) ** now * 15), (320, 200), now)
        self.assertTrue(result.blocked)

    def test_real_route_progress_keeps_agent_moving(self):
        self.navigator.update(self.group, 0.)
        for now in range(20):
            result = self.navigator.steer(1, (80 + now * 5, 200), (320, 200), now)
            self.assertFalse(result.blocked)
        self.assertEqual(self.navigator.snapshot()[1]["replans"], 0)

    def test_intentional_waiting_does_not_trigger_recovery(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        for now in range(1, 30):
            result = self.navigator.steer(1, (80, 200), (320, 200), now, waiting=True)
            self.assertEqual(result.status, "waiting")
            self.assertFalse(result.blocked)
        self.assertEqual(self.navigator.steer(1, (80, 200), (320, 200), 30.).status, "following")
        self.assertEqual(self.navigator.snapshot()[1]["replans"], 0)

    def test_new_wall_invalidates_only_obstructed_route(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        self.navigator.steer(2, (80, 40), (320, 40), 0.)
        stable_route = self.navigator.routes[2].points
        self.group.edges.append(edge((200, 100), (200, 300)))
        self.navigator.update(self.group, 1.)
        result = self.navigator.steer(1, (80, 200), (320, 200), 1.)
        self.navigator.steer(2, (80, 40), (320, 40), 1.)
        self.assertFalse(result.blocked)
        self.assertFalse(np.allclose(result.waypoint, (320, 200)))
        self.assertIs(self.navigator.routes[2].points, stable_route)

    def test_frame_changes_discard_old_routes(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        self.group.frame_revision += 1
        self.navigator.update(self.group, 1.)
        self.assertEqual(self.navigator.snapshot(), {})

    def test_repeated_geometry_invalidation_cannot_hide_stationary_agent(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        # Isolate the recovery clock from route geometry: every revision
        # invalidates the old route, and a replacement route remains possible.
        with patch.object(self.navigator, "_clear", return_value=False), \
                patch.object(self.navigator, "_plan", return_value=[np.array([320., 200.])]):
            for now in range(1, 12):
                self.navigator.revision += 1
                result = self.navigator.steer(1, (80, 200), (320, 200), now)
        self.assertTrue(result.blocked)
        self.assertEqual(self.navigator.snapshot()[1]["replans"], 1)

    def test_released_failed_goal_can_be_retried_after_cooldown(self):
        self.navigator.update(self.group, 0.)
        for now in (0., 5., 10.):
            result = self.navigator.steer(1, (80, 200), (320, 200), now)
        self.assertTrue(result.blocked)
        self.navigator.release(1)
        result = self.navigator.steer(1, (80, 200), (320, 200), 40.)
        self.assertFalse(result.blocked)
        self.assertEqual(self.navigator.snapshot()[1]["replans"], 0)

    def test_small_destination_noise_keeps_route_and_stall_timer(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        initial = self.navigator.routes[1]
        result = self.navigator.steer(1, (80, 200), (322, 200), 5.)
        self.assertIs(self.navigator.routes[1], initial)
        self.assertEqual(result.status, "replanned")

    def test_agent_can_escape_inflated_wall_padding(self):
        result, path = self.navigate((195, 200), (300, 200), [edge((200, 100), (200, 300))])
        self.assertLess(result.waypoint[0], 200)
        self.assertTrue(any(p[1] < 100 or p[1] > 300 for p in path))

    def test_partial_bounds_route_uses_only_observed_envelope(self):
        self.group.world_size = None
        self.group.known_width = 400.
        result, _ = self.navigate((80, 200), (320, 200), [edge((200, 100), (200, 300))])
        self.assertFalse(result.blocked)
        self.assertEqual(self.navigator.origin[0], 0.)
        self.assertEqual(self.navigator.extent[0], 400.)

    def test_unreachable_goal_is_blocked_and_dead_agents_pruned(self):
        self.group.edges = [edge((200, 0), (200, 400))]
        self.navigator.update(self.group, 0.)
        self.assertTrue(self.navigator.steer(1, (80, 200), (320, 200), 0.).blocked)
        self.navigator.prune([])
        self.assertFalse(self.navigator.snapshot())

    def test_grid_memory_is_bounded(self):
        self.group.world_size = (10000., 10000.)
        self.navigator = Navigator(max_cells=1000)
        self.navigator.update(self.group, 0.)
        self.navigator._grid(np.array([100., 100.]), np.array([9000., 9000.]))
        self.assertLessEqual(self.navigator.blocked.size, 1000)


if __name__ == "__main__":
    unittest.main()

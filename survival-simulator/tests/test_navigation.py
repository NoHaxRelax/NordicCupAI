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

    def test_twenty_pixel_opening_remains_usable_between_coarse_grid_rows(self):
        for center in (200., 204., 207.):
            with self.subTest(center=center):
                self.navigator = Navigator()
                result, path = self.navigate((80., 180.), (320., 180.),
                    [edge((200., 0.), (200., center - 10.)),
                     edge((200., center + 10.), (200., 400.))])
                self.assertLess(result.remaining, 270.)
                self.assertTrue(any(center - 10. < p[1] < center + 10. for p in path))

    def test_grid_itself_keeps_physical_clearance_without_erasing_a_gap(self):
        self.group.edges = [edge((200., 0.), (200., 190.)), edge((200., 210.), (200., 400.))]
        self.navigator.update(self.group, 0.)
        with patch.object(self.navigator, "_short_passage", return_value=[]):
            result = self.navigator.steer(1, (80., 180.), (320., 180.), 0.)
        self.assertFalse(result.blocked)
        self.assertLess(result.remaining, 280.)
        path = [np.array([80., 180.])] + self.navigator.routes[1].points
        self.assertTrue(all(self.navigator._clear(a, b) for a, b in zip(path, path[1:])))

    def test_narrow_shortcut_avoids_a_long_route_around_the_outer_wall_ends(self):
        result, _ = self.navigate((80., 180.), (320., 180.),
            [edge((200., 40.), (200., 194.)), edge((200., 214.), (200., 360.))])
        self.assertLess(result.remaining, 270.)

    def test_grid_connections_cannot_cross_a_thin_wall_between_free_nodes(self):
        self.group.edges = [edge((208., 0.), (208., 400.))]
        self.navigator.update(self.group, 0.)
        with patch.object(self.navigator, "_short_passage", return_value=[]):
            result = self.navigator.steer(1, (80., 200.), (320., 200.), 0.)
        self.assertTrue(result.blocked)
        # Both adjacent samples are clear; their connecting segment is not.
        self.assertFalse(self.navigator.blocked[12, 12])
        self.assertFalse(self.navigator.blocked[12, 13])
        self.assertFalse(self.navigator._connection_clear((12, 12), (12, 13)))

    def test_grid_connections_cannot_cut_a_corner_between_free_nodes(self):
        self.group.edges = [edge((208., 208.), (240., 208.)),
                            edge((208., 208.), (208., 240.))]
        self.navigator.update(self.group, 0.)
        self.navigator._grid(np.array([200., 200.]), np.array([216., 216.]))
        self.assertFalse(self.navigator.blocked[12, 12])
        self.assertFalse(self.navigator.blocked[13, 13])
        self.assertFalse(self.navigator._connection_clear((12, 12), (13, 13)))

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

    def test_deliberate_pause_preserves_route_without_spending_stall_budget(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 3.)
        route = self.navigator.routes[1]
        points, best = route.points, route.best_remaining
        self.navigator.pause(1, 3.)
        self.navigator.pause(1, 10.)
        result = self.navigator.steer(1, (80, 200), (320, 200), 13.)
        self.assertEqual(result.status, "following")
        self.assertIs(route.points, points)
        self.assertEqual(route.best_remaining, best)
        self.assertEqual(route.retries, 0)
        self.assertEqual(route.progress_at, 10.)

    def test_pause_preserves_retries_and_existing_active_stall_time(self):
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 0.)
        self.navigator.steer(1, (80, 200), (320, 200), 5.)
        self.navigator.pause(1, 6.)
        self.navigator.pause(1, 15.)
        result = self.navigator.steer(1, (80, 200), (320, 200), 16.)
        self.assertFalse(result.blocked)
        self.assertEqual(self.navigator.routes[1].retries, 1)
        self.assertTrue(self.navigator.steer(1, (80, 200), (320, 200), 20.).blocked)

    def test_short_motion_windows_still_detect_a_stuck_route_across_long_rests(self):
        self.navigator.update(self.group, 0.)
        blocked = False
        for tick in range(400):
            now = tick / 10.
            if tick % 100 < 30:
                result = self.navigator.steer(1, (80, 200), (320, 200), now)
                blocked |= result.blocked
            else:
                self.navigator.pause(1, now)
        self.assertTrue(blocked)
        self.assertEqual(self.navigator.routes[1].retries, 1)

    def test_pause_cannot_revive_a_blocked_route_or_create_a_missing_one(self):
        self.navigator.pause(42, 100.)
        self.assertNotIn(42, self.navigator.routes)
        self.navigator.update(self.group, 0.)
        for now in (0., 5., 10.):
            self.navigator.steer(1, (80, 200), (320, 200), now)
        self.navigator.pause(1, 11.)
        self.assertTrue(self.navigator.steer(1, (80, 200), (320, 200), 30.).blocked)

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

    def test_fruit_beside_wall_is_reachable_without_entering_wall_padding(self):
        self.group.edges = [edge((150., 100.), (150., 300.))]
        self.navigator.update(self.group, 0.)
        start, fruit = np.array([80., 200.]), np.array([145., 200.])
        self.assertTrue(self.navigator.steer(1, start, fruit, 0.).blocked)
        result = self.navigator.steer(2, start, fruit, 0., arrival_radius=8.)
        self.assertFalse(result.blocked)
        self.assertLessEqual(np.linalg.norm(result.waypoint - fruit), 8.)
        self.assertLess(result.waypoint[0], 150. - self.navigator.clearance)
        self.assertTrue(self.navigator._clear(start, result.waypoint))
        arrived = self.navigator.steer(2, result.waypoint, fruit, 1., arrival_radius=8.)
        self.assertEqual(arrived.status, "arrived")
        self.assertIsNone(arrived.waypoint)
        self.assertEqual(arrived.remaining, 0.)

    def test_pickup_radius_does_not_authorize_collecting_through_a_wall(self):
        self.group.edges = [edge((150., 0.), (150., 400.))]
        self.navigator.update(self.group, 0.)
        result = self.navigator.steer(1, (147., 200.), (153., 200.), 0., arrival_radius=8.)
        self.assertTrue(result.blocked)
        self.assertNotEqual(result.status, "arrived")

    def test_collection_route_keeps_clearance_while_detouring_around_a_rock(self):
        corners = [(150, 100), (250, 100), (250, 300), (150, 300)]
        self.group.edges = [edge(a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
        self.navigator.update(self.group, 0.)
        start, fruit = np.array([80., 200.]), np.array([255., 200.])
        result = self.navigator.steer(1, start, fruit, 0., arrival_radius=8.)
        self.assertFalse(result.blocked)
        path = [start] + self.navigator.routes[1].points
        self.assertTrue(all(self.navigator._clear(a, b) for a, b in zip(path, path[1:])))
        self.assertLessEqual(np.linalg.norm(path[-1] - fruit), 8. + 1e-9)
        self.assertGreater(path[-1][0], 250.)
        self.assertTrue(any(p[1] < 100. or p[1] > 300. for p in path))

    def test_ordinary_destinations_keep_their_exact_endpoint(self):
        self.navigator.update(self.group, 0.)
        result = self.navigator.steer(1, (80., 200.), (145., 200.), 0.)
        np.testing.assert_array_equal(result.waypoint, (145., 200.))
        self.assertEqual(result.remaining, 65.)

    def test_refined_fruit_position_updates_pickup_point_without_resetting_stall_clock(self):
        self.navigator.update(self.group, 0.)
        first = self.navigator.steer(1, (80., 200.), (145., 200.), 0., arrival_radius=8.)
        route = self.navigator.routes[1]
        refined = self.navigator.steer(1, first.waypoint, (148., 200.), 1., arrival_radius=8.)
        self.assertIs(self.navigator.routes[1], route)
        self.assertNotEqual(refined.status, "arrived")
        self.assertGreater(refined.waypoint[0], first.waypoint[0])
        self.assertLessEqual(np.linalg.norm(refined.waypoint - (148., 200.)), 8.)
        # Repeated coordinate refinements cannot keep a stationary agent's
        # route alive indefinitely.
        for now in range(2, 13):
            result = self.navigator.steer(1, first.waypoint, (148. + .1 * (now % 2), 200.),
                                          float(now), arrival_radius=8.)
        self.assertTrue(result.blocked)

    def test_invalid_arrival_radius_is_rejected(self):
        self.navigator.update(self.group, 0.)
        for radius in (-1., float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.navigator.steer(1, (80., 200.), (145., 200.), 0., arrival_radius=radius)

    def test_radius_jitter_cannot_reset_a_stationary_agents_stall_recovery(self):
        self.group.edges = [edge((200., 100.), (200., 300.))]
        self.navigator.update(self.group, 0.)
        self.navigator.steer(1, (80., 200.), (320., 200.), 0., arrival_radius=7.)
        initial = self.navigator.routes[1]
        for now in range(1, 22):
            result = self.navigator.steer(1, (80., 200.), (320., 200.), float(now),
                                          arrival_radius=7. - .001 * (now % 2))
            self.assertIs(self.navigator.routes[1], initial)
        self.assertTrue(result.blocked)
        self.assertEqual(initial.retries, 1)
        self.assertLessEqual(initial.progress_at, 5.)

    def test_large_radius_changes_do_not_count_as_agent_motion(self):
        self.navigator.update(self.group, 0.)
        for now in range(12):
            result = self.navigator.steer(1, (80., 200.), (320., 200.), float(now),
                                          arrival_radius=4. if now % 2 else 8.)
        self.assertTrue(result.blocked)
        self.assertEqual(self.navigator.routes[1].retries, 1)

    def test_actual_motion_still_counts_as_progress_while_pickup_radius_changes(self):
        self.navigator.update(self.group, 0.)
        for now in range(20):
            result = self.navigator.steer(1, (80. + 5 * now, 200.), (320., 200.), float(now),
                                          arrival_radius=7. - .001 * (now % 2))
            self.assertFalse(result.blocked)
        self.assertEqual(self.navigator.routes[1].retries, 0)
        self.assertGreater(self.navigator.routes[1].progress_at, 15.)

    def test_smaller_pickup_radius_rechecks_wall_clearance_without_resetting_route(self):
        self.group.edges = [edge((150., 100.), (150., 300.))]
        self.navigator.update(self.group, 0.)
        first = self.navigator.steer(1, (80., 200.), (147., 200.), 0., arrival_radius=8.)
        initial = self.navigator.routes[1]
        self.assertFalse(first.blocked)
        closer = self.navigator.steer(1, (80., 200.), (147., 200.), 1., arrival_radius=4.)
        self.assertIs(self.navigator.routes[1], initial)
        self.assertFalse(closer.blocked)
        self.assertLessEqual(np.linalg.norm(closer.waypoint - (147., 200.)), 4.)
        self.assertTrue(self.navigator._clear(np.array([80., 200.]), closer.waypoint))
        # A one-pixel pickup radius cannot reach this fruit while maintaining
        # the agent's six-pixel clearance. The former safe endpoint is invalid.
        impossible = self.navigator.steer(1, (80., 200.), (147., 200.), 2., arrival_radius=1.)
        self.assertIs(self.navigator.routes[1], initial)
        self.assertTrue(impossible.blocked)
        self.assertIsNone(impossible.waypoint)

    def test_grid_memory_is_bounded(self):
        self.group.world_size = (10000., 10000.)
        self.navigator = Navigator(max_cells=1000)
        self.navigator.update(self.group, 0.)
        self.navigator._grid(np.array([100., 100.]), np.array([9000., 9000.]))
        self.assertLessEqual(self.navigator.blocked.size, 1000)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import numpy as np

from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.world_estimator import EdgeLandmark
from test_expert_policy import agent_state
from test_exploration import accelerated_config, connected_states


def rapid_planner(**changes):
    rapid = dict(enabled=True, sprint_enabled=True)
    rapid.update(changes)
    return GlobalPlanner(accelerated_config(rapid_mapping=rapid))


class RapidExplorationTests(unittest.TestCase):
    def moving_scout(self, planner, energy=250):
        states = [agent_state(agent_id=0, energy=energy)]
        planner.instructions(states, 0)
        pose = planner.estimator.poses[0]
        pose.position = np.array([10., 0.])
        hints = planner.exploration.instructions(states, planner.estimator, 0.1)
        return states, pose, hints[0]

    def test_default_fixture_preserves_walking_and_no_camera_override(self):
        planner = GlobalPlanner(accelerated_config())
        planner.instructions([agent_state(agent_id=0, energy=500)], 0)
        hint = planner.exploration_hints[0]
        self.assertIsNone(hint.max_speed)
        self.assertIsNone(hint.look_direction)
        self.assertLessEqual(math.hypot(*hint.vector), 10)

    def test_preanchor_scout_keeps_straight_route_while_camera_sweeps(self):
        planner = rapid_planner(sprint_enabled=False)
        states = [agent_state(agent_id=0)]
        planner.instructions(states, 0)
        pose = planner.estimator.poses[0]
        duty = planner.exploration.duties[0]
        original_heading = duty.heading
        first = planner.exploration.instructions(states, planner.estimator, 0.5)[0]
        np.testing.assert_allclose(first.vector, [8.5, 0], atol=1e-9)
        self.assertAlmostEqual(first.look_direction, math.pi / 3)
        # Apply only a camera turn; world-space travel must stay straight.
        pose.heading = math.pi / 3
        second = planner.exploration.instructions(states, planner.estimator, 1.5)[0]
        self.assertAlmostEqual(duty.heading, original_heading)
        local_direction = math.atan2(second.vector[1], second.vector[0])
        self.assertAlmostEqual(local_direction + pose.heading, original_heading)
        self.assertAlmostEqual(second.look_direction + pose.heading, -math.pi / 3)
        self.assertIsNone(duty.target)

    def test_sprint_starts_only_after_progress_and_has_deterministic_burst_and_cooldown(self):
        planner = rapid_planner(scan_enabled=False)
        states = [agent_state(agent_id=0, energy=250)]
        planner.instructions(states, 0)
        self.assertIsNone(planner.exploration_hints[0].max_speed)
        pose = planner.estimator.poses[0]
        pose.position = np.array([10., 0.])
        first = planner.exploration.instructions(states, planner.estimator, 0.1)[0]
        self.assertEqual(first.max_speed, 20)
        self.assertEqual(math.hypot(*first.vector), 20)
        self.assertEqual(first.minimum_energy_reserve, 100)
        self.assertEqual(planner.exploration.instructions(states, planner.estimator, 0.1)[0], first)
        for now, expected in ((0.4, True), (0.6, False), (1.0, False), (2.6, True)):
            pose.position += np.array([10., 0.])
            hint = planner.exploration.instructions(states, planner.estimator, now)[0]
            self.assertEqual(hint.max_speed is not None, expected)

    def test_sprint_stops_if_progress_stalls_or_camera_loses_route(self):
        planner = rapid_planner()
        states, pose, first = self.moving_scout(planner)
        self.assertEqual(first.max_speed, 20)
        self.assertAlmostEqual(first.look_direction, 0)
        stationary = planner.exploration.instructions(states, planner.estimator, 0.7)[0]
        self.assertIsNone(stationary.max_speed)
        pose.position += np.array([10., 0.])
        pose.heading = math.pi / 2
        off_camera = planner.exploration.instructions(states, planner.estimator, 3.0)[0]
        self.assertIsNone(off_camera.max_speed)

    def test_newborns_and_low_energy_agents_do_not_receive_sprint_requests(self):
        for energy in (75, 99, 149, 159):
            with self.subTest(energy=energy):
                planner = rapid_planner()
                _, _, hint = self.moving_scout(planner, energy)
                self.assertIsNone(hint.max_speed)
                self.assertLessEqual(math.hypot(*hint.vector), 10)

    def test_wall_in_sprint_path_blocks_burst_and_speed_checks_do_not_overshoot(self):
        planner = rapid_planner(scan_enabled=False)
        states = [agent_state(agent_id=0, energy=250)]
        planner.instructions(states, 0)
        pose = planner.estimator.poses[0]
        pose.position = np.array([10., 0.])
        blocked = [agent_state([{"type": "Edge", "coords": [[15, -100], [15, 100]]}],
                               agent_id=0, energy=250)]
        hint = planner.exploration.instructions(blocked, planner.estimator, 0.1)[0]
        self.assertIsNone(hint.max_speed)
        self.assertLessEqual(math.hypot(*hint.vector), 10)
        duty = planner.exploration.duties[0]
        self.assertIsNone(planner.exploration._sprint_speed(
            states[0], pose, duty, planner.estimator.groups[0], 3, 0,
            pose.position + np.array([12., 0.]), (),
        ))

    def test_anchored_frontier_prefers_long_forward_route_and_keeps_goal_on_cell_entry(self):
        planner = rapid_planner(sprint_enabled=False)
        states = [agent_state(agent_id=0)]
        planner.instructions(states, 0)
        group, pose = planner.estimator.groups[0], planner.estimator.poses[0]
        group.anchored, group.world_size = True, (1000., 1000.)
        pose.position = np.array([60., 60.])
        group.visited = {(0, 0): (pose.position.copy(), 0.05, 0)}
        planner.exploration.instructions(states, planner.estimator, 1)
        duty = planner.exploration.duties[0]
        np.testing.assert_allclose(duty.target, [300., 60.])
        pose.position = np.array([260., 60.])
        group.visited[(2, 0)] = (pose.position.copy(), 0.05, 3)
        planner.exploration.instructions(states, planner.estimator, 3)
        np.testing.assert_allclose(duty.target, [300., 60.])

    def test_long_frontier_rejects_known_wall_and_high_coverage_stops_sprinting(self):
        planner = rapid_planner()
        states = [agent_state(agent_id=0, energy=250)]
        planner.instructions(states, 0)
        group, pose = planner.estimator.groups[0], planner.estimator.poses[0]
        group.anchored, group.world_size = True, (1000., 1000.)
        pose.position = np.array([60., 60.])
        group.visited = {(0, 0): (pose.position.copy(), 0.05, 0)}
        group.edges = [EdgeLandmark(np.array([180., 0.]), np.array([180., 1000.]), 0)]
        planner.exploration.instructions(states, planner.estimator, 1)
        target = planner.exploration.duties[0].target
        self.assertLess(target[0], 180)
        plan = planner.exploration.frontiers[0]
        plan.coverage = 0.9
        pose.position += np.array([10., 0.])
        hint = planner.exploration.instructions(states, planner.estimator, 1.1)[0]
        self.assertIsNone(hint.max_speed)

    def test_newly_observed_wall_replans_existing_long_goal_before_timeout(self):
        planner = rapid_planner(sprint_enabled=False)
        states = [agent_state(agent_id=0)]
        planner.instructions(states, 0)
        group, pose = planner.estimator.groups[0], planner.estimator.poses[0]
        group.anchored, group.world_size = True, (1000., 1000.)
        pose.position = np.array([60., 60.])
        group.visited = {(0, 0): (pose.position.copy(), 0.05, 0)}
        planner.exploration.instructions(states, planner.estimator, 1)
        duty = planner.exploration.duties[0]
        previous = duty.target.copy()
        self.assertGreater(previous[0], 180)
        group.edges = [EdgeLandmark(np.array([180., 0.]), np.array([180., 1000.]), 2)]
        planner.exploration.instructions(states, planner.estimator, 3)
        self.assertTrue(duty.target is None or duty.target[0] < 180)
        self.assertIn(tuple(previous), duty.failed_targets)

    def test_anchor_ends_rapid_burst_scan_and_long_targets_and_restores_normal_invalidation(self):
        planner = rapid_planner(only_until_anchored=True)
        states, pose, sprint_hint = self.moving_scout(planner)
        self.assertEqual(sprint_hint.max_speed, 20)
        group = planner.estimator.groups[0]
        self.assertTrue(planner.exploration.rapid_active(group))
        group.anchored, group.world_size = True, (1000., 1000.)
        pose.position = np.array([60., 60.])
        group.visited = {(0, 0): (pose.position.copy(), 0.05, 0.2)}
        hint = planner.exploration.instructions(states, planner.estimator, 0.2)[0]
        duty = planner.exploration.duties[0]
        self.assertFalse(planner.exploration.rapid_active(group))
        self.assertFalse(duty.sprint_requested)
        self.assertIsNone(hint.max_speed)
        self.assertIsNone(hint.look_direction)
        self.assertLessEqual(math.hypot(*hint.vector), 10)
        self.assertFalse(planner.exploration.frontiers[0].rapid_mode)
        self.assertEqual(set(map(tuple, planner.exploration.frontiers[0].cells)), {(0, 1), (1, 0)})
        np.testing.assert_allclose(duty.target, [180., 60.])
        # Normal mapping reassigns a goal when its cell has been explored.
        pose.position = np.array([130., 60.])
        group.visited[(1, 0)] = (pose.position.copy(), 0.05, 2.3)
        planner.exploration.instructions(states, planner.estimator, 2.3)
        self.assertFalse(np.array_equal(duty.target, [180., 60.]))

    def test_rapid_phase_is_per_group_and_other_unanchored_groups_keep_scanning(self):
        planner = rapid_planner(only_until_anchored=True, sprint_enabled=False)
        states = [agent_state(agent_id=0), agent_state(agent_id=1)]
        planner.instructions(states, 0)
        anchored = planner.estimator.groups[0]
        anchored.anchored, anchored.world_size = True, (1000., 1000.)
        hints = planner.exploration.instructions(states, planner.estimator, 0.5)
        self.assertIsNone(hints[0].look_direction)
        self.assertIsNotNone(hints[1].look_direction)
        self.assertFalse(planner.exploration.rapid_active(anchored))
        self.assertTrue(planner.exploration.rapid_active(planner.estimator.groups[1]))


class DirectedForagingTests(unittest.TestCase):
    def test_only_healthy_reliable_scouts_limit_food_detours(self):
        planner = GlobalPlanner(accelerated_config(directed_foraging_enabled=True))
        states = [agent_state(agent_id=0, energy=130)]
        planner.instructions(states, 0)
        self.assertEqual(planner.exploration_hints[0].food_distance_limit, 25)
        for energy, capacity, uncertainty in ((129, 500, 0), (200, 1000, 0), (75, 500, 0), (150, 500, 100)):
            with self.subTest(energy=energy, capacity=capacity, uncertainty=uncertainty):
                planner.estimator.poses[0].uncertainty = uncertainty
                state = agent_state(agent_id=0, energy=energy, max_energy=capacity)
                hint = planner.exploration.instructions([state], planner.estimator, 1)[0]
                self.assertIsNone(hint.food_distance_limit)

    def test_residents_keep_full_food_priority_and_defaults_do_not_limit_food(self):
        planner = GlobalPlanner(accelerated_config(directed_foraging_enabled=True))
        planner.instructions(connected_states(), 0)
        for agent_id, duty in planner.exploration.duties.items():
            limit = planner.exploration_hints[agent_id].food_distance_limit
            self.assertEqual(limit, 25 if duty.role == "scout" else None)
        baseline = rapid_planner()
        baseline.instructions([agent_state(agent_id=0, energy=250)], 0)
        self.assertIsNone(baseline.exploration_hints[0].food_distance_limit)

    def test_nearly_complete_anchored_map_restores_full_food_priority(self):
        planner = GlobalPlanner(accelerated_config(directed_foraging_enabled=True))
        states = [agent_state(agent_id=0, energy=250)]
        planner.instructions(states, 0)
        group = planner.estimator.groups[0]
        group.anchored, group.world_size = True, (600., 600.)
        planner.exploration.instructions(states, planner.estimator, 1)
        frontier = planner.exploration.frontiers[0]
        frontier.coverage = 0.899
        self.assertEqual(planner.exploration.instructions(states, planner.estimator, 1.1)[0].food_distance_limit, 25)
        frontier.coverage = 0.9
        self.assertIsNone(planner.exploration.instructions(states, planner.estimator, 1.2)[0].food_distance_limit)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import numpy as np

from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.population import TraitRating
from src.utils.controllers.world_estimator import EdgeLandmark, TreeLandmark, rotate
from test_expert_policy import agent_state
from test_exploration import accelerated_config, connected_states, ratings


def boundary_planner(**changes):
    config = dict(boundary_seek_enabled=True, rapid_mapping={"enabled": True, "sprint_enabled": False})
    config.update(changes)
    return GlobalPlanner(accelerated_config(**config))


class BoundarySeekingTests(unittest.TestCase):
    def test_nearer_corner_uses_inward_clearance_in_rotated_local_frames(self):
        for angle in (0.0, 0.7, -1.4):
            with self.subTest(angle=angle):
                planner = boundary_planner()
                states = [agent_state(agent_id=0, energy=250)]
                planner.instructions(states, 0)
                pose, group = planner.estimator.poses[0], planner.estimator.groups[0]
                offset = np.array([-200., 500.])
                transform = lambda point: rotate(point, angle) + offset
                pose.position, pose.heading = transform((80., 140.)), angle
                group.edges = [EdgeLandmark(transform((0., 30.)), transform((1000., 30.)), 0)]
                hints = planner.exploration.instructions(states, planner.estimator, 1)
                task = planner.exploration.boundary_tasks[0]
                np.testing.assert_allclose(task.target, transform((40., 70.)), atol=1e-8)
                local = np.asarray(hints[0].vector)
                np.testing.assert_allclose(local / np.linalg.norm(local), np.array([-40., -70.]) / math.hypot(40, 70), atol=1e-8)
                self.assertEqual(hints[0].objective, "find perpendicular boundary at nearer corner")

    def test_one_existing_healthy_scout_is_assigned_and_assignment_stays_stable(self):
        planner = boundary_planner()
        states = connected_states()
        planner.instructions(states, 0)
        group = planner.estimator.groups[1]
        group.edges = [EdgeLandmark(np.array([-100., -100.]), np.array([900., -100.]), 0)]
        original_roles = {key: duty.role for key, duty in planner.exploration.duties.items()}
        hints = planner.exploration.instructions(states, planner.estimator, 1)
        chosen = [key for key, hint in hints.items() if hint.objective.startswith("find perpendicular")]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(original_roles[chosen[0]], "scout")
        self.assertEqual({key: duty.role for key, duty in planner.exploration.duties.items()}, original_roles)
        task = planner.exploration.boundary_tasks[1]
        planner.exploration.instructions(states, planner.estimator, 1.1)
        self.assertIs(planner.exploration.boundary_tasks[1], task)

    def test_anchor_immediately_removes_corner_task_and_snapshot_target(self):
        planner = boundary_planner()
        states = [agent_state(agent_id=0, energy=250)]
        planner.instructions(states, 0)
        group, pose = planner.estimator.groups[0], planner.estimator.poses[0]
        pose.position = np.array([80., 140.])
        group.edges = [EdgeLandmark(np.array([0., 30.]), np.array([1000., 30.]), 0)]
        planner.exploration.instructions(states, planner.estimator, 1)
        self.assertIn(0, planner.exploration.boundary_tasks)
        group.anchored, group.world_size = True, (1000., 1000.)
        hints = planner.exploration.instructions(states, planner.estimator, 1.1)
        self.assertNotIn(0, planner.exploration.boundary_tasks)
        self.assertIsNone(planner.exploration.snapshot()[0]["boundary_target"])
        self.assertNotEqual(hints[0].objective, "find perpendicular boundary at nearer corner")

    def test_hungry_or_uncertain_scout_does_not_get_corner_task_and_short_stone_is_ignored(self):
        planner = boundary_planner()
        states = [agent_state(agent_id=0, energy=250)]
        planner.instructions(states, 0)
        group, pose = planner.estimator.groups[0], planner.estimator.poses[0]
        pose.position = np.array([80., 140.])
        group.edges = [EdgeLandmark(np.array([0., 30.]), np.array([1000., 30.]), 0)]
        group.trees = [TreeLandmark(np.array([80., 170.]), 0)]
        hungry = [agent_state(agent_id=0, energy=60)]
        hints = planner.exploration.instructions(hungry, planner.estimator, 1)
        self.assertEqual(planner.exploration.boundary_tasks, {})
        self.assertEqual(hints[0].objective, "find food near remembered tree")
        pose.uncertainty = 100
        planner.exploration.instructions(states, planner.estimator, 2)
        self.assertEqual(planner.exploration.boundary_tasks, {})
        pose.uncertainty = 0
        group.edges = [EdgeLandmark(np.array([0., 30.]), np.array([180., 30.]), 0)]
        planner.exploration.instructions(states, planner.estimator, 3)
        self.assertEqual(planner.exploration.boundary_tasks, {})


class PopulationRoleTests(unittest.TestCase):
    def test_population_phase_caps_scouts_and_keeps_residents_existing_food_homes(self):
        planner = boundary_planner()
        states = connected_states()
        values = ratings()
        planner.instructions(states, 0, trait_ratings=values)
        resident = planner.exploration.duties[1]
        resident.home, resident.food_home_set = np.array([20., 30.]), True
        group = planner.estimator.groups[1]
        group.anchored, group.world_size = True, (1000., 1000.)
        group.trees = [TreeLandmark(np.array([120., 160.]), 0)]
        hints = planner.exploration.instructions(states, planner.estimator, 1, values, population_phase=True)
        scouts = [key for key, duty in planner.exploration.duties.items() if duty.role == "scout"]
        self.assertEqual(len(scouts), 1)
        self.assertNotIn(1, scouts)
        np.testing.assert_array_equal(resident.home, [20, 30])
        for agent_id, duty in planner.exploration.duties.items():
            if duty.role == "resident":
                self.assertEqual(hints[agent_id].objective, "forage near breeding area")
                if agent_id != 1:
                    np.testing.assert_array_equal(duty.home, [120, 160])

    def test_nonelite_replaces_elite_scout_immediately_and_low_energy_population_walkers_stay_resident(self):
        planner = boundary_planner()
        states = connected_states(2)
        values = {1: TraitRating(1.2, True, False, 1.0, {}), 2: TraitRating(0.8, False, True, 0.0, {})}
        states[1]["energy"] = 40
        planner.instructions(states, 0, trait_ratings=values)
        self.assertEqual(planner.exploration.duties[1].role, "scout")
        states[1]["energy"] = 250
        planner.exploration.instructions(states, planner.estimator, 1, values, population_phase=True)
        self.assertEqual(planner.exploration.duties[1].role, "resident")
        self.assertEqual(planner.exploration.duties[2].role, "scout")
        singleton = boundary_planner()
        child = [agent_state(agent_id=0, energy=75)]
        singleton.instructions(child, 0)
        singleton.exploration.instructions(child, singleton.estimator, 1, population_phase=True)
        self.assertEqual(singleton.exploration.duties[0].role, "resident")

    def test_population_residents_are_not_forced_to_disperse_and_elites_scout_only_without_alternatives(self):
        planner = boundary_planner()
        states = connected_states()
        lower = {key: TraitRating(0.8, False, True, 0.0, {}) for key in range(1, 6)}
        planner.instructions(states, 0, trait_ratings=lower)
        planner.exploration.instructions(states, planner.estimator, 1, lower, population_phase=True)
        for duty in planner.exploration.duties.values():
            if duty.role == "resident":
                self.assertFalse(duty.dispersing)
        self.assertEqual(sum(duty.role == "scout" for duty in planner.exploration.duties.values()), 1)
        elites = {key: TraitRating(1.2, True, False, 1.0, {}) for key in range(1, 6)}
        planner.exploration.instructions(states, planner.estimator, 31, elites, population_phase=True)
        self.assertEqual(sum(duty.role == "scout" for duty in planner.exploration.duties.values()), 1)


if __name__ == "__main__":
    unittest.main()

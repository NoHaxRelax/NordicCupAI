import unittest

import numpy as np

from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.world_estimator import EdgeLandmark, TreeLandmark
from test_exploration import accelerated_config, connected_states


def resident_planner(energy=250, **changes):
    config = dict(balanced_resident_homes_enabled=True, population_scout_fraction=0)
    config.update(changes)
    planner = GlobalPlanner(accelerated_config(**config))
    states = connected_states(6, energy=energy)
    planner.instructions(states, 0)
    group = planner.estimator.groups[1]
    group.anchored, group.world_size = True, (1000., 1000.)
    group.trees = [TreeLandmark(np.array(position, dtype=float), 0)
                   for position in ((20, 0), (30, 0), (200, 0))]
    for pose in planner.estimator.poses.values():
        pose.position = np.zeros(2)
    return planner, states, group


class ResidentHomeTests(unittest.TestCase):
    def test_new_residents_share_separate_patches_and_adjacent_trees_share_load(self):
        planner, states, _ = resident_planner()
        planner.exploration.instructions(states, planner.estimator, 1, population_phase=True)
        homes = [duty.home for duty in planner.exploration.duties.values()]
        np.testing.assert_array_equal(homes[:3], [[20, 0], [20, 0], [200, 0]])
        self.assertGreater(sum(np.array_equal(home, [200, 0]) for home in homes), 1)
        self.assertTrue(all(duty.food_home_set for duty in planner.exploration.duties.values()))

    def test_existing_homes_survive_new_tree_discovery_and_repeated_calls(self):
        planner, states, group = resident_planner()
        planner.exploration.instructions(states, planner.estimator, 1, population_phase=True)
        expected = {key: duty.home.copy() for key, duty in planner.exploration.duties.items()}
        group.trees.append(TreeLandmark(np.array([0., 0.]), 2))
        for now in (1, 2, 40):
            planner.exploration.instructions(states, planner.estimator, now, population_phase=True)
            for key, duty in planner.exploration.duties.items():
                np.testing.assert_array_equal(duty.home, expected[key])

    def test_disabled_or_unanchored_uses_nearest_tree(self):
        for enabled, anchored in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, anchored=anchored):
                planner, states, group = resident_planner(balanced_resident_homes_enabled=enabled)
                group.anchored = anchored
                planner.exploration.instructions(states, planner.estimator, 1, population_phase=True)
                for duty in planner.exploration.duties.values():
                    np.testing.assert_array_equal(duty.home, [20, 0])

    def test_low_energy_residents_stay_nearby_and_hungry_residents_prioritize_food(self):
        planner, states, _ = resident_planner(energy=100)
        planner.exploration.instructions(states, planner.estimator, 1, population_phase=True)
        for duty in planner.exploration.duties.values():
            np.testing.assert_array_equal(duty.home, [20, 0])
            self.assertFalse(duty.food_home_set)
        states[0]["energy"] = 60
        duty = planner.exploration.duties[1]
        duty.home = np.array([200., 0.])
        hints = planner.exploration.instructions(states, planner.estimator, 2, population_phase=True)
        self.assertEqual(hints[1].objective, "find food near remembered tree")
        self.assertIsNone(hints[1].food_distance_limit)
        np.testing.assert_array_equal(duty.home, [200, 0])

    def test_young_residents_get_balanced_homes_after_recovering_energy(self):
        planner, states, _ = resident_planner(energy=100)
        planner.exploration.instructions(states, planner.estimator, 1, population_phase=True)
        self.assertTrue(all(not duty.food_home_set for duty in planner.exploration.duties.values()))
        for state in states:
            state["energy"] = 250
        planner.exploration.instructions(states, planner.estimator, 2, population_phase=True)
        self.assertTrue(all(duty.food_home_set for duty in planner.exploration.duties.values()))
        np.testing.assert_array_equal(planner.exploration.duties[3].home, [200, 0])

    def test_remote_or_obstructed_patches_do_not_pull_residents_away(self):
        for blocked in (False, True):
            with self.subTest(blocked=blocked):
                planner, states, group = resident_planner()
                if blocked:
                    group.edges = [EdgeLandmark(np.array([100., -100.]), np.array([100., 100.]), 0)]
                else:
                    group.trees[-1].position = np.array([500., 0.])
                planner.exploration.instructions(states, planner.estimator, 1, population_phase=True)
                for duty in planner.exploration.duties.values():
                    np.testing.assert_array_equal(duty.home, [20, 0])

    def test_balancing_requires_population_phase_and_replay_is_deterministic(self):
        first, states, group = resident_planner()
        first.exploration.instructions(states, first.estimator, 1)
        self.assertTrue(all(not duty.food_home_set for duty in first.exploration.duties.values()))
        first.exploration.instructions(states, first.estimator, 2, population_phase=True)
        expected = {key: duty.home.copy() for key, duty in first.exploration.duties.items()}
        first.exploration.reset()
        first.exploration.instructions(states, first.estimator, 2, population_phase=True)
        for key, duty in first.exploration.duties.items():
            np.testing.assert_array_equal(duty.home, expected[key])


if __name__ == "__main__":
    unittest.main()

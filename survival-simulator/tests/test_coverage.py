import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.utils.controllers.coverage import CoverageConfig, CoverageCoordinator
from src.utils.controllers.territories import connected_parts
from src.utils.controllers.survey_gaps import SurveyGaps, sensed_points
from src.utils.controllers.world_estimator import EstimatedPose, MapGroup, EdgeLandmark, TreeLandmark
from src.utils.controllers.harvest import HarvestConfig, HarvestCoordinator
import test_harvest
from test_expert_policy import agent_state, fruit


def view(position=(100., 100.), hearing=50., vision=200., heading=0., edges=()):
    return (0., np.array(position), heading, hearing, vision, math.pi / 3, edges, [])


class GapTests(unittest.TestCase):
    def setUp(self):
        self.group = MapGroup(1, anchored=True, world_size=(400, 400))
        self.pose = EstimatedPose(1, 1, np.array([100., 100.]))
        self.gaps = SurveyGaps((400, 400))

    def test_single_pixel_and_long_narrow_gap_are_ignored(self):
        self.gaps.seen[:] = True
        self.gaps.seen[10, 10] = False
        self.gaps.seen[20, 5:25] = False
        self.assertEqual(self.gaps.find(self.group, [self.pose]), [])

    def test_tree_sized_unseen_patch_gets_a_target(self):
        self.gaps.seen[:] = True
        self.gaps.seen[10:15, 10:15] = False
        patches = self.gaps.find(self.group, [self.pose])
        self.assertEqual(len(patches), 1)
        self.assertGreaterEqual(patches[0]["area"], 625)
        self.assertGreaterEqual(patches[0]["clearance"], 12.5)

    def test_closed_observed_rock_is_not_a_scouting_gap(self):
        corners = [(140, 140), (220, 140), (220, 220), (140, 220)]
        self.group.edges = [EdgeLandmark(np.array(a), np.array(b), 0) for a, b in zip(corners, corners[1:] + corners[:1])]
        self.gaps.seen[:] = True
        points = self.gaps.points
        self.gaps.seen.ravel()[(points[:, 0] > 140) & (points[:, 0] < 220)
                              & (points[:, 1] > 140) & (points[:, 1] < 220)] = False
        self.assertFalse(self.gaps.find(self.group, [self.pose]))

    def test_sensing_credits_hearing_but_not_occluded_vision(self):
        v = view(edges=(((60., -100.), (60., 100.)),))
        points = np.array([[130., 100.], [180., 100.], [90., 200.]])
        self.assertEqual(sensed_points(points, v).tolist(), [True, False, False])
        self.gaps.observe({1: v})
        self.assertTrue(self.gaps.seen.any())
        self.assertFalse(self.gaps.seen.all())

    def test_memory_budget_scales_on_large_maps(self):
        gaps = SurveyGaps((10000, 10000), maximum=1000)
        self.assertLessEqual(len(gaps.points), 1000)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.coverage = CoverageCoordinator(CoverageConfig())
        self.group = MapGroup(1, anchored=True, world_size=(1000, 800))
        self.states = [agent_state(agent_id=i, energy=300) for i in range(4)]
        self.poses = {i: EstimatedPose(i, 1, np.array([100. + i, 100.])) for i in range(4)}

    def update(self, now=0, views=None, biome_layer=None, fruits=()):
        visible = HarvestCoordinator(HarvestConfig())._visible
        self.coverage.update(self.states, self.poses, self.group, views or {}, visible, {}, now, 4.,
                             biome_layer=biome_layer, fruits=fruits)

    def test_clustered_agents_receive_distinct_distributed_areas(self):
        self.update()
        homes = np.array(list(self.coverage.homes.values()))
        self.assertEqual(len(homes), 4)
        self.assertGreater(np.ptp(homes[:, 0]), 400)
        self.assertGreater(np.ptp(homes[:, 1]), 300)
        self.assertEqual(len({tuple(self.coverage.destination(i)) for i in range(4)}), 4)
        self.assertEqual(len(self.coverage.patrol_ids), 4)
        for agent_id in self.coverage.homes:
            self.assertEqual(self.coverage.owner(self.coverage.destination(agent_id)), agent_id)
            self.assertEqual(len(connected_parts(np.flatnonzero(self.coverage.owners == agent_id), self.coverage.graph)), 1)
        self.assertEqual(self.coverage.snapshot()["survey_percent"], 0)

    def test_observation_not_assignment_marks_surveyed_and_old_orchards_become_due(self):
        self.group.trees = [TreeLandmark(np.array([200., 200.]), 0)]
        self.update(0, {1: view(hearing=150.)})
        before = self.coverage.snapshot()["survey_percent"]
        self.assertGreater(before, 0)
        self.update(30)
        self.assertEqual(self.coverage.snapshot()["survey_percent"], before)
        self.assertEqual(self.coverage.snapshot()["recent_percent"], 0)

    def test_hungry_agents_keep_their_areas_across_refreshes(self):
        self.update()
        owners = self.coverage.owners.copy()
        self.states[0]["energy"] = 40
        self.update(30)
        np.testing.assert_array_equal(self.coverage.owners, owners)
        self.assertIn(0, self.coverage.homes)

    def test_frame_change_resets_survey_memory(self):
        self.update(0, {1: view(hearing=150.)})
        self.group.frame_revision += 1
        self.update(1)
        self.assertEqual(self.coverage.snapshot()["survey_percent"], 0)

    def test_coarse_duties_exclude_enclosed_observed_rocks(self):
        self.coverage._grid(self.group)
        index = len(self.coverage.points) // 2
        x, y = self.coverage.points[index]
        corners = [(x - 30, y - 30), (x + 30, y - 30),
                   (x + 30, y + 30), (x - 30, y + 30)]
        self.group.edges = [EdgeLandmark(np.array(a), np.array(b), 0)
                            for a, b in zip(corners, corners[1:] + corners[:1])]
        self.update()
        self.assertFalse(self.coverage.surveyable[index])
        self.assertNotIn(index, self.coverage.targets.values())
        self.assertFalse(any(np.array_equal(home, [x, y]) for home in self.coverage.homes.values()))

    def test_nearby_agent_is_recruited_to_a_worthwhile_hole(self):
        self.update()
        gap = self.coverage.gaps
        gap.seen[:] = True
        # A 50 x 50 patch just beyond the clustered agents' sensing disk.
        gap.seen[10:15, 10:15] = False
        self.update(2)
        self.assertEqual(len(self.coverage.gap_targets), 1)
        agent_id = next(iter(self.coverage.gap_targets))
        self.assertIn(agent_id, self.coverage.patrol_ids)
        self.assertTrue(self.coverage.hint(agent_id, self.poses[agent_id]).survey)
        self.assertLess(np.linalg.norm(self.coverage.gap_targets[agent_id] - self.poses[agent_id].position), 220)

    def test_food_rich_regions_are_smaller_and_shares_are_balanced(self):
        self.group.trees = [TreeLandmark(np.array([x, y]), 0)
                            for x in (100., 180., 260.) for y in (160., 320., 480., 640.)]
        self.update()
        shares = self.coverage.snapshot()["food_by_owner"]
        self.assertLess(max(shares.values()) / min(shares.values()), 1.6)
        counts = [np.sum(self.coverage.owners == i) for i in shares]
        self.assertGreater(max(counts) / min(counts), 2.)

    def test_all_fine_cells_in_each_territory_are_connected_around_rocks(self):
        from scipy.ndimage import label
        corners = [(300., 200.), (600., 200.), (600., 350.), (300., 350.)]
        self.group.edges = [EdgeLandmark(np.array(a), np.array(b), 0)
                            for a, b in zip(corners, corners[1:] + corners[:1])]
        self.update()
        grid = self.coverage.cell_labels
        for agent_id in self.coverage.homes:
            owned = np.zeros_like(grid, dtype=bool)
            valid = grid >= 0
            owned[valid] = self.coverage.owners[grid[valid]] == agent_id
            self.assertEqual(label(owned)[1], 1)
        self.assertIsNone(self.coverage.owner((450., 250.)))

    def test_birth_gets_a_region_without_shuffling_every_owner(self):
        self.update()
        before = self.coverage.owners.copy()
        self.states.append(agent_state(agent_id=4, energy=80))
        self.poses[4] = EstimatedPose(4, 1, np.array([100., 100.]))
        self.update(.1)
        self.assertIn(4, self.coverage.homes)
        self.assertGreater(np.mean(self.coverage.owners == before), .5)
        self.assertTrue(np.all(self.coverage.owners >= 0))
        for agent_id in self.coverage.homes:
            self.assertEqual(len(connected_parts(np.flatnonzero(self.coverage.owners == agent_id), self.coverage.graph)), 1)

    def test_death_releases_food_to_connected_surviving_territories(self):
        self.update()
        self.states = self.states[1:]
        self.update(.1)
        self.assertNotIn(0, self.coverage.homes)
        self.assertTrue(np.all(self.coverage.owners > 0))
        for agent_id in self.coverage.homes:
            self.assertEqual(len(connected_parts(np.flatnonzero(self.coverage.owners == agent_id), self.coverage.graph)), 1)

    def test_isolated_components_keep_their_residents_even_when_food_is_unequal(self):
        self.group.edges = [EdgeLandmark(np.array([500., 0.]), np.array([500., 800.]), 0)]
        self.group.trees = [TreeLandmark(np.array([x, y]), 0)
                            for x in (100., 200., 300.) for y in (100., 300., 500., 700.)]
        for agent_id in (1, 2, 3):
            self.poses[agent_id].position = np.array([800., 100. + agent_id * 100.])
        self.update()
        left = self.coverage.points[:, 0] < 500
        self.assertTrue(np.all(self.coverage.owners[left] == 0))
        self.assertEqual(set(self.coverage.owners[~left]), {1, 2, 3})
        # The newborn must split a reachable eastern region even though the
        # western owner has by far the largest food budget.
        self.states.append(agent_state(agent_id=4, energy=80))
        self.poses[4] = EstimatedPose(4, 1, np.array([800., 500.]))
        self.update(.1)
        self.assertTrue(np.all(self.coverage.owners[left] == 0))
        self.assertIn(4, self.coverage.homes)
        for agent_id in self.coverage.homes:
            same_side = (self.coverage.homes[agent_id][0] < 500) == (self.poses[agent_id].position[0] < 500)
            self.assertTrue(same_side)

    def test_inferred_biomes_are_a_weak_food_prior(self):
        layer = dict(bounds=[0, 0, 1000, 800], palette=["forest", "desert"], labels=[[0, 1]], confidence=[[1., 1.]])
        self.update(biome_layer=layer)
        forest = self.coverage.food_weights[self.coverage.points[:, 0] < 500].mean()
        desert = self.coverage.food_weights[self.coverage.points[:, 0] > 500].mean()
        self.assertGreater(forest, desert)
        self.group.trees = [TreeLandmark(np.array([700., 400.]), 30)]
        self.update(30, biome_layer=layer)
        near_tree = np.argmin(np.linalg.norm(self.coverage.points - [700., 400.], axis=1))
        self.assertGreater(self.coverage.food_weights[near_tree], forest)

    def test_expired_fruit_sites_do_not_permanently_bias_food_potential(self):
        self.update(fruits=[SimpleNamespace(position=np.array([100., 100.]))])
        initial = self.coverage.food_weights.sum()
        self.assertTrue(self.coverage.fruit_sites)
        self.update(121)
        self.assertFalse(self.coverage.fruit_sites)
        self.assertLess(self.coverage.food_weights.sum(), initial)

    def test_survey_goal_persists_until_observed_and_rejected_goals_are_avoided(self):
        self.update()
        agent_id = 1
        destination = self.coverage.destination(agent_id)
        self.poses[agent_id].position += [50., 20.]
        self.update(2)
        np.testing.assert_array_equal(self.coverage.destination(agent_id), destination)
        hint = self.coverage.hint(agent_id, self.poses[agent_id])
        np.testing.assert_allclose(hint.vector, destination - self.poses[agent_id].position)
        self.coverage.reject_target(agent_id, 2)
        self.update(2.1)
        alternate = self.coverage.destination(agent_id)
        self.assertTrue(alternate is None or np.linalg.norm(alternate - destination) >= 25.)

    def test_single_unsensed_dot_does_not_become_a_patrol_goal(self):
        self.update()
        self.coverage.seen[:] = 0
        self.coverage.gaps.seen[:] = True
        cell = 22
        x, y = self.coverage._fine_indices([self.coverage.points[cell]])[0]
        self.coverage.seen[cell] = -1
        self.coverage.gaps.seen[y, x] = False
        for agent_id in range(4):
            self.coverage._clear_target(agent_id)
        self.update(2)
        self.assertFalse(self.coverage.patrol_ids)

    def test_known_dimensions_required_and_disabled_mode_is_inert(self):
        self.group.world_size = None
        self.update()
        self.assertFalse(self.coverage.homes)
        self.coverage = CoverageCoordinator(CoverageConfig(enabled=False))
        self.group.world_size = (1000, 800)
        self.update()
        self.assertFalse(self.coverage.homes)

    def test_hungry_patrol_prefers_food_biomes_over_an_equally_close_desert(self):
        self.states = [agent_state(agent_id=0, energy=75)]
        self.poses[0].position = np.array([500., 400.])
        layer = dict(bounds=[0, 0, 1000, 800], palette=["forest", "desert"],
                     labels=[[0, 1]], confidence=[[1., 1.]])
        self.update(biome_layer=layer)
        self.assertLess(self.coverage.destination(0)[0], 500.)

    def test_energy_drop_replaces_an_unaffordable_persistent_target_immediately(self):
        self.states = [agent_state(agent_id=0, energy=300)]
        self.update()
        farthest = int(np.argmax(np.linalg.norm(self.coverage.points - self.poses[0].position, axis=1)))
        self.coverage.gap_targets.pop(0, None)
        self.coverage.targets[0] = farthest
        self.coverage.target_since[0] = 0.
        self.states[0]["energy"] = 20.
        self.update(.1)  # Earlier than the ordinary two-second replan.
        distance = np.linalg.norm(self.coverage.destination(0) - self.poses[0].position)
        self.assertLess(distance, 200.)
        self.assertNotEqual(self.coverage.targets.get(0), farthest)

    def test_low_energy_newborn_can_search_nearby_across_a_territory_boundary(self):
        self.update()
        owners = self.coverage.owners.copy()
        agent_id = next(i for i in self.coverage.homes
                        if self.coverage.owner(self.poses[i].position) != i)
        self.states[agent_id]["energy"] = 30.
        self.coverage._clear_target(agent_id)
        self.coverage.next_plan = 0.
        self.update(.1)
        destination = self.coverage.destination(agent_id)
        self.assertLess(np.linalg.norm(destination - self.poses[agent_id].position), 200.)
        self.assertNotEqual(self.coverage.owner(destination), agent_id)
        np.testing.assert_array_equal(self.coverage.owners, owners)

    def test_hungry_search_still_respects_disconnected_observed_components(self):
        self.states = [agent_state(agent_id=0, energy=75), agent_state(agent_id=1, energy=300)]
        self.poses[0].position = np.array([450., 400.])
        self.poses[1].position = np.array([800., 400.])
        self.group.edges = [EdgeLandmark(np.array([500., 0.]), np.array([500., 800.]), 0)]
        self.group.trees = [TreeLandmark(np.array([550., 400.]), 0)]
        self.update()
        self.assertLess(self.coverage.destination(0)[0], 500.)

    def test_food_search_does_not_wait_for_the_normal_survey_revisit_deadline(self):
        self.states = [agent_state(agent_id=0, energy=75)]
        self.update()
        self.coverage.seen[:] = 0.
        self.coverage.gaps.seen[:] = True
        self.coverage._clear_target(0)
        self.update(2.)
        self.assertEqual(self.coverage.tasks[0], "forage")
        self.assertGreater(np.linalg.norm(self.coverage.destination(0) - self.poses[0].position), 25.)

    def test_tree_food_prior_fades_without_new_sightings(self):
        self.group.trees = [TreeLandmark(np.array([200., 200.]), 0)]
        self.update()
        before = self.coverage.food_weights.sum()
        self.update(46.)
        self.assertLess(self.coverage.food_weights.sum(), before)
        self.assertFalse(self.coverage.orchards.any())

    def test_slow_terrain_and_age_reduce_safe_search_range(self):
        young = agent_state(agent_id=0, energy=75, age=10)
        old = dict(young, age=100)
        river = dict(young, biome="river")
        self.assertLess(self.coverage._travel_budget(old), self.coverage._travel_budget(young))
        self.assertLess(self.coverage._travel_budget(river), self.coverage._travel_budget(young))

    def test_unaffordable_search_stops_but_still_periodically_looks_for_food(self):
        self.states = [agent_state(agent_id=0, energy=1)]
        self.update(views={0: view()})
        hint = self.coverage.hint(0, self.poses[0])
        self.assertEqual(self.coverage.tasks[0], "conserve energy")
        self.assertEqual(hint.vector, (0., 0.))
        self.assertTrue(hint.scan_while_stationary)
        self.assertEqual(hint.look_direction, math.pi / 4)
        self.update(1.5)
        hint = self.coverage.hint(0, self.poses[0])
        self.assertFalse(hint.scan_while_stationary)

    def test_low_energy_still_allows_an_affordable_emergency_search(self):
        state = agent_state(agent_id=0, energy=10, age=10)
        self.assertGreater(self.coverage._travel_budget(state), 50.)


class CoverageIntegrationTests(unittest.TestCase):
    setUp = test_harvest.HarvestTests.setUp
    step = test_harvest.HarvestTests.step

    def test_hungry_agent_can_borrow_food_without_losing_territory(self):
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True))
        states = [agent_state([fruit(80)], agent_id=i, energy=300) for i in range(4)]
        hints, _ = self.step(states, 0)
        owner = self.harvest.coverage.owner(np.array([180., 100.]))
        patroller = next(i for i in range(4) if i != owner)
        before = self.harvest.coverage.owners.copy()
        for state in states:
            if state["agent_id"] == patroller:
                state["energy"] = 30
        hints, _ = self.step(states, 1)
        self.assertFalse(hints[patroller].survey)
        self.assertIsNotNone(hints[patroller].track_id)
        np.testing.assert_array_equal(self.harvest.coverage.owners, before)

    def test_stationary_search_scan_survives_harvest_navigation(self):
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True))
        hints, _ = self.step([agent_state(agent_id=1, energy=1)], 0)
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertTrue(hints[1].scan_while_stationary)
        self.assertEqual(hints[1].look_direction, math.pi / 4)


if __name__ == "__main__":
    unittest.main()

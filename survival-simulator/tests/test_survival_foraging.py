"""Regressions from full-horizon no-predator colony failures."""

import unittest
from types import SimpleNamespace

import numpy as np

from src.utils.controllers.expert_policy import ExpertPolicy
from src.utils.controllers.harvest import FruitTrack
from src.utils.controllers.policy_inputs import HarvestHint
from src.utils.controllers.world_estimator import TreeLandmark
import test_coverage
from test_expert_policy import agent_state, fruit
import test_harvest


class SurvivalSearchTests(unittest.TestCase):
    setUp = test_coverage.CoverageTests.setUp
    update = test_coverage.CoverageTests.update

    def test_search_takes_affordable_partial_step_when_all_centers_are_out_of_range(self):
        self.states = [agent_state(agent_id=0, energy=6., biome="river")]
        self.update()
        self.coverage.gaps.seen[:] = True
        self.coverage.seen[:] = 0.
        self.coverage._clear_target(0)
        self.update(2.)
        budget = self.coverage._travel_budget(self.states[0])
        self.assertGreater(np.min(np.linalg.norm(self.coverage.points - self.poses[0].position, axis=1)), budget)
        distance = np.linalg.norm(self.coverage.destination(0) - self.poses[0].position)
        self.assertGreater(distance, 8.)
        self.assertLessEqual(distance, budget)
        self.assertEqual(self.coverage.tasks[0], "short food search")

    def test_equal_distance_search_accounts_for_slow_swamp_movement(self):
        self.states = [agent_state(agent_id=0, energy=75.)]
        self.poses[0].position = np.array([500., 400.])
        layer = dict(bounds=[0, 0, 1000, 800], palette=["grassland", "swamp"],
                     labels=[[0, 1]], confidence=[[1., 1.]])
        self.update(biome_layer=layer)
        self.assertLess(self.coverage.destination(0)[0], 500.)

    def test_old_track_does_not_refresh_an_empty_sites_history(self):
        track = SimpleNamespace(position=np.array([100., 100.]), last_seen=0.)
        self.update(fruits=[track])
        self.update(20., fruits=[track])
        self.assertEqual(next(iter(self.coverage.fruit_sites.values()))[1], 0.)
        self.update(31., fruits=[track])
        self.assertFalse(self.coverage.fruit_sites)

    def test_fresh_empty_view_discounts_old_food_evidence(self):
        track = SimpleNamespace(position=np.array([100., 100.]), last_seen=0.)
        self.update(fruits=[track])
        before = self.coverage.food_weights.sum()
        self.update(2., views={0: test_coverage.view()}, fruits=[])
        self.assertLess(self.coverage.food_weights.sum(), before)
        self.assertLessEqual(next(iter(self.coverage.fruit_sites.values()))[1], -23.)

    def test_river_search_relocates_before_an_empty_local_food_history_expires(self):
        self.states = [agent_state(agent_id=0, energy=150., biome="river")]
        self.poses[0].position = np.array([550., 400.])
        layer = dict(bounds=[0, 0, 1000, 800], palette=["forest", "river"],
                     labels=[[0, 1]], confidence=[[1., 1.]])
        stale = [SimpleNamespace(position=np.array([x, 400.]), last_seen=0.)
                 for x in (530., 550., 570., 590.)]
        self.update(biome_layer=layer, fruits=stale)
        self.assertEqual(self.coverage.tasks[0], "seek productive biome")
        self.assertLess(self.coverage.destination(0)[0], 500.)
        self.assertLessEqual(np.linalg.norm(self.coverage.destination(0) - self.poses[0].position),
                             self.coverage._travel_budget(self.states[0]))

    def test_productive_swamp_is_not_forced_to_relocate(self):
        self.states = [agent_state(agent_id=0, energy=150., biome="swamp")]
        self.poses[0].position = np.array([550., 400.])
        layer = dict(bounds=[0, 0, 1000, 800], palette=["forest", "swamp"],
                     labels=[[0, 1]], confidence=[[1., 1.]])
        self.update(biome_layer=layer)
        self.assertNotEqual(self.coverage.tasks.get(0), "seek productive biome")

    def test_relocation_requires_known_reachable_affordable_terrain(self):
        self.states = [agent_state(agent_id=0, energy=20., biome="river")]
        self.poses[0].position = np.array([900., 400.])
        layer = dict(bounds=[0, 0, 1000, 800], palette=["forest", "river"],
                     labels=[[0, 1]], confidence=[[1., 1.]])
        self.update(biome_layer=layer)
        self.assertNotEqual(self.coverage.tasks.get(0), "seek productive biome")


class SurvivalFoodTests(unittest.TestCase):
    setUp = test_harvest.HarvestTests.setUp
    step = test_harvest.HarvestTests.step

    def test_young_agent_can_wait_safely_below_old_emergency_threshold(self):
        self.step([agent_state(agent_id=1, energy=75.)], 0.)
        state = agent_state([fruit(20.)], agent_id=1, energy=75.)
        hints, _ = self.step([state], 1.)
        self.assertTrue(hints[1].waiting)
        action = ExpertPolicy(self.cfg).action_decision(state, harvest_hint=hints[1])
        self.assertLessEqual(action.move_distance, 2.)
        state["energy"] = 40.
        hints, _ = self.step([state], 1.1)
        self.assertFalse(hints[1].waiting)
        action = ExpertPolicy(self.cfg).action_decision(state, harvest_hint=hints[1])
        self.assertGreater(action.move_distance, 2.)

    def test_reachable_food_can_extend_life_despite_large_passive_age_drain(self):
        self.step([agent_state(agent_id=1)], 0.)
        state = agent_state(agent_id=1, energy=300., age=100.)
        track = FruitTrack(1, np.array([400., 100.]), 0., 0., 0., True, 0.)
        utility, _, wait = self.harvest._candidate(state, self.poses[1], track, 0., .05)
        self.assertGreater(utility, 0.)
        self.assertEqual(wait, 0.)
        state["energy"] = 20.
        self.assertLess(self.harvest._candidate(state, self.poses[1], track, 0., .05)[0], 0.)

    def test_river_agent_does_not_camp_on_tree_across_biome_boundary(self):
        self.group.trees = [TreeLandmark(np.array([115., 100.]), 0.)]
        self.step([agent_state(agent_id=1, energy=300., biome="river")], 0.)
        self.assertNotIn(self.harvest.tasks[1]["kind"], ("watch orchard", "rest with food reserve"))

    def test_local_pickup_cannot_bypass_central_route_rejection(self):
        # The remembered wall need not be in the agent's current vision cone.
        state = agent_state([fruit(20.)], agent_id=1, energy=20.)
        hint = HarvestHint((0., 0.), None, 0., False, allow_local_food=False)
        action = ExpertPolicy(self.cfg).action_decision(state, harvest_hint=hint)
        self.assertEqual(action.move_distance, 0.)

    def test_central_food_reservations_reach_the_local_policy(self):
        hints, _ = self.step([agent_state([fruit(20.)], agent_id=1, energy=40.)], 0.)
        self.assertFalse(hints[1].allow_local_food)

    def test_well_fed_agent_also_leaves_an_unproductive_tree(self):
        tree = TreeLandmark(np.array([115., 100.]), 0.)
        self.group.trees = [tree]
        state = agent_state(agent_id=1, energy=150.)
        self.step([state], 0.)
        tree.last_seen = 8.1
        self.step([state], 8.1)
        self.assertNotEqual(self.harvest.tasks[1]["kind"], "watch orchard")


if __name__ == "__main__":
    unittest.main()

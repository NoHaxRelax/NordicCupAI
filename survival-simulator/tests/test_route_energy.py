"""Affordability uses the route around observed rocks, not target distance."""

from unittest.mock import patch
import unittest
from types import SimpleNamespace

import numpy as np

from src.utils.controllers.policy_inputs import HarvestHint
from src.utils.controllers.world_estimator import EdgeLandmark
from test_expert_policy import agent_state, fruit
import test_harvest
import test_territory_harvest


class RouteEnergyTests(unittest.TestCase):
    setUp = test_harvest.HarvestTests.setUp
    step = test_harvest.HarvestTests.step
    prepare = test_territory_harvest.TerritoryHarvestTests.prepare

    def wall(self, height):
        self.group.edges = [EdgeLandmark(np.array([120., 0.]), np.array([120., height]), 0.)]

    def biomes(self, far_biome):
        layer = dict(bounds=[100., 0., 200., 1000.], palette=["forest", far_biome],
                     labels=[[0, 1]], confidence=[[1., 1.]])
        self.planner.biome_estimator = SimpleNamespace(layers={1: layer})

    def test_meal_across_a_river_is_not_priced_as_its_forest_start(self):
        state = agent_state([fruit(100.)], agent_id=1, energy=10., biome="forest")
        self.biomes("forest")
        hints, _ = self.step([state], 0.)
        self.assertIsNotNone(hints[1].track_id)
        self.setUp()
        self.biomes("river")
        hints, _ = self.step([state], 0.)
        self.assertIsNone(hints[1].track_id)

    def test_patrol_across_a_river_preserves_search_energy(self):
        states = [agent_state(agent_id=1, energy=13., biome="forest")]
        coverage = self.prepare(states, owner=1,
            patrol=HarvestHint((100., 0.), None, 0., False, survey=True))
        self.biomes("river")
        hints, route = self.observe_route(states)
        self.assertAlmostEqual(route.remaining, 100.)
        coverage.reject_target.assert_called_once_with(1, 0.)
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertIn("energy budget", self.harvest.tasks[1]["kind"])

    def observe_route(self, states, now=0.):
        routes = []
        original = self.harvest.navigator.steer

        def steer(*args, **kwargs):
            route = original(*args, **kwargs)
            routes.append(route)
            return route

        with patch.object(self.harvest.navigator, "steer", side_effect=steer):
            hints, _ = self.step(states, now)
        return hints, routes[0]

    def test_nearby_fruit_with_a_fatal_rock_detour_is_released(self):
        self.wall(900.)
        states = [agent_state([fruit(40.)], agent_id=1, energy=40.)]
        hints, route = self.observe_route(states)
        self.assertFalse(route.blocked)
        self.assertGreater(route.remaining, 1500.)
        self.assertIsNone(hints[1].track_id)
        self.assertNotIn(1, self.harvest.assignments)
        self.assertTrue(self.harvest.blocked_until)
        self.assertNotIn(1, self.harvest.navigator.routes)
        self.assertIn("energy budget", self.harvest.tasks[1]["kind"])

    def test_an_affordable_meal_detour_remains_assigned(self):
        self.wall(150.)
        states = [agent_state([fruit(40.)], agent_id=1, energy=40.)]
        hints, route = self.observe_route(states)
        self.assertFalse(route.blocked)
        self.assertGreater(route.remaining, 40.)
        self.assertIsNotNone(hints[1].track_id)
        self.assertIn(1, self.harvest.assignments)
        self.assertFalse(self.harvest.blocked_until)
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)

    def test_unaffordable_patrol_detour_replans_to_a_shorter_route(self):
        states = [agent_state(agent_id=1, energy=40.)]
        coverage = self.prepare(states, owner=1,
            patrol=HarvestHint((40., 0.), None, 0., False, survey=True))
        self.wall(900.)
        hints, route = self.observe_route(states)
        self.assertFalse(route.blocked)
        self.assertGreater(route.remaining, 1500.)
        coverage.reject_target.assert_called_once_with(1, 0.)
        self.assertEqual(hints[1].vector, (0., 0.))
        coverage.hint.return_value = HarvestHint((-40., 0.), None, 0., False, survey=True)
        hints, route = self.observe_route(states, .1)
        self.assertLess(route.remaining, 100.)
        self.assertLess(hints[1].vector[0], 0.)
        self.assertNotIn("energy budget", self.harvest.tasks[1]["kind"])


if __name__ == "__main__":
    unittest.main()

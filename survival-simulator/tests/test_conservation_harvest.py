"""The conservation dial pauses optional work without blocking survival tasks."""

import unittest

import numpy as np

from src.utils.controllers.conservation import ConservationConfig, ConservationSchedule
from src.utils.controllers.expert_policy import ExpertPolicy
from src.utils.controllers.policy_inputs import HarvestHint
from src.utils.controllers.world_estimator import TreeLandmark
from test_expert_policy import agent_state, fruit, predator
import test_harvest
import test_territory_harvest


class ConservationHarvestTests(unittest.TestCase):
    step = test_harvest.HarvestTests.step
    prepare = test_territory_harvest.TerritoryHarvestTests.prepare

    def setUp(self):
        test_harvest.HarvestTests.setUp(self)
        self.harvest.conservation = ConservationSchedule(ConservationConfig(
            start_seconds=0., full_seconds=.1))

    def patrol(self, states):
        return self.prepare(states, owner=1,
            patrol=HarvestHint((100., 0.), None, 0., False, survey=True,
                               look_direction=.7))

    def test_optional_patrol_rests_and_resumes_the_same_route(self):
        states = [agent_state(agent_id=1, energy=150.)]
        coverage = self.patrol(states)
        hints, _ = self.step(states, 0.)
        route = self.harvest.navigator.routes[1]
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)
        for tick in range(1, 96):
            hints, _ = self.step(states, tick / 10.)
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertEqual(self.harvest.tasks[1]["kind"], "rest; conserve energy")
        hints, _ = self.step(states, 9.7)
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)
        self.assertIs(self.harvest.navigator.routes[1], route)
        self.assertEqual(route.retries, 0)
        coverage.reject_target.assert_not_called()
        # Full conservation removes optional head sweeping during travel.
        self.assertAlmostEqual(hints[1].look_direction, 0.)

    def test_known_meal_is_collected_during_a_rest_window(self):
        states = [agent_state([fruit(20.)], agent_id=1, energy=40.)]
        self.patrol(states)
        hints, _ = self.step(states, 5.)
        self.assertFalse(self.harvest.conservation.should_scout(1, 5.))
        self.assertIsNotNone(hints[1].track_id)
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)

    def test_observed_orchard_trip_bypasses_optional_rest(self):
        states = [agent_state(agent_id=1, energy=150.)]
        self.patrol(states)
        self.group.trees = [TreeLandmark(np.array([180., 100.]), 5.)]
        hints, _ = self.step(states, 5.)
        self.assertEqual(self.harvest.tasks[1]["kind"], "find food at orchard")
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)

    def test_river_escape_bypasses_optional_rest(self):
        states = [agent_state(agent_id=1, energy=150., biome="river")]
        coverage = self.patrol(states)
        coverage.tasks[1] = "seek productive biome"
        hints, _ = self.step(states, 5.)
        self.assertEqual(self.harvest.tasks[1]["kind"], "seek productive biome")
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)

    def test_predator_escape_overrides_rest_hint(self):
        state = agent_state([predator(20.)], agent_id=1, energy=150.)
        self.patrol([state])
        hints, _ = self.step([state], 5.)
        self.assertEqual(hints[1].vector, (0., 0.))
        action = ExpertPolicy(self.cfg).action_decision(state, harvest_hint=hints[1])
        self.assertGreater(action.move_distance, 0.)
        self.assertFalse(action.spawn_agent)


if __name__ == "__main__":
    unittest.main()

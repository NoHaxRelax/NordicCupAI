"""Integration contracts between territory ownership, harvesting, and routing."""

from types import SimpleNamespace
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

import test_harvest
from test_expert_policy import agent_state
from src.utils.controllers.expert_policy import ExpertPolicy
from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.harvest import FruitTrack
from src.utils.controllers.policy_inputs import HarvestHint, ReproductionHint
from src.utils.controllers.world_estimator import EstimatedPose, MapGroup


class TerritoryHarvestTests(unittest.TestCase):
    setUp = test_harvest.HarvestTests.setUp
    step = test_harvest.HarvestTests.step

    def prepare(self, states, *, owner=None, patrol=None):
        for state in states:
            agent_id = state["agent_id"]
            self.poses[agent_id] = EstimatedPose(agent_id, 1, np.array([100., 100.]))
        self.population.update(states, 0.)
        self.harvest.frame = (1, self.group.frame_revision)
        # Territory geometry is tested independently. Here the coordinator
        # supplies a definite owner and patrol so policy precedence is explicit.
        self.harvest.coverage = SimpleNamespace(
            update=Mock(), reset=Mock(), hint=Mock(return_value=patrol),
            owner=Mock(return_value=owner), homes={s["agent_id"]: np.array([100., 100.]) for s in states},
            gap_targets={}, tasks={}, reject_target=Mock())
        return self.harvest.coverage

    def ripe_track(self, track_id, x, *, now=20.):
        track = FruitTrack(track_id, np.array([x, 100.]), 0., now, -20., True, 0.)
        self.harvest.tracks[track_id] = track
        return track

    def assign(self, states, now=20.):
        self.harvest._assign(states, self.poses, now, self.population, self.cfg.mechanics)
        return self.harvest.assignments

    def test_nearby_food_is_shared_when_its_owner_is_farther_away(self):
        states = [agent_state(agent_id=1, energy=300), agent_state(agent_id=2, energy=300)]
        self.prepare(states, owner=2)
        self.poses[2].position = np.array([220., 100.])
        self.ripe_track(10, 130.)
        self.assertEqual(self.assign(states), {1: 10})

    def test_hungry_agent_can_borrow_foreign_reservation_without_duplicates(self):
        states = [agent_state(agent_id=1, energy=300), agent_state(agent_id=2, energy=40)]
        self.prepare(states, owner=1)
        self.ripe_track(10, 130.)
        self.ripe_track(11, 160.)
        self.harvest.assignments = {1: 10}
        assigned = self.assign(states)
        self.assertEqual(assigned, {2: 10, 1: 11})
        self.assertEqual(len(assigned), len(set(assigned.values())))

    def test_much_nearer_fruit_replaces_an_expensive_old_target(self):
        states = [agent_state(agent_id=1, energy=300)]
        self.prepare(states, owner=1)
        self.ripe_track(10, 180.)
        self.assertEqual(self.assign(states), {1: 10})
        self.ripe_track(11, 120., now=21.)
        self.assertEqual(self.assign(states, 21.), {1: 11})

    def test_legacy_scout_role_does_not_disable_owned_fruit(self):
        states = [agent_state(agent_id=1, energy=300)]
        self.prepare(states, owner=1, patrol=HarvestHint((150., 0.), None, 0., False, survey=True))
        self.planner.exploration_hints = {1: SimpleNamespace(role="scout")}
        self.ripe_track(10, 180.)
        with patch.object(self.harvest, "_observe"):
            hints, _ = self.step(states, 0.)
        self.assertEqual(hints[1].track_id, 10)
        self.assertFalse(hints[1].survey)

    def test_stalled_fruit_enters_cooldown_and_navigation_is_released(self):
        states = [agent_state(agent_id=1, energy=300)]
        self.prepare(states, owner=1)
        self.ripe_track(10, 180., now=0.)
        with patch.object(self.harvest, "_observe"), \
                patch.object(self.harvest.navigator, "release", wraps=self.harvest.navigator.release) as release:
            for now in range(11):
                hints, _ = self.step(states, float(now))
            self.assertNotIn(1, self.harvest.assignments)
            self.assertEqual(self.harvest.blocked_until[(1, 10)], 30.)
            self.assertEqual(hints[1].vector, (0., 0.))
            self.assertNotIn(1, self.harvest.navigator.routes)
            release.assert_called_once_with(1)
            for now in range(11, 30):
                self.step(states, float(now))
                self.assertNotIn(1, self.harvest.assignments)
            hints, _ = self.step(states, 30.)
        self.assertEqual(hints[1].track_id, 10)
        self.assertEqual(self.harvest.navigator.snapshot()[1]["replans"], 0)

    def test_patrol_uses_same_stall_recovery_and_rejects_failed_target(self):
        states = [agent_state(agent_id=1, energy=150)]
        coverage = self.prepare(states, owner=1,
                                patrol=HarvestHint((80., 0.), None, 0., False, survey=True))
        with patch.object(self.harvest.navigator, "release", wraps=self.harvest.navigator.release) as release:
            for now in range(11):
                hints, _ = self.step(states, float(now))
        coverage.reject_target.assert_called_once_with(1, 10.)
        release.assert_called_once_with(1)
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertNotIn(1, self.harvest.navigator.routes)
        self.assertEqual(self.harvest.tasks[1]["kind"], "blocked; choose another target")

    def test_aligned_territory_groups_skip_legacy_scouting_and_food_sections(self):
        planner = GlobalPlanner()
        planner.config = planner.config.model_copy(update={"enabled": True})
        planner.estimator.groups = {1: self.group}
        planner.estimator.poses = {1: EstimatedPose(1, 1, np.array([100., 100.]))}
        planner.population_phase = True
        with patch.object(planner.estimator, "update"), \
                patch.object(planner.biome_estimator, "update"), \
                patch.object(planner.exploration, "_assign") as old_roles, \
                patch.object(planner.exploration, "_frontier_plan") as old_frontiers, \
                patch("src.utils.controllers.global_planner.food_grid") as old_territories:
            planner.instructions([agent_state(agent_id=1)], 20., territory_policy=True)
        old_roles.assert_not_called()
        old_frontiers.assert_not_called()
        old_territories.assert_not_called()
        self.assertEqual(planner.exploration_hints, {})

    def test_disconnected_and_uncertain_agents_cannot_bypass_central_birth_slot_or_cap(self):
        self.harvest.config = self.harvest.config.model_copy(update={"survival_population": 12})
        states = [agent_state(agent_id=i, energy=400.) for i in (1, 2, 3)]
        self.prepare(states, owner=1)
        self.poses[2].group_id = 2
        self.planner.estimator.groups[2] = MapGroup(2, anchored=False)
        self.poses[3].uncertainty = 10.
        _, breeding = self.step(states, 0.)
        self.assertEqual(set(breeding), {1, 2, 3})
        self.assertEqual([i for i, hint in breeding.items() if hint.allowed], [1])
        self.harvest.config = self.harvest.config.model_copy(update={"minimum_population": 3, "maximum_population": 3})
        _, breeding = self.step(states, 1.)
        self.assertFalse(any(hint.allowed for hint in breeding.values()))

    def test_parent_reserve_is_checked_after_movement_cost(self):
        policy = ExpertPolicy(self.cfg)
        state = agent_state(agent_id=1, age=50.,
                            energy=self.cfg.mechanics.spawn_energy_cost + 100.1)
        breeding = ReproductionHint(0., True, minimum_energy_reserve=100.)
        still = policy.action_decision(state, harvest_hint=HarvestHint((0., 0.), None, 0., False),
                                       reproduction_hint=breeding)
        moving = policy.action_decision(state, harvest_hint=HarvestHint((100., 0.), 10, 60., False),
                                        reproduction_hint=breeding)
        self.assertTrue(still.spawn_agent)
        self.assertGreater(moving.move_distance, 0.)
        self.assertFalse(moving.spawn_agent)

    def test_hungry_agent_returns_to_its_territory_before_idling(self):
        states = [agent_state(agent_id=1, energy=150)]
        coverage = self.prepare(states, owner=2)
        coverage.homes[1] = np.array([300., 100.])
        hints, _ = self.step(states, 0.)
        self.assertEqual(self.harvest.tasks[1]["kind"], "return to territory")
        self.assertGreater(hints[1].vector[0], 0)
        coverage.owner.return_value = 1
        hints, _ = self.step(states, 1.)
        self.assertEqual(self.harvest.tasks[1]["kind"], "idle")
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertNotIn(1, self.harvest.navigator.routes)

    def test_unreachable_home_is_not_retried_every_step(self):
        states = [agent_state(agent_id=1, energy=150)]
        coverage = self.prepare(states, owner=2)
        coverage.homes[1] = np.array([300., 100.])
        failure = SimpleNamespace(blocked=True, waypoint=None, remaining=math.inf, status="blocked")
        with patch.object(self.harvest.navigator, "steer", return_value=failure) as steer:
            self.step(states, 0.)
            self.step(states, 1.)
            self.assertEqual(steer.call_count, 1)
            self.step(states, 20.)
            self.assertEqual(steer.call_count, 2)

    def test_hungry_agent_does_not_walk_to_an_unaffordable_territory_home(self):
        states = [agent_state(agent_id=1, energy=10)]
        coverage = self.prepare(states, owner=2)
        coverage.homes[1] = np.array([900., 100.])
        hints, _ = self.step(states, 0.)
        self.assertEqual(self.harvest.tasks[1]["kind"], "idle")
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertTrue(hints[1].scan_while_stationary)


if __name__ == "__main__":
    unittest.main()

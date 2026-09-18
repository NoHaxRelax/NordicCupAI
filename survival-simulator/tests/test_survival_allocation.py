from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.harvest import HarvestConfig, HarvestCoordinator
from src.utils.controllers.policy_inputs import ReproductionHint
from src.utils.controllers.population import PopulationTracker
from src.utils.controllers.world_estimator import EstimatedPose, MapGroup
from test_expert_policy import agent_state, fruit


class SurvivalAllocationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True, coverage={"enabled": False}))
        self.population = PopulationTracker(self.cfg.reproduction.selection)
        self.group = MapGroup(1, anchored=True, world_size=(1000, 1000))
        self.poses = {i: EstimatedPose(i, 1, np.array([100., 100.])) for i in (1, 2, 3)}
        self.planner = SimpleNamespace(population_phase=True,
            estimator=SimpleNamespace(poses=self.poses, groups={1: self.group}))
        self.states = [agent_state(agent_id=1, age=80., energy=200.),
                       agent_state(agent_id=2, age=10., energy=100.),
                       agent_state(agent_id=3, age=20., energy=100.)]

    def confirm_aging(self):
        tracker = self.harvest.metabolic_tracker
        tracker.update(self.states, 0.)
        for step in (1, 2):
            tracker.remember_actions([SimpleNamespace(agent_id=state["agent_id"],
                move_distance=0., turn_angle=0., spawn_agent=False) for state in self.states])
            for state in self.states:
                state["age"] += .1
                state["energy"] -= .1 + (state["age"] * .01 if state["agent_id"] == 1 else 0.)
            tracker.update(self.states, step / 10.)
        self.assertTrue(tracker.history[1].aging)

    def update(self, now=.2):
        self.population.update(self.states, now)
        return self.harvest.update(self.states, now, self.planner, self.population, self.cfg.mechanics, 350.)

    def test_age_or_unconfirmed_drain_alone_does_not_retire_an_agent(self):
        self.harvest.metabolic_tracker.update(self.states, 0.)
        self.harvest.metabolic_tracker.rates[1] = 9.2
        self.assertEqual(self.harvest._retiring_agents(self.states), set())
        self.harvest.metabolic_tracker.history[1].positive_samples = 1
        self.assertEqual(self.harvest._retiring_agents(self.states), set())

    def test_confirmed_senescence_retires_only_with_two_healthy_young(self):
        self.confirm_aging()
        self.assertEqual(self.harvest._retiring_agents(self.states), {1})
        self.states[2]["energy"] = 60.
        self.assertEqual(self.harvest._retiring_agents(self.states), set())
        self.states[2]["energy"] = 100.
        self.states[2]["age"] = 55.
        self.assertEqual(self.harvest._retiring_agents(self.states), set())
        self.states[2]["age"] = 54.9
        self.harvest.metabolic_tracker.rates[1] = 4.
        self.assertEqual(self.harvest._retiring_agents(self.states), set())

    def test_old_agent_rests_when_young_claim_all_food_and_cannot_override(self):
        self.confirm_aging()
        for state in self.states:
            state["observations"] = [fruit(30), fruit(50)]
        hints, _ = self.update()
        self.assertNotIn(1, self.harvest.assignments)
        self.assertEqual(set(self.harvest.assignments), {2, 3})
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertFalse(hints[1].allow_local_food)
        self.assertEqual(self.harvest.tasks[1]["kind"], "retired; preserve food for young")
        policy = ExpertPolicy(self.cfg)
        hungry = dict(self.states[0], energy=40.)
        action = policy.action_decision(hungry, harvest_hint=hints[1])
        self.assertEqual(action.move_distance, 0.)

    def test_old_agent_keeps_an_existing_meal_between_replans(self):
        self.confirm_aging()
        for state in self.states:
            state["observations"] = [fruit(30)]
        self.update()
        track_id = next(iter(self.harvest.tracks))
        self.harvest.assignments = {1: track_id}
        self.harvest.next_plan = 100.
        self.update(.3)
        self.assertIn(1, self.harvest.assignments)
        self.assertGreater(np.linalg.norm(self.harvest.hints[1].vector), 0.)

    def test_old_agent_harvests_remote_food_that_young_agents_cannot_use(self):
        self.confirm_aging()
        self.states[0]["observations"] = [fruit(30)]
        self.poses[2].position[:] = [900., 900.]
        self.poses[3].position[:] = [900., 850.]
        hints, _ = self.update()
        self.assertEqual(self.harvest._retiring_agents(self.states), {1})
        self.assertEqual(set(self.harvest.assignments), {1})
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)

    def test_aging_onset_before_ninety_does_not_steal_the_weak_young_agents_meal(self):
        self.states = self.states[:2]
        self.states[0].update(age=70., energy=100.)
        self.states[1]["energy"] = 50.
        self.confirm_aging()
        for state in self.states:
            state["observations"] = [fruit(30)]
        self.update()
        self.assertEqual(self.harvest._retiring_agents(self.states), set())
        self.assertEqual(set(self.harvest.assignments), {2})

    def test_old_agent_cannot_take_the_weak_young_agents_second_viable_meal(self):
        self.states = self.states[:2]
        self.states[0].update(age=70., energy=100.)
        self.states[1]["energy"] = 50.
        self.confirm_aging()
        for state in self.states:
            state["observations"] = [fruit(30), fruit(50)]
        self.update()
        self.assertEqual(len(self.harvest.tracks), 2)
        self.assertEqual(self.harvest._retiring_agents(self.states), set())
        self.assertEqual(set(self.harvest.assignments), {2})

    def test_publicly_healthy_older_carrier_also_protects_its_future_food(self):
        self.states = self.states[:2]
        self.states[0].update(age=70., energy=100.)
        self.states[1].update(age=100., energy=200.)
        self.confirm_aging()
        self.assertFalse(self.harvest.metabolic_tracker.history[2].aging)
        for state in self.states:
            state["observations"] = [fruit(30), fruit(50)]
        self.update()
        self.assertEqual(set(self.harvest.assignments), {2})

    def test_distant_positive_options_do_not_block_old_agents_local_food(self):
        self.confirm_aging()
        self.poses[2].position[:] = [400., 100.]
        self.poses[3].position[:] = [450., 100.]
        self.states[0]["observations"] = [fruit(30)]
        self.states[1]["observations"] = [fruit(10)]
        self.states[2]["observations"] = [fruit(10)]
        self.update()
        self.assertEqual(set(self.harvest.assignments), {1, 2, 3})
        old_meal = self.harvest.tracks[self.harvest.assignments[1]]
        np.testing.assert_allclose(old_meal.position, [130., 100.])
        for state in self.states[1:]:
            value, _, _ = self.harvest._candidate(state, self.poses[state["agent_id"]],
                old_meal, .2, self.cfg.mechanics.walking_energy_per_unit)
            self.assertGreater(value, 0.)

    def test_nearly_full_young_agents_do_not_protect_extra_meals_from_old_agents(self):
        self.confirm_aging()
        for state in self.states:
            state["observations"] = [fruit(30), fruit(50), fruit(70)]
        for state in self.states[1:]:
            state["energy"] = state["max_energy"] - 10.
        self.update()
        self.assertEqual(set(self.harvest.assignments), {1, 2, 3})

    def test_food_search_resumes_when_young_agents_are_no_longer_safe(self):
        self.confirm_aging()
        self.update()
        self.states[2]["energy"] = 40.
        for state in self.states:
            state["age"] += .1
        self.states[0]["observations"] = [fruit(30)]
        self.poses[2].position[:] = self.poses[3].position[:] = [800., 800.]
        self.harvest.next_plan = 0.
        hints, _ = self.update(.3)
        self.assertIn(1, self.harvest.assignments)
        self.assertGreater(np.linalg.norm(hints[1].vector), 0.)

    def test_rest_preserves_stored_energy_breeding_decision(self):
        self.confirm_aging()
        breeding = {1: ReproductionHint(140., True, 40., preserve_lineage=True)}
        with patch.object(self.harvest, "_breeders", return_value=breeding):
            hints, result = self.update()
        self.assertIs(result[1], breeding[1])
        policy = ExpertPolicy(self.cfg)
        action = policy.action_decision(self.states[0], harvest_hint=hints[1], reproduction_hint=result[1])
        self.assertTrue(action.spawn_agent)
        self.assertEqual(action.move_distance, 0.)

    def test_retired_agents_scan_only_for_bounded_intervals(self):
        self.confirm_aging()
        scans = 0
        for step in range(60):
            now = .2 + step / 10.
            hints, _ = self.update(now)
            self.assertEqual(hints[1].vector, (0., 0.))
            scans += hints[1].scan_while_stationary
        self.assertEqual(scans, 8)


if __name__ == "__main__":
    unittest.main()

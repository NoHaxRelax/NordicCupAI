"""Renewal forecasts use observed income and observed aging independently."""

import unittest

import test_harvest
from test_expert_policy import agent_state


class SurvivalForecastTests(unittest.TestCase):
    setUp = test_harvest.GlobalReproductionTests.setUp
    plan = test_harvest.GlobalReproductionTests.plan
    add_food = test_harvest.GlobalReproductionTests.add_food
    observe_metabolism = test_harvest.GlobalReproductionTests.observe_metabolism

    def young_colony(self):
        states = [agent_state(agent_id=i, age=10., energy=75.) for i in range(6)]
        states[0]["energy"] = 300.
        self.harvest.discoveries = [0.] * 26  # 36.4 expected energy/second.
        self.add_food(3)  # 180 standing energy, shared once among all six.
        return states

    def test_observed_arrivals_prevent_unnecessary_overlap_for_six_young_agents(self):
        states = self.young_colony()
        hints = self.plan(states)
        plan = self.harvest.population_plan
        self.assertEqual(plan["population"], plan["target"])
        self.assertEqual(plan["viable_in_30_seconds"], 6)
        self.assertEqual(plan["viable_in_60_seconds"], 6)
        self.assertAlmostEqual(plan["observed_arrival_energy_per_second"], 36.4)
        self.assertAlmostEqual(plan["credited_forecast_food_per_second"], 18.)
        self.assertFalse(plan["replacement_needed"])
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_standing_stock_does_not_create_an_arrival_forecast(self):
        states = self.young_colony()
        self.harvest.discoveries.clear()
        hints = self.plan(states)
        self.assertEqual(self.harvest.population_plan["observed_arrival_energy_per_second"], 0.)
        self.assertEqual(self.harvest.population_plan["credited_forecast_food_per_second"], 0.)
        self.assertTrue(self.harvest.population_plan["replacement_needed"])
        self.assertTrue(hints[0].allowed)

    def test_loss_of_current_local_food_immediately_removes_forecast_credit(self):
        states = self.young_colony()
        self.plan(states)
        self.harvest.tracks.clear()
        self.plan(states, .1)
        self.assertGreater(self.harvest.population_arrivals_supply, 0.)
        self.assertEqual(self.harvest.population_plan["credited_forecast_food_per_second"], 0.)
        self.assertTrue(self.harvest.population_plan["replacement_needed"])

    def test_remote_young_agent_gets_no_income_from_other_agents_food(self):
        states = self.young_colony()
        self.plan(states)
        self.poses[5].position[:] = [900., 900.]
        self.plan(states, .1)
        self.assertEqual(self.harvest._breeding_food(states, self.poses, .1)[5], 0.)
        self.assertEqual(self.harvest.population_plan["viable_in_30_seconds"], 5)
        self.assertEqual(self.harvest.population_plan["viable_in_60_seconds"], 5)

    def test_sustained_arrival_decline_restores_renewal_demand(self):
        states = self.young_colony()
        self.plan(states)
        self.harvest.discoveries.clear()
        self.add_food(3, now=120.)
        hints = self.plan(states, 120.)
        self.assertLess(self.harvest.population_arrivals_supply, 3.)
        self.assertLess(self.harvest.population_plan["viable_in_60_seconds"], 4)
        self.assertTrue(self.harvest.population_plan["replacement_needed"])
        self.assertTrue(hints[0].allowed)

    def test_arrivals_cannot_mask_confirmed_senescent_parents(self):
        states = [agent_state(agent_id=i, age=70., energy=100.) for i in range(6)]
        self.observe_metabolism(states, set(range(6)))
        self.harvest.discoveries = [0.] * 26
        self.add_food(3)
        self.plan(states)
        self.assertEqual(self.harvest.population_plan["viable_in_30_seconds"], 0)
        self.assertEqual(self.harvest.population_plan["viable_in_60_seconds"], 0)
        self.assertTrue(self.harvest.population_plan["critical"])

    def test_clean_healthy_observation_conditions_the_future_onset_interval(self):
        state = agent_state(agent_id=0, age=100., energy=300.)
        unobserved = self.harvest._forecast_age_drain(state, 30.)
        self.observe_metabolism([state], set())
        observed = self.harvest._forecast_age_drain(state, 30.)
        self.assertLess(observed, unobserved)
        self.assertAlmostEqual(self.harvest.metabolic_tracker.history[0].healthy_at_age, 100.)

    def test_confirmed_early_onset_exceeds_the_unconditioned_age_prior(self):
        state = agent_state(agent_id=0, age=70., energy=300.)
        prior = self.harvest._forecast_age_drain(state, 30.)
        self.assertAlmostEqual(prior, self.harvest._expected_age_drain(70., 30.))
        self.observe_metabolism([state], {0})
        confirmed = self.harvest._forecast_age_drain(state, 30.)
        self.assertGreater(confirmed, prior)
        self.assertAlmostEqual(confirmed, .1 * (70. * 30. + .5 * 30. ** 2))


if __name__ == "__main__":
    unittest.main()

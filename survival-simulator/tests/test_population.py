import math
import unittest

from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.policy_inputs import ReproductionHint
from src.utils.controllers.population import PopulationConfig, PopulationTracker, TRAITS
from test_expert_policy import agent_state, predator
from test_global_planner import action, planner_config, FIXTURES


def scaled(agent_id, multiplier=1.0, **extra):
    state = agent_state(agent_id=agent_id, **extra)
    for trait in TRAITS:
        state[trait] *= multiplier
    return state


class PopulationTests(unittest.TestCase):
    def tracker(self, **config):
        return PopulationTracker(PopulationConfig(enabled=True, **config))

    def test_score_averages_all_six_ratios_against_frozen_founders(self):
        tracker = self.tracker()
        tracker.update([agent_state(agent_id=i) for i in range(5)], 0)
        tracker.update([agent_state(agent_id=9, speed=12)], 1)
        self.assertAlmostEqual(tracker.ratings[9].ratios["speed"], 1.2)
        self.assertAlmostEqual(tracker.ratings[9].score, (1.2 + 5) / 6)
        self.assertEqual(tracker.baseline["speed"], 10)
        tracker.update([scaled(10, 1.2)], 2)
        self.assertAlmostEqual(tracker.ratings[10].score, 1.2)
        self.assertEqual(tracker.baseline["max_energy"], 500)

    def test_initial_population_mean_is_used_and_identical_agents_are_equal(self):
        tracker = self.tracker()
        tracker.update([scaled(1, 0.8), scaled(2, 1.2)], 0)
        self.assertEqual(tracker.baseline["speed"], 10)
        tracker.reset()
        tracker.update([agent_state(agent_id=i) for i in range(5)], 0)
        self.assertFalse(any(r.elite or r.low_rank for r in tracker.ratings.values()))
        self.assertTrue(all(r.percentile == 0.5 for r in tracker.ratings.values()))

    def test_top_and_bottom_twenty_percent_recompute_on_population_change(self):
        tracker = self.tracker()
        tracker.update([agent_state(agent_id=0)], 0)
        tracker.update([scaled(i, 0.7 + i * 0.1) for i in range(10)], 1)
        self.assertEqual({key for key, r in tracker.ratings.items() if r.elite}, {8, 9})
        self.assertEqual({key for key, r in tracker.ratings.items() if r.low_rank}, {0, 1})
        self.assertEqual(tracker.ratings[9].percentile, 1)
        tracker.update([scaled(i, 0.7 + i * 0.1) for i in range(5)], 2)
        self.assertEqual({key for key, r in tracker.ratings.items() if r.elite}, {4})
        self.assertEqual(set(tracker.ratings), set(range(5)))

    def test_cutoff_ties_are_stable_and_do_not_label_the_entire_population_elite(self):
        tracker, other = self.tracker(), self.tracker()
        founders = [agent_state(agent_id=0)]
        states = [scaled(i, 1.1 if i < 4 else 0.9) for i in range(5)]
        for instance, observations in ((tracker, states), (other, list(reversed(states)))):
            instance.update(founders, 0)
            instance.update(observations, 1)
        self.assertEqual(tracker.ratings, other.ratings)
        self.assertEqual(tracker.snapshot()["elite_ids"], [0])

    def test_energy_thresholds_cooldowns_retries_and_growth_target(self):
        tracker = self.tracker(growth_population_target=5)
        tracker.update([agent_state(agent_id=0)], 0)
        tracker.update([scaled(1, 1.2), scaled(2, 0.9), scaled(3, 1.0)], 1)
        self.assertEqual(tracker.reproduction_hint(1, 1, 350).energy_threshold, 220)
        self.assertEqual(tracker.reproduction_hint(2, 1, 350).energy_threshold, 300)
        self.assertEqual(tracker.reproduction_hint(3, 1, 350).energy_threshold, 260)
        birth = action(agent_id=1).model_copy(update={"spawn_agent": True})
        tracker.remember_actions([birth], 1)
        self.assertTrue(tracker.reproduction_hint(1, 1, 350).allowed)
        self.assertFalse(tracker.reproduction_hint(1, 1.1, 350).allowed)
        self.assertTrue(tracker.reproduction_hint(1, 9, 350).allowed)
        tracker.update([scaled(i, 1.2 if i == 1 else 1) for i in range(5)], 10)
        self.assertEqual(tracker.reproduction_hint(1, 10, 350).energy_threshold, 350)

    def test_low_capacity_can_breed_but_independent_reserve_still_applies(self):
        tracker = self.tracker()
        tracker.update([agent_state(max_energy=200)], 0)
        hint = tracker.reproduction_hint(7, 0, 350)
        self.assertEqual(hint.energy_threshold, 170)
        policy = ExpertPolicy(load_config(FIXTURES / "expert_policy.json"), planner_config(enabled=False))
        config = policy.config.model_dump()
        config["reproduction"]["minimum_energy_reserve"] = 100
        policy = ExpertPolicy(type(policy.config).model_validate(config), planner_config(enabled=False))
        self.assertFalse(policy.action_decision(agent_state(energy=190, max_energy=200),
                                               reproduction_hint=hint).spawn_agent)

    def test_rewind_reused_id_and_empty_population_reset_baseline_and_cooldown(self):
        tracker = self.tracker()
        tracker.update([agent_state()], 5)
        tracker.remember_actions([action().model_copy(update={"spawn_agent": True})], 5)
        tracker.update([agent_state(age=0, speed=20)], 6)
        self.assertEqual(tracker.baseline["speed"], 20)
        self.assertEqual(tracker.last_birth_request, {})
        tracker.update([], 7)
        self.assertEqual(tracker.snapshot()["baseline"], {})

    def test_visible_and_remembered_escape_override_breeding_bonus(self):
        policy = ExpertPolicy(load_config(FIXTURES / "expert_policy.json"), planner_config(enabled=False))
        for tick in range(3):
            state = agent_state([predator(10)] if tick == 0 else [], energy=450, age=5 + tick * 0.1)
            result = policy.action_decision(state, sim_time=tick * 0.1, reproduction_hint=ReproductionHint(150))
            self.assertFalse(result.spawn_agent)

    def test_batch_retries_keep_birth_decision_and_do_not_repeat_cooldown(self):
        config = load_config(FIXTURES / "expert_policy.json").model_dump()
        config["reproduction"]["selection"] = PopulationConfig(enabled=True).model_dump()
        policy = ExpertPolicy(type(load_config(FIXTURES / "expert_policy.json")).model_validate(config),
                              planner_config(enabled=False))
        states = [agent_state(energy=300)]
        first = policy.actions_for_step(states, 0)
        self.assertTrue(first[0].spawn_agent)
        self.assertEqual(first, policy.actions_for_step(states, 0))
        self.assertFalse(policy.actions_for_step(states, 0.1)[0].spawn_agent)
        self.assertEqual(policy.population.last_birth_request, {7: 0})


if __name__ == "__main__":
    unittest.main()

"""The colony grows after a stable shared absolute frame is established."""

from types import SimpleNamespace
import unittest

from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.population import PopulationConfig, PopulationTracker
from test_expert_policy import agent_state
from test_global_planner import planner_config, FIXTURES


class CoordinationPhaseTests(unittest.TestCase):
    def planner(self):
        planner = GlobalPlanner(planner_config(population_after_alignment=True, alignment_hold_seconds=1))
        planner.estimator.groups = {1: SimpleNamespace(anchored=False)}
        planner.estimator.poses = {7: SimpleNamespace(group_id=1)}
        return planner

    def test_requires_stable_shared_absolute_frame_then_latches_across_births(self):
        planner = self.planner()
        planner._update_phase(0)
        self.assertFalse(planner.population_phase)
        planner.estimator.groups[1].anchored = True
        planner._update_phase(1)
        planner._update_phase(1.9)
        self.assertFalse(planner.population_phase)
        planner._update_phase(2)
        self.assertTrue(planner.population_phase)
        self.assertEqual(planner.shared_frame_established_at, 2)
        planner.estimator.groups[8] = SimpleNamespace(anchored=False)
        planner.estimator.poses[8] = SimpleNamespace(group_id=8)
        planner._update_phase(2.1)
        self.assertTrue(planner.population_phase)
        planner.reset()
        self.assertFalse(planner.population_phase)

    def test_unjoined_agents_restart_hold_and_loss_of_absolute_frame_restarts_alignment(self):
        planner = self.planner()
        planner.estimator.groups[1].anchored = True
        planner._update_phase(0)
        planner.estimator.groups[8] = SimpleNamespace(anchored=False)
        planner.estimator.poses[8] = SimpleNamespace(group_id=8)
        planner._update_phase(.8)
        planner.estimator.poses[8].group_id = 1
        planner._update_phase(1)
        self.assertFalse(planner.population_phase)
        planner._update_phase(2)
        self.assertTrue(planner.population_phase)
        planner.estimator.groups[1].anchored = False
        planner._update_phase(2.1)
        self.assertFalse(planner.population_phase)
        self.assertIsNone(planner.shared_frame_established_at)

    def test_population_phase_shortens_cooldowns_and_preserves_parent_energy(self):
        tracker = PopulationTracker(PopulationConfig(enabled=True, post_alignment={"enabled": True}))
        initial = [agent_state(agent_id=1), agent_state(agent_id=2), agent_state(agent_id=3)]
        tracker.update(initial, 0)
        self.assertEqual(tracker.reproduction_hint(1, 0, 350).energy_threshold, 260)
        tracker.update([initial[0], {**initial[1], "speed": 12}, {**initial[2], "speed": 8}], 1)
        tracker.population_phase = True
        self.assertEqual(tracker.reproduction_hint(1, 1, 350).energy_threshold, 260)
        self.assertEqual(tracker.reproduction_hint(2, 1, 350).energy_threshold, 220)
        self.assertEqual(tracker.reproduction_hint(3, 1, 350).energy_threshold, 300)
        self.assertEqual(tracker.snapshot()["growth_population_target"], 60)
        tracker.remember_actions([SimpleNamespace(agent_id=2, spawn_agent=True)], 1)
        self.assertFalse(tracker.reproduction_hint(2, 5.9, 350).allowed)
        self.assertTrue(tracker.reproduction_hint(2, 6, 350).allowed)
        tracker.reset()
        self.assertFalse(tracker.population_phase)

    def test_step_activates_breeding_only_after_observed_absolute_alignment(self):
        expert = load_config(FIXTURES / "expert_policy.json")
        expert = expert.model_copy(update={"reproduction": expert.reproduction.model_copy(update={
            "selection": PopulationConfig(enabled=True, post_alignment={
                "enabled": True, "normal_energy_threshold": 220})})})
        config = planner_config(enabled=False, mapping_enabled=True, population_after_alignment=True,
                                alignment_hold_seconds=0, estimator={"anchor_boundaries": True})
        local = ExpertPolicy(expert, config)
        self.assertFalse(local.actions_for_step([agent_state(energy=230)], 0)[0].spawn_agent)
        absolute = ExpertPolicy(expert, config)
        walls = [{"type": "Edge", "coords": [[-100, -100], [1500, -100]]},
                 {"type": "Edge", "coords": [[-100, -100], [-100, 1100]]}]
        action = absolute.actions_for_step([agent_state(walls, energy=230)], 0)[0]
        self.assertTrue(absolute.planner.population_phase)
        self.assertTrue(action.spawn_agent)
        snapshot = absolute.planner.snapshot()
        self.assertEqual(snapshot["phase"], "population")
        self.assertEqual(snapshot["population"]["growth_population_target"], 60)


if __name__ == "__main__":
    unittest.main()

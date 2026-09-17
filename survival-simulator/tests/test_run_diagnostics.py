import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.core import SimulationCore
from src.utils.controllers.expert_policy import ExpertPolicy
from src.utils.run_diagnostics import TRAITS, RunDiagnostics, load_diagnostics_config
from src.utils.run_report import normalized_traits


def agent(agent_id=0, **overrides):
    values = dict(agent_id=agent_id, energy=150.0, speed=10.0, sprint_speed=20.0,
                  max_energy=500.0, hearing_radius=50.0, vision_radius=200.0, cone_angle=1.0)
    return SimpleNamespace(**{**values, **overrides})


def fake_sim(agents=None, fruits=None):
    agents = [agent()] if agents is None else agents
    env = SimpleNamespace(
        time=0.0, score=0.0, event_sink=None, agents=agents, fruits=fruits or [],
        predators=[], agents_dict={a.agent_id: a for a in agents}, agent_observations={},
    )
    return SimpleNamespace(env=env, seed=42, dt=0.1, env_width=1600, env_height=1200,
                           chunk_size=400, starting_agents=len(agents), starting_predators=0,
                           starting_fruits=len(env.fruits), starting_trees=0)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_diagnostics_config()

    def recorder(self, sim):
        recorder = RunDiagnostics(sim, self.config, self.temp.name)
        self.addCleanup(recorder.events.close)
        return recorder

    def test_trait_percentages_use_living_agents_and_fixed_starting_population_mean(self):
        sim = fake_sim(agents=[agent(0, speed=10), agent(1, speed=20)])
        rec = self.recorder(sim)
        # Both founders are gone; the living population now has one new agent.
        sim.env.agents = [agent(2, speed=30, sprint_speed=30, max_energy=750,
                                hearing_radius=75, vision_radius=300, cone_angle=1.5)]
        sim.env.time = 1
        rec.sample()
        sim.env.agents = []
        sim.env.time = 2
        rec.sample()
        values = normalized_traits(rec.samples)
        for trait in TRAITS:
            self.assertEqual(values[trait][0], 100)
            self.assertEqual(values[trait][1], 200 if trait == "speed" else 150)
            self.assertTrue(np.isnan(values[trait][2]))

    def test_trait_percentages_leave_undefined_baselines_missing(self):
        for baseline in (None, 0):
            samples = [{f"{trait}_mean": baseline for trait in TRAITS},
                       {f"{trait}_mean": 10 for trait in TRAITS}]
            for values in normalized_traits(samples).values():
                self.assertTrue(np.isnan(values).all())
        self.assertTrue(all(len(values) == 0 for values in normalized_traits([]).values()))

    def test_score_bins_exactly_account_for_food_predation_and_partial_second(self):
        food = SimpleNamespace(fruit_id=1, energy=60.0)
        sim = fake_sim(fruits=[food])
        rec = self.recorder(sim)
        for tick in range(3):
            rec.before_step([])
            if tick == 0:
                rec.on_event("fruit_eaten", agent=sim.env.agents[0], fruit=food)
                sim.env.score += 0.06
            if tick == 2:
                victim = sim.env.agents[0]
                victim.energy = 125.0
                rec.on_event("death", agent=victim, cause="predation", predator=None)
                sim.env.score -= 1.25
                sim.env.agents.clear()
            sim.env.time += sim.dt
            sim.env.score += sim.dt
            rec.after_step()
        rec.finish("extinction", make_plots=False)
        summary = rec.summary
        self.assertAlmostEqual(summary["final_score"], -0.89)
        self.assertAlmostEqual(summary["score_reconstruction_error"], 0)
        self.assertLess(summary["benchmark_percent"], 0)
        self.assertIsNone(summary["true_theoretical_maximum"])
        self.assertAlmostEqual(summary["optimistic_score_benchmark"], 3000.0602)
        with (rec.directory / "per_second.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["observed_seconds"]), 0.3)
        self.assertAlmostEqual(float(rows[0]["score_gain"]), -0.89)
        self.assertAlmostEqual(float(rows[0]["score_gain_per_second"]), -0.89 / 0.3)

    def test_float_tick_boundaries_do_not_create_extra_seconds(self):
        sim = fake_sim()
        rec = self.recorder(sim)
        for _ in range(20):
            rec.before_step([])
            sim.env.time += 0.1
            sim.env.score += 0.1
            rec.after_step()
        rec.finish("time_limit", make_plots=False)
        self.assertEqual(set(rec.bins), {0, 1})
        self.assertAlmostEqual(sum(row["survival_gain"] for row in rec.bins.values()), 2)
        self.assertAlmostEqual(rec.summary["survival_target_percent"], 2 / 3000 * 100)

    def test_birth_and_death_within_one_tick_preserve_parentage_and_peak(self):
        sim = fake_sim()
        rec = self.recorder(sim)
        rec.before_step([])
        baby = agent(1, energy=75, sprint_speed=25)
        rec.on_event("birth", agent=baby, parent=sim.env.agents[0])
        rec.on_event("death", agent=baby, cause="predation", predator=None)
        sim.env.time = 0.1
        sim.env.score = 0.1 - 0.75
        rec.after_step()
        rec.finish("time_limit", make_plots=False)
        self.assertEqual(rec.agents[1]["parent_id"], 0)
        self.assertEqual(rec.agents[1]["generation"], 1)
        self.assertEqual(rec.agents[0]["children"], 1)
        self.assertEqual(rec.agents[1]["sprint_speed"], 25)
        self.assertEqual(rec.summary["births"], 1)
        self.assertEqual(rec.summary["peak_population"], 2)
        self.assertEqual(rec.summary["predation_without_prior_decision"], 1)

    def test_recent_deaths_and_living_agents_are_not_given_shorter_fitness_windows(self):
        sim = fake_sim(agents=[agent(i) for i in range(5)])
        rec = self.recorder(sim)
        rec.step_start = 10
        rec.step_end = 10.1
        for i in range(5):
            rec.on_event("birth", agent=agent(i + 5), parent=sim.env.agents[i])
        rec.step_start = 39
        rec.step_end = 39.1
        latest = agent(10)
        rec.on_event("birth", agent=latest, parent=agent(5))
        rec.on_event("death", agent=latest, cause="energy_depletion", predator=None)
        sim.env.time = 40
        analysis = rec.generation_analysis()
        self.assertEqual(analysis["last_generation"]["generation"], 2)
        self.assertEqual(analysis["last_generation"]["eligible_at_comparison_age"], 0)
        self.assertIsNone(analysis["last_generation"]["survival_at_comparison_age"])
        self.assertIn("Insufficient follow-up", analysis["last_generation_verdict"])
        self.assertEqual(analysis["latest_assessable_generation"]["generation"], 1)
        self.assertEqual(analysis["generations"][0]["eligible_at_comparison_age"], 5)
        self.assertEqual(analysis["generations"][1]["eligible_at_comparison_age"], 5)

    def test_bigger_traits_alone_do_not_produce_a_better_generation_verdict(self):
        sim = fake_sim(agents=[agent(i) for i in range(5)])
        rec = self.recorder(sim)
        rec.step_start = 1
        for i in range(5):
            rec.on_event("birth", agent=agent(i + 5, sprint_speed=40), parent=sim.env.agents[i])
            rec.agents[i + 5]["children_by_comparison_age"] = 1
        sim.env.time = 40
        analysis = rec.generation_analysis()
        self.assertIn("No clear improvement", analysis["last_generation_verdict"])
        for i in range(5, 10):
            rec.agents[i]["children_by_comparison_age"] = 3
        self.assertIn("performed better", rec.generation_analysis()["last_generation_verdict"])

    def test_signed_negative_energy_predation_matches_engine_scoring(self):
        sim = fake_sim()
        rec = self.recorder(sim)
        rec.before_step([])
        sim.env.agents[0].energy = -2
        rec.on_event("death", agent=sim.env.agents[0], cause="predation", predator=None)
        sim.env.agents.clear()
        sim.env.time = 0.1
        sim.env.score = 0.12
        rec.after_step()
        rec.finish("extinction", make_plots=False)
        self.assertAlmostEqual(rec.summary["predation_loss"], -0.02)
        self.assertAlmostEqual(rec.summary["score_reconstruction_error"], 0)
        self.assertGreaterEqual(rec.summary["elapsed_run_score_upper_bound"], sim.env.score)

    def test_zero_tick_recap_and_unique_directories(self):
        first = self.recorder(fake_sim())
        second = self.recorder(fake_sim())
        self.assertNotEqual(first.directory, second.directory)
        first.finish("window_closed", make_plots=False)
        self.assertTrue((first.directory / "report.html").is_file())
        self.assertEqual(first.summary["final_score"], 0)
        self.assertIsNone(first.summary["elapsed_bound_percent"])
        self.assertEqual(first.finish("window_closed", make_plots=False), first.directory)
        data = json.loads((first.directory / "summary.json").read_text())
        self.assertEqual(data["end_reason"], "window_closed")

    def test_recording_does_not_change_simulation_trajectory(self):
        def run(record):
            policy = ExpertPolicy()
            sim = SimulationCore(seed=1, env_width=400, env_height=300,
                                 starting_agents=2, starting_predators=1,
                                 starting_fruits=8, starting_trees=5)
            rng_before = sim.rng.getstate()
            rec = self.recorder(sim) if record else None
            self.assertEqual(sim.rng.getstate(), rng_before)
            actions = []
            for _ in range(25):
                if rec:
                    rec.before_step(actions)
                state = sim.step(actions)
                if rec:
                    rec.after_step()
                actions = [(action.agent_id, action) for action in policy.actions_for_step(
                    state["observations"], sim_time=state["sim_time"]
                )]
            snapshot = (sim.env.time, sim.env.score, sim.rng.getstate(),
                        [(a.agent_id, a.x, a.y, a.energy) for a in sim.env.agents])
            if rec:
                rec.finish("time_limit", make_plots=False)
                self.assertAlmostEqual(rec.summary["score_reconstruction_error"], 0, places=9)
            return snapshot

        self.assertEqual(run(False), run(True))

    def test_window_close_and_keyboard_interrupt_save_partial_recaps(self):
        from local_playground import local_simulation

        for render, reason in ((True, "window_closed"), (False, "interrupted")):
            with self.subTest(reason=reason):
                sim = fake_sim()
                sim.step = MagicMock(side_effect=KeyboardInterrupt)
                display = MagicMock()
                display.QUIT = 256
                display.display.Info.return_value.current_h = 800
                display.event.get.return_value = [SimpleNamespace(type=256)]
                with patch("local_playground.SimulationCore", return_value=sim), \
                     patch("local_playground.pygame", display), \
                     patch("src.utils.run_report._make_plots"), redirect_stdout(io.StringIO()):
                    directory = local_simulation(verbose=render, seed=42, output_dir=self.temp.name)
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual(summary["end_reason"], reason)
                self.assertEqual(summary["simulated_seconds"], 0)
                self.assertTrue((directory / "report.html").is_file())
                if render:
                    sim.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()

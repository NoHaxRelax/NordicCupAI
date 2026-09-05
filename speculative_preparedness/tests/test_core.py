"""Tests for the layers that must not break: parsing, tracking, robustness."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actions
import belief
import capture
import mock
import policies
import schema

SCREENSHOT = {
    "game_status": "running",
    "score": 123.4,
    "agent_status": [
        {
            "agent_id": 1,
            "observations": [
                {"type": "tree", "distance": 12.5, "angle": 1.57},
                {"type": "predator", "distance": 30.0, "angle": -1.57, "rel_dir": -2.57},
                {"type": "edge", "coords": [[50.0, 50.0], [100.0, 100.0]]},
            ],
            "energy": 85.0,
            "max_energy": 500.0,
            "biome": "forest",
            "age": 5.2,
            "speed": 12.5,
            "sprint_speed": 13.5,
            "hearing_radius": 10.0,
            "vision_angle": 1.57,
            "vision_range": 50.0,
        }
    ],
}


class TestSchema(unittest.TestCase):
    def test_parses_the_screenshot(self):
        state = schema.GameState.parse(SCREENSHOT)
        self.assertTrue(state.is_running)
        self.assertEqual(state.score, 123.4)
        agent = state.agent(1)
        self.assertAlmostEqual(agent.energy_fraction, 0.17)
        self.assertEqual(len(agent.observations), 3)
        self.assertEqual(len(agent.observations_of("predator")), 1)

    def test_no_drift_on_known_payload(self):
        self.assertEqual(schema.schema_drift(SCREENSHOT), {})

    def test_survives_garbage(self):
        """A schema surprise must degrade, never raise."""
        for payload in (
            {},
            {"game_status": None, "agent_status": None},
            {"agent_status": [{"agent_id": "x", "observations": [None, 3, "s"]}]},
            {"agent_status": {"agent_id": 1}},  # unwrapped single agent
            {"agent_status": [{"agent_id": 1, "observations": [{"type": "wolf"}]}]},
        ):
            state = schema.GameState.parse(payload)
            self.assertIsInstance(state, schema.GameState)

    def test_reports_unknown_content(self):
        payload = {
            "game_status": "running",
            "weather": "rain",
            "agent_status": [
                {"agent_id": 1, "hunger": 3, "observations": [{"type": "food", "distance": 2}]}
            ],
        }
        drift = schema.schema_drift(payload)
        self.assertIn("weather", drift["state_keys"])
        self.assertIn("hunger", drift["agent_keys"])
        self.assertIn("food", drift["observation_types"])

    def test_missing_energy_is_not_starving(self):
        agent = schema.AgentStatus.parse({"agent_id": 1})
        self.assertEqual(agent.energy_fraction, 1.0)


class TestActions(unittest.TestCase):
    def test_turn_wraps_and_throttle_clamps(self):
        a = actions.Action(turn=3 * math.pi, throttle=5.0)
        self.assertLessEqual(abs(a.turn), math.pi)
        self.assertEqual(a.throttle, 1.0)

    def test_every_codec_encodes(self):
        payload = {1: actions.Action(turn=0.5, throttle=1.0, sprint=True)}
        for name in actions.CODECS:
            self.assertIsNotNone(actions.get_codec(name).encode(payload))


class TestBelief(unittest.TestCase):
    def _track(self, frames, dt=1.0):
        beliefs = belief.BeliefSet()
        for obs in frames:
            status = schema.AgentStatus.parse(
                {"agent_id": 1, "observations": obs, "energy": 400, "max_energy": 500,
                 "speed": 12.5, "sprint_speed": 13.5}
            )
            beliefs.update([status], dt=dt, commanded={1: actions.Action()})
        return beliefs[1]

    def test_remembers_predator_after_it_leaves_view(self):
        seen = [[{"type": "predator", "distance": d, "angle": 0.0}] for d in (30, 24, 18)]
        b = self._track(seen + [[], []])
        threat = b.nearest_threat()
        self.assertIsNotNone(threat, "predator forgotten the moment it left the cone")
        self.assertLess(threat.distance, 18.0, "dead reckoning did not advance the track")
        self.assertLess(threat.confidence, 1.0)

    def test_estimates_closing_speed(self):
        seen = [[{"type": "predator", "distance": d, "angle": 0.0}] for d in (30, 24, 18)]
        threat = self._track(seen).nearest_threat()
        self.assertGreater(threat.closing_speed(), 0.0)
        self.assertLess(threat.time_to_contact(), math.inf)

    def test_does_not_duplicate_a_stationary_object(self):
        seen = [[{"type": "tree", "distance": 10.0, "angle": 0.3}] for _ in range(6)]
        self.assertEqual(len(self._track(seen).of_type("tree")), 1)

    def test_forgets_eventually(self):
        seen = [[{"type": "predator", "distance": 30.0, "angle": 0.0}]]
        b = self._track(seen + [[]] * 40)
        self.assertEqual(len(b.of_type("predator")), 0)

    def test_threat_vector_accounts_for_flanking(self):
        obs = [
            {"type": "predator", "distance": 10.0, "angle": 0.7},
            {"type": "predator", "distance": 10.0, "angle": -0.7},
        ]
        vx, vy = self._track([obs]).threat_vector()
        self.assertGreater(vx, 0.0, "both predators ahead should push the vector forward")
        self.assertAlmostEqual(vy, 0.0, places=6, msg="symmetric flank should cancel laterally")


class TestPolicies(unittest.TestCase):
    def test_flees_a_predator_dead_ahead(self):
        state = schema.GameState.parse(
            {"game_status": "running", "agent_status": [
                {"agent_id": 1, "energy": 400, "max_energy": 500, "speed": 12.5,
                 "sprint_speed": 13.5,
                 "observations": [{"type": "predator", "distance": 8.0, "angle": 0.0}]}]}
        )
        for name in ("reactive", "belief"):
            action = policies.build(name).act(state)[1]
            self.assertGreater(abs(action.turn), math.pi / 2, f"{name} did not turn away")
            self.assertGreater(action.throttle, 0.5, f"{name} did not run")

    def test_sprints_on_first_sighting_at_close_range(self):
        state = schema.GameState.parse(
            {"game_status": "running", "agent_status": [
                {"agent_id": 1, "energy": 450, "max_energy": 500, "speed": 12.5,
                 "sprint_speed": 18.0,
                 "observations": [{"type": "predator", "distance": 6.0, "angle": 0.2}]}]}
        )
        self.assertTrue(policies.build("belief").act(state)[1].sprint)

    def test_conserves_energy_when_nearly_empty(self):
        state = schema.GameState.parse(
            {"game_status": "running", "agent_status": [
                {"agent_id": 1, "energy": 1.0, "max_energy": 500.0, "speed": 12.5,
                 "sprint_speed": 18.0, "observations": []}]}
        )
        self.assertEqual(policies.build("reactive").act(state)[1].throttle, 0.0)

    def test_handles_every_agent_in_a_multi_agent_state(self):
        state = schema.GameState.parse(
            {"game_status": "running", "agent_status": [
                {"agent_id": i, "energy": 400, "max_energy": 500, "observations": []}
                for i in (1, 2, 7)]}
        )
        self.assertEqual(set(policies.build("belief").act(state)), {1, 2, 7})


class TestRobustness(unittest.TestCase):
    def test_beats_noop_across_randomised_rules(self):
        noop = mock.evaluate(lambda: policies.build("noop"), n_draws=20, seed=7)
        reactive = mock.evaluate(lambda: policies.build("reactive"), n_draws=20, seed=7)
        self.assertGreater(reactive["median_survival_s"], noop["median_survival_s"])

    def test_no_policy_crashes_under_any_rule_draw(self):
        import random

        rng = random.Random(3)
        for _ in range(12):
            rules = mock.RuleDraw.sample(rng)
            for name in ("noop", "reactive", "belief"):
                mock.run_episode(policies.build(name), rules, seed=rng.randrange(1000), max_steps=60)


class TestCapture(unittest.TestCase):
    def test_roundtrip_and_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            policy = policies.build("reactive")
            with capture.Capture(path, policy="reactive", codec="continuous") as cap:
                state = schema.GameState.parse(SCREENSHOT)
                for _ in range(3):
                    acts = policy.act(state)
                    cap.record(SCREENSHOT, state=state, actions=acts,
                               wire=actions.get_codec("continuous").encode(acts))
            self.assertEqual(len(list(capture.read_frames(path))), 3)
            self.assertEqual(len(list(capture.transitions(path))), 2)
            self.assertEqual(capture.replay(path, policies.build("reactive"))["frames"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

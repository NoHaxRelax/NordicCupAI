import asyncio
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from agent_server import app, predict
from src.utils.DTOs import ObservationResponse, StepResponse
from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from src.utils.controllers.policy_inputs import prepare_inputs


def agent_state(observations=(), **stats):
    return {
        "agent_id": 7, "energy": 150.0, "biome": "grassland", "age": 5.0,
        "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
        "vision_angle": math.pi / 3, "vision_range": 200.0, "max_energy": 500.0,
        "observations": list(observations), **stats,
    }


def fruit(distance, angle=0.0, **extra):
    return {"type": "Fruit", "distance": distance, "angle": angle, **extra}


def predator(distance, angle=0.0):
    return {"type": "Predator", "distance": distance, "angle": angle}


class ExpertPolicyTests(unittest.TestCase):
    def setUp(self):
        # Keep behavior tests independent of the user's live tuning file.
        self.config = load_config(Path(__file__).parent / "fixtures" / "expert_policy.json")
        self.policy = ExpertPolicy(self.config)
        policy_patch = patch("agent_server.policy", self.policy)
        policy_patch.start()
        self.addCleanup(policy_patch.stop)

    def configured_policy(self, **sections):
        data = self.config.model_dump()
        for section, values in sections.items():
            data[section].update(values)
        return ExpertPolicy(ExpertConfig.model_validate(data))

    def test_inputs_keep_vectors_nearest_slots_and_all_stats(self):
        state = agent_state([
            fruit(30), fruit(10, math.pi / 2, ripeness=0.8), fruit(20),
            predator(110), predator(40, -math.pi / 2), predator(60),
            {"type": "Edge", "coords": [[0, 0], [1, 1]]},
        ])
        inputs = prepare_inputs(state, nearest_fruits=2, predator_danger_radius=100)
        self.assertEqual([f.distance for f in inputs.fruits], [10, 20])
        self.assertAlmostEqual(inputs.fruits[0].vector[0], 0)
        self.assertAlmostEqual(inputs.fruits[0].vector[1], 10)
        self.assertEqual(inputs.fruits[0].ripeness, 0.8)
        self.assertIsNone(inputs.fruits[1].ripeness)
        self.assertEqual(inputs.predator.distance, 40)
        self.assertAlmostEqual(inputs.predator.vector[1], -40)
        self.assertEqual(inputs.stats, {k: v for k, v in state.items()
                                       if k not in ("agent_id", "observations")})

    def test_ripest_is_selected_only_within_nearest_n(self):
        policy = self.configured_policy(perception={"nearest_fruits": 2})
        action = policy.action_decision(agent_state([
            fruit(15, 1.0, ripeness=0.9), fruit(5, -1.0, ripeness=0.2),
            fruit(20, 2.0, ripeness=1.0),
        ]))
        self.assertAlmostEqual(action.move_direction, 1.0)
        self.assertEqual(action.move_distance, 10.0)

    def test_missing_or_partial_ripeness_falls_back_to_nearest(self):
        for extra in ({}, {"ripeness": 1.0}):
            with self.subTest(extra=extra):
                action = self.policy.action_decision(agent_state([
                    fruit(30, 1.0, **extra), fruit(3, -1.0),
                ]))
                self.assertAlmostEqual(action.move_direction, -1.0)
                self.assertEqual(action.move_distance, 3.0)

    def test_equal_ripeness_prefers_nearest_and_order_is_stable(self):
        observations = [fruit(20, 1, ripeness=0.8), fruit(5, -1, ripeness=0.8)]
        first = self.policy.action_decision(agent_state(observations))
        second = self.policy.action_decision(agent_state(reversed(observations)))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.move_direction, -1)

    def test_nearest_predator_overrides_food_and_reproduction(self):
        for angle in (0, math.pi / 2, -math.pi / 2, math.pi, -2.8):
            with self.subTest(angle=angle):
                action = self.policy.action_decision(agent_state([
                    fruit(2, ripeness=1), predator(20, angle), predator(80, -angle),
                ], energy=400))
                self.assertAlmostEqual(math.cos(action.move_direction), -math.cos(angle))
                self.assertAlmostEqual(math.sin(action.move_direction), -math.sin(angle))
                self.assertEqual(action.move_distance, 20)
                self.assertFalse(action.spawn_agent)
                self.assertLessEqual(abs(action.turn_angle), self.config.movement.max_turn_angle)

    def test_predator_radius_is_inclusive_and_configurable(self):
        policy = self.configured_policy(
            perception={"predator_danger_radius": 25}, memory={"predator_escape_seconds": 0}
        )
        at_boundary = policy.action_decision(agent_state([predator(25), fruit(3)]))
        outside = policy.action_decision(agent_state([predator(25.1), fruit(3)]))
        self.assertAlmostEqual(abs(at_boundary.move_direction), math.pi)
        self.assertEqual(outside.move_direction, 0)
        self.assertEqual(outside.move_distance, 3)

    def test_low_energy_prevents_sprinting_and_mutated_speed_is_respected(self):
        low = self.policy.action_decision(agent_state([predator(20)], energy=99))
        at_boundary = self.policy.action_decision(agent_state([predator(20)], energy=100))
        self.policy.reset()
        mutated = self.policy.action_decision(agent_state([fruit(50)], speed=30, sprint_speed=15))
        self.assertEqual(low.move_distance, 10)
        self.assertEqual(at_boundary.move_distance, 20)
        self.assertEqual(mutated.move_distance, 15)

    def test_reproduction_threshold_reserve_age_and_enable_switch(self):
        self.assertFalse(self.policy.action_decision(agent_state(energy=200)).spawn_agent)
        self.assertTrue(self.policy.action_decision(agent_state(energy=201)).spawn_agent)
        policy = self.configured_policy(reproduction={
            "energy_threshold": 100, "minimum_energy_reserve": 75,
        })
        # Walking costs 0.5 and turning pi/4 costs 0.125 before spawning.
        state = agent_state([fruit(20, math.pi / 2)], energy=175.5)
        self.assertFalse(policy.action_decision(state).spawn_agent)
        self.assertTrue(policy.action_decision({**state, "energy": 175.625}).spawn_agent)
        young = self.configured_policy(reproduction={"minimum_age": 10})
        disabled = self.configured_policy(reproduction={"enabled": False})
        self.assertFalse(young.action_decision(agent_state(age=9, energy=400)).spawn_agent)
        self.assertFalse(disabled.action_decision(agent_state(energy=400)).spawn_agent)

    def test_exploration_and_food_at_zero_distance_produce_valid_actions(self):
        empty = agent_state()
        first = self.policy.action_decision(empty)
        self.assertEqual(first, self.policy.action_decision(empty))
        self.assertEqual(first.move_distance, 8)
        self.assertEqual(first.move_direction, 0)
        action = self.policy.action_decision(agent_state([fruit(0)]))
        self.assertEqual(action.move_distance, 0)
        self.assertEqual(action.turn_angle, 0)

    def test_custom_config_loads_and_invalid_parameters_fail(self):
        data = self.config.model_dump()
        data["perception"]["nearest_fruits"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_config(path).perception.nearest_fruits, 2)
        for section, field, value in (
            ("perception", "nearest_fruits", 0),
            ("perception", "nearest_fruits", 1.5),
            ("memory", "predator_escape_seconds", -1),
            ("movement", "walk_speed_fraction", 1.1),
            ("exploration", "turn_period_seconds", 0),
            ("reproduction", "energy_threshold", float("nan")),
            ("reproduction", "energy_threshhold", 200),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.configured_policy(**{section: {field: value}})

    def test_endpoint_handles_empty_status_births_and_repeated_games(self):
        def step(states):
            return StepResponse(game_status="ok", score=0, sim_time=0, n_agents=len(states),
                                agent_status=[ObservationResponse(**s) for s in states])
        self.assertEqual(predict(step([])), {"actions": []})
        original = predict(step([agent_state()]))
        newborn = predict(step([agent_state(agent_id=8, energy=75, age=0)]))
        self.assertEqual(newborn["actions"][0]["agent_id"], 8)
        self.assertFalse(newborn["actions"][0]["spawn_agent"])
        self.assertEqual(predict(step([agent_state()])), original)

    def test_predict_http_json_contract(self):
        payload = {
            "game_status": "ok", "score": 0, "sim_time": 0, "n_agents": 1,
            "agent_status": [agent_state([predator(20)], energy=400)],
        }

        async def request():
            messages = []

            async def receive():
                return {"type": "http.request", "body": json.dumps(payload).encode(),
                        "more_body": False}

            async def send(message):
                messages.append(message)

            await app({
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": "POST", "scheme": "http", "path": "/predict",
                "raw_path": b"/predict", "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1), "server": ("test", 80),
            }, receive, send)
            return messages

        messages = asyncio.run(request())
        self.assertEqual(messages[0]["status"], 200)
        body = json.loads(b"".join(m.get("body", b"") for m in messages))
        self.assertEqual(set(body), {"actions"})
        self.assertEqual(len(body["actions"]), 1)
        self.assertEqual(body["actions"][0]["agent_id"], 7)
        self.assertFalse(body["actions"][0]["spawn_agent"])
        self.assertEqual(body["actions"][0]["move_distance"], 20)


if __name__ == "__main__":
    unittest.main()

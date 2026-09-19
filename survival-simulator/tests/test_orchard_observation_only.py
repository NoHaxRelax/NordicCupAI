"""Tests for Orchard's public observation transport; no engine imports."""
import copy
import math
import os
import unittest

from models.observation_only import (
    ACTION_FIELDS, AGENT_FIELDS, OBSERVATION_FIELDS,
    ObservationOnlyOrchard, sanitize_states,
)


def agent_state():
    return {
        "agent_id": 1, "energy": 150.0, "biome": "forest", "age": 0.0,
        "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
        "vision_angle": math.pi / 3, "vision_range": 200.0, "max_energy": 500.0,
        "observations": [],
    }


class ObservationSanitizationTests(unittest.TestCase):
    def test_strips_privileged_fields_at_every_level_and_copies_coordinates(self):
        state = agent_state()
        observations = [
            {"type": "Fruit", "distance": 40.0, "angle": 0.2, "age": 19, "energy": 58},
            {"type": "Tree", "distance": 70.0, "angle": 0.3, "age": 25, "death_age": 90},
            {"type": "Agent", "distance": 15.0, "angle": 0.0, "id": 2, "rel_dir": 0.5,
             "energy": 200, "max_age": 100, "world_position": [900, 500]},
            {"type": "Predator", "distance": 90.0, "angle": 0.7, "rel_dir": 1.0, "id": 3},
            {"type": "Edge", "coords": [[-10, -20], [10, -20]], "world_coords": object()},
        ]
        state.update({"x": 100, "y": 200, "direction": 1.0, "world_seed": 42,
                      "max_age": 80, "world": object(), "observations": observations})
        clean = sanitize_states([state])[0]
        self.assertEqual(set(clean), set(AGENT_FIELDS))
        for observation in clean["observations"]:
            self.assertEqual(set(observation), set(OBSERVATION_FIELDS[observation["type"]]))
        observations[-1]["coords"][0][0] = 9999
        self.assertEqual(clean["observations"][-1]["coords"][0][0], -10)

    def test_rejects_unknown_types_missing_fields_and_nonfinite_values(self):
        malformed = [
            {"type": "HiddenWorld", "objects": []},
            {"type": "Fruit", "angle": 0.0},
            {"type": "Fruit", "distance": float("nan"), "angle": 0.0},
            {"type": "Agent", "distance": 10.0, "angle": 0.0, "id": 2},
            {"type": "Edge", "coords": [[1, 2, 3], [4, 5]]},
        ]
        for observation in malformed:
            with self.subTest(observation=observation):
                state = agent_state()
                state["observations"] = [observation]
                with self.assertRaises(ValueError):
                    sanitize_states([state])
        state = agent_state()
        del state["speed"]
        with self.assertRaises(ValueError):
            sanitize_states([state])
        with self.assertRaises(ValueError):
            sanitize_states([agent_state(), agent_state()])


class ObservationWorkerTests(unittest.TestCase):
    def test_real_process_actions_ignore_injected_hidden_data(self):
        clean = agent_state()
        clean["observations"] = [
            {"type": "Tree", "distance": 80.0, "angle": 0.2},
            {"type": "Fruit", "distance": 40.0, "angle": 0.1},
        ]
        injected = copy.deepcopy(clean)
        injected.update({"x": 900, "y": 800, "world_seed": 9001,
                         "max_age": 61, "unseen_fruit": object()})
        for observation in injected["observations"]:
            observation.update({"age": 45, "energy": 60, "id": 999, "world_x": 900})
        with ObservationOnlyOrchard(policy_seed=17) as left:
            with ObservationOnlyOrchard(policy_seed=17) as right:
                self.assertNotEqual(left.audit["worker_pid"], os.getpid())
                self.assertEqual(left.audit["multiprocessing_start_method"], "spawn")
                self.assertFalse(left.audit["env_loaded"])
                self.assertEqual(left.audit["policy_seed"], 17)
                for t in (0.0, 0.1, 0.2):
                    clean["age"] = injected["age"] = t
                    actions = left([clean], t)
                    self.assertEqual(actions, right([injected], t))
                    self.assertEqual(actions[0][0], 1)
                    self.assertEqual(set(actions[0][1]), set(ACTION_FIELDS))
                    self.assertFalse(left.last_audit["env_loaded"])
        self.assertFalse(left._process.is_alive())
        self.assertFalse(right._process.is_alive())
        left.close()
        with self.assertRaises(RuntimeError):
            left([clean], 1.0)

    def test_invalid_worker_configuration_returns_error_without_hanging(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown policy parameters"):
            ObservationOnlyOrchard(policy_kwargs={"world_seed": 42})

    def test_empty_public_input_is_allowed(self):
        with ObservationOnlyOrchard() as actor:
            self.assertEqual(actor([], 0.0), [])


if __name__ == "__main__":
    unittest.main()

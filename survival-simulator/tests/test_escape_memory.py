import math
from pathlib import Path
import unittest
from unittest.mock import patch

from agent_server import predict
from src.utils.DTOs import StepResponse
from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from test_expert_policy import agent_state, fruit, predator


class EscapeMemoryTests(unittest.TestCase):
    def policy(self, duration=2.0, max_turn=math.pi / 4):
        data = load_config(Path(__file__).parent / "fixtures" / "expert_policy.json").model_dump()
        data["memory"]["predator_escape_seconds"] = duration
        data["movement"]["max_turn_angle"] = max_turn
        return ExpertPolicy(ExpertConfig.model_validate(data))

    def test_escape_heading_stays_fixed_while_agent_turns(self):
        policy = self.policy()
        heading = 0.7
        escape_heading = heading - 0.2 + math.pi
        for tick in range(12):
            observations = [predator(20, -0.2)] if tick == 0 else [fruit(1, 0.5)]
            action = policy.action_decision(
                agent_state(observations, energy=400, age=10 + tick * 0.1),
                sim_time=10 + tick * 0.1,
            )
            self.assertAlmostEqual(math.cos(heading + action.move_direction), math.cos(escape_heading))
            self.assertAlmostEqual(math.sin(heading + action.move_direction), math.sin(escape_heading))
            self.assertEqual(action.move_distance, 20)
            self.assertFalse(action.spawn_agent)
            heading += action.turn_angle

    def test_timer_starts_when_threat_disappears_and_expires_on_boundary(self):
        policy = self.policy(duration=0.3, max_turn=0)
        policy.action_decision(agent_state([predator(20)], energy=400), sim_time=10)
        for time in (10.1, 10.2, 10.3):
            action = policy.action_decision(agent_state([fruit(3, 1)], energy=400), sim_time=time)
            self.assertEqual(action.move_distance, 20)
            self.assertFalse(action.spawn_agent)
        expired = policy.action_decision(agent_state([fruit(3, 1)], energy=400), sim_time=10.4)
        self.assertEqual(expired.move_distance, 3)
        self.assertAlmostEqual(expired.move_direction, 1)
        self.assertTrue(expired.spawn_agent)

    def test_new_sighting_refreshes_direction_and_timer(self):
        policy = self.policy(duration=0.3, max_turn=0)
        policy.action_decision(agent_state([predator(20)]), sim_time=0)
        policy.action_decision(agent_state(), sim_time=0.1)
        seen = policy.action_decision(agent_state([predator(20, math.pi / 2)]), sim_time=0.2)
        self.assertAlmostEqual(seen.move_direction, -math.pi / 2)
        policy.action_decision(agent_state(), sim_time=0.3)
        remembered = policy.action_decision(agent_state(), sim_time=0.5)
        self.assertAlmostEqual(remembered.move_direction, -math.pi / 2)
        self.assertEqual(remembered.move_distance, 20)
        expired = policy.action_decision(agent_state([fruit(3)]), sim_time=0.6)
        self.assertEqual(expired.move_distance, 3)

    def test_zero_timeout_disables_memory_but_still_flees_visible_predator(self):
        policy = self.policy(duration=0)
        seen = policy.action_decision(agent_state([predator(20)]), sim_time=0)
        lost = policy.action_decision(agent_state([fruit(3)]), sim_time=0.1)
        self.assertEqual(seen.move_distance, 20)
        self.assertEqual(lost.move_distance, 3)

    def test_low_energy_memory_uses_walking_speed(self):
        policy = self.policy(max_turn=0)
        policy.action_decision(agent_state([predator(20)], energy=99), sim_time=0)
        remembered = policy.action_decision(agent_state([fruit(1)], energy=99), sim_time=0.1)
        self.assertEqual(remembered.move_distance, 10)
        self.assertAlmostEqual(abs(remembered.move_direction), math.pi)
        self.assertFalse(remembered.spawn_agent)

    def test_memory_is_per_agent_and_dead_agents_are_pruned(self):
        policy = self.policy()
        actions = policy.actions_for_step([
            agent_state([predator(20)], agent_id=7),
            agent_state([fruit(3)], agent_id=8),
        ], sim_time=0)
        self.assertEqual([action.move_distance for action in actions], [20, 3])
        policy.actions_for_step([agent_state(agent_id=8)], sim_time=0.1)
        self.assertNotIn(7, policy._escape_memories)
        reused = policy.actions_for_step([agent_state([fruit(3)], agent_id=7)], sim_time=0.2)
        self.assertEqual(reused[0].move_distance, 3)

    def test_simulation_clock_controls_expiry_even_if_agent_age_stalls(self):
        policy = self.policy(duration=0.3)
        policy.actions_for_step([agent_state([predator(20)], age=5)], sim_time=50)
        remembered = policy.actions_for_step([agent_state([fruit(3)], age=5)], sim_time=50.1)
        expired = policy.actions_for_step([agent_state([fruit(3)], age=5)], sim_time=50.4)
        self.assertEqual(remembered[0].move_distance, 20)
        self.assertEqual(expired[0].move_distance, 3)

    def test_direct_calls_can_use_age_and_reused_younger_ids_start_fresh(self):
        policy = self.policy(duration=0.3)
        policy.action_decision(agent_state([predator(20)], age=5))
        remembered = policy.action_decision(agent_state([fruit(3)], age=5.1))
        expired = policy.action_decision(agent_state([fruit(3)], age=5.4))
        self.assertEqual(remembered.move_distance, 20)
        self.assertEqual(expired.move_distance, 3)
        policy.action_decision(agent_state([predator(20)], age=6))
        newborn = policy.action_decision(agent_state([fruit(3)], age=0))
        self.assertEqual(newborn.move_distance, 3)

    def test_retried_endpoint_requests_do_not_apply_turn_twice(self):
        policy = self.policy()

        def request(time, observations):
            return StepResponse(game_status="ok", score=0, sim_time=time, n_agents=1,
                                agent_status=[agent_state(observations)])

        with patch("agent_server.policy", policy):
            first_request = request(0, [predator(20)])
            first = predict(first_request)
            self.assertEqual(predict(first_request), first)
            lost_request = request(0.1, [fruit(3)])
            lost = predict(lost_request)
            self.assertEqual(predict(lost_request), lost)
            next_action = predict(request(0.2, [fruit(3)]))["actions"][0]
        heading = first["actions"][0]["turn_angle"] + lost["actions"][0]["turn_angle"]
        self.assertAlmostEqual(math.cos(heading + next_action["move_direction"]), -1)
        self.assertAlmostEqual(math.sin(heading + next_action["move_direction"]), 0)

    def test_new_episode_empty_population_and_game_over_clear_memory(self):
        for boundary in ("rewind", "empty", "game_over", "reset"):
            with self.subTest(boundary=boundary):
                policy = self.policy()
                policy.actions_for_step([agent_state([predator(20)])], sim_time=10)
                next_time = 10.2
                if boundary == "rewind":
                    next_time = 0
                elif boundary == "empty":
                    self.assertEqual(policy.actions_for_step([], sim_time=10.1), [])
                elif boundary == "game_over":
                    self.assertEqual(policy.actions_for_step(
                        [agent_state()], sim_time=10.1, game_status="game_over"
                    ), [])
                else:
                    policy.reset()
                safe = policy.actions_for_step([agent_state([fruit(3)])], sim_time=next_time)
                self.assertEqual(safe[0].move_distance, 3)


if __name__ == "__main__":
    unittest.main()

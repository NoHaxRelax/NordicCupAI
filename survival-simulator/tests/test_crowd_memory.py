import math
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from agent_server import predict
from src.utils.DTOs import StepResponse
from src.utils.controllers.crowd_memory import CrowdTracker, crowd_direction, wrap
from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from src.utils.controllers.global_planner import load_planner_config
from src.utils.controllers.policy_inputs import RememberedNeighbor
from test_expert_policy import agent_state, fruit, predator


FIXTURES = Path(__file__).parent / "fixtures"


def config(**overrides):
    data = load_config(FIXTURES / "expert_policy.json").model_dump()
    data["crowd"].update(enabled=True, **overrides)
    return ExpertConfig.model_validate(data)


def neighbor(agent_id=8, distance=20, angle=0):
    return {"type": "Agent", "id": agent_id, "distance": distance, "angle": angle}


def policy(**overrides):
    planner = load_planner_config(FIXTURES / "global_planner.json").model_copy(update={"enabled": False})
    return ExpertPolicy(config(**overrides), planner)


class CrowdMemoryTests(unittest.TestCase):
    def test_repeated_sightings_count_unique_agents_and_refresh_independently(self):
        tracker = CrowdTracker(config().crowd)
        seen = tracker.observe(agent_state([neighbor(), neighbor(), neighbor(9)]), 0)
        self.assertEqual([item.agent_id for item in seen], [8, 9])
        tracker.observe(agent_state([neighbor()]), 3)
        remembered = tracker.observe(agent_state(), 4)
        self.assertEqual([item.agent_id for item in remembered], [8])
        self.assertEqual(tracker.observe(agent_state(), 7), ())

    def test_weight_fades_with_time_and_distance_and_fresh_sighting_replaces_bearing(self):
        tracker = CrowdTracker(config().crowd)
        first = tracker.observe(agent_state([neighbor(distance=30), neighbor(9, distance=90)]), 0)
        self.assertAlmostEqual(first[0].weight, 0.75)
        self.assertAlmostEqual(first[1].weight, 0.25)
        later = tracker.observe(agent_state(), 2)
        self.assertAlmostEqual(later[0].weight, first[0].weight / 2)
        refreshed = tracker.observe(agent_state([neighbor(angle=math.pi / 2)]), 3)
        self.assertAlmostEqual(refreshed[0].angle, math.pi / 2)
        self.assertGreater(refreshed[0].weight, later[0].weight)
        tracker.observe(agent_state([neighbor(distance=200)]), 3.1)
        self.assertNotIn(8, tracker.memories[7].sightings)

    def test_remembered_directions_follow_turns_once_per_tick(self):
        tracker = CrowdTracker(config().crowd)
        tracker.observe(agent_state([neighbor(angle=0.3)]), 0)
        tracker.remember_turn(7, math.pi / 2)
        first = tracker.observe(agent_state(), 0.1)
        self.assertAlmostEqual(first[0].angle, wrap(0.3 - math.pi / 2))
        self.assertEqual(tracker.observe(agent_state(), 0.1), first)
        tracker.remember_turn(7, -math.pi / 4)
        second = tracker.observe(agent_state(), 0.2)
        self.assertAlmostEqual(second[0].angle, wrap(0.3 - math.pi / 4))

    def test_sightings_are_local_bounded_and_do_not_include_self_or_anonymous_agents(self):
        tracker = CrowdTracker(config(max_remembered_neighbors=2).crowd)
        observations = [neighbor(), neighbor(9, distance=100), neighbor(10, distance=1),
                        neighbor(7), {"type": "Agent", "distance": 1, "angle": 0}, fruit(1)]
        kept = tracker.observe(agent_state(observations), 0)
        self.assertEqual([item.agent_id for item in kept], [8, 10])
        self.assertEqual(tracker.observe(agent_state(agent_id=11), 0), ())
        tracker.prune({11})
        self.assertEqual(set(tracker.memories), {11})

    def test_reused_ids_and_clock_rewinds_clear_history(self):
        for reset in ("age", "clock"):
            tracker = CrowdTracker(config().crowd)
            tracker.observe(agent_state([neighbor()], age=5), 10)
            result = tracker.observe(agent_state(age=0 if reset == "age" else 5),
                                     10.1 if reset == "age" else 0)
            self.assertEqual(result, ())

    def test_zero_memory_disables_tracking_and_invalid_settings_fail(self):
        tracker = CrowdTracker(config(memory_seconds=0).crowd)
        self.assertEqual(tracker.observe(agent_state([neighbor()]), 0), ())
        self.assertEqual(tracker.memories, {})
        for key, value in (("memory_seconds", -1), ("neighbor_radius", 0),
                           ("angular_spread", 0), ("direction_bins", 3),
                           ("foraging_strength", 1.1), ("max_remembered_neighbors", 0)):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                config(**{key: value})


class CrowdSteeringTests(unittest.TestCase):
    def test_crowd_ahead_selects_an_open_direction_behind(self):
        neighbors = (RememberedNeighbor(8, 0, 20, 1), RememberedNeighbor(9, 0.1, 20, 1))
        direction, strength = crowd_direction(neighbors, 0, config().crowd, 7)
        self.assertGreater(abs(direction), math.pi / 2)
        self.assertGreater(strength, 0)

    def test_opposite_crowds_choose_sideways_instead_of_cancelling(self):
        neighbors = (RememberedNeighbor(8, 0, 20, 1), RememberedNeighbor(9, math.pi, 20, 1))
        direction, _ = crowd_direction(neighbors, 0, config(tie_score_fraction=0).crowd, 7)
        self.assertAlmostEqual(abs(direction), math.pi / 2)

    def test_uniform_crowding_does_not_invent_an_open_direction(self):
        neighbors = tuple(RememberedNeighbor(index, math.tau * index / 24, 20, 1) for index in range(24))
        self.assertIsNone(crowd_direction(neighbors, 0, config().crowd, 7))
        self.assertIsNone(crowd_direction((), 0, config().crowd, 7))

    def test_more_remembered_neighbors_increase_bias_and_fading_reduces_it(self):
        one = (RememberedNeighbor(8, 0, 20, 0.5),)
        two = one + (RememberedNeighbor(9, 0, 20, 0.5),)
        faded = (RememberedNeighbor(8, 0, 20, 0.25),)
        args = (0, config().crowd, 7)
        self.assertGreater(crowd_direction(two, *args)[1], crowd_direction(one, *args)[1])
        self.assertLess(crowd_direction(faded, *args)[1], crowd_direction(one, *args)[1])


class CrowdPolicyTests(unittest.TestCase):
    def test_push_deflects_without_sprinting_and_is_gentler_while_foraging(self):
        exploring = policy().action_decision(agent_state([neighbor()]))
        foraging = policy().action_decision(agent_state([neighbor(), fruit(100)]))
        self.assertGreater(abs(exploring.move_direction), 0)
        self.assertGreater(abs(exploring.move_direction), abs(foraging.move_direction))
        self.assertLess(abs(exploring.move_direction), math.pi / 2)
        self.assertEqual(exploring.move_distance, 8)
        self.assertEqual(foraging.move_distance, 10)
        arrived = policy().action_decision(agent_state([neighbor(), fruit(0)]))
        self.assertEqual(arrived.move_distance, 0)

    def test_crowd_fades_out_and_returns_to_local_expert(self):
        biased = policy()
        biased.action_decision(agent_state([neighbor()]), sim_time=0)
        remembered = biased.action_decision(agent_state(), sim_time=0.1)
        self.assertGreater(abs(remembered.move_direction), 0)
        expired = biased.action_decision(agent_state(), sim_time=4)
        self.assertEqual(expired, policy().action_decision(agent_state(), sim_time=4))

    def test_visible_and_remembered_escape_override_crowd(self):
        biased, baseline = policy(), policy(exploration_strength=0, foraging_strength=0)
        for tick in range(12):
            observations = [neighbor(), fruit(10)]
            if tick == 0:
                observations.append(predator(20, 0.3))
            state = agent_state(observations, energy=400, age=5 + tick / 10)
            expected = baseline.action_decision(state, sim_time=tick / 10)
            actual = biased.action_decision(state, sim_time=tick / 10)
            self.assertEqual(actual, expected)
            self.assertFalse(actual.spawn_agent)

    def test_reproduction_reserve_accounts_for_extra_turn(self):
        data = config(foraging_strength=1, full_strength_neighbors=1).model_dump()
        data["reproduction"]["minimum_energy_reserve"] = 100
        controller = ExpertPolicy(ExpertConfig.model_validate(data), policy().planner.config)
        state = agent_state([fruit(100)], energy=200.6)
        self.assertTrue(controller.action_decision(state).spawn_agent)
        state["observations"].append(neighbor(distance=1))
        self.assertFalse(controller.action_decision(state, sim_time=6).spawn_agent)

    def test_batch_and_endpoint_retry_cleanup_and_episode_boundaries(self):
        for boundary in ("game_over", "empty", "rewind", "reset"):
            controller = policy()
            request = StepResponse(game_status="ok", score=0, sim_time=10, n_agents=1,
                                   agent_status=[agent_state([neighbor()])])
            with patch("agent_server.policy", controller):
                first = predict(request)
                self.assertEqual(predict(request), first)
                lost = StepResponse.model_validate({**request.model_dump(), "sim_time": 10.1,
                                                    "agent_status": [agent_state()]})
                second = predict(lost)
                self.assertEqual(predict(lost), second)
            if boundary == "game_over":
                controller.actions_for_step([agent_state()], 10.2, "game_over")
            elif boundary == "empty":
                controller.actions_for_step([], 10.2)
            elif boundary == "rewind":
                controller.actions_for_step([agent_state()], 0)
            else:
                controller.reset()
            action = controller.actions_for_step([agent_state()], 0.1)[0]
            self.assertEqual(action.move_direction, 0)
            controller.actions_for_step([agent_state(agent_id=9)], 0.2)
            self.assertNotIn(7, controller.crowd_tracker.memories)
            self.assertEqual(controller.planner.estimator.poses, {})


if __name__ == "__main__":
    unittest.main()

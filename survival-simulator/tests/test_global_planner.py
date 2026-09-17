import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from agent_server import predict
from src.utils.DTOs import ActionRequest, StepResponse
from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.global_planner import (
    FoodGrid, GlobalPlanner, GroupPlan, PlannerConfig, Section, food_grid,
    load_planner_config, partition_food,
)
from src.utils.controllers.policy_inputs import SectionHint
from src.utils.controllers.world_estimator import (
    EstimatedPose, MapGroup, TreeLandmark, WorldEstimator, wrap,
)
from test_expert_policy import agent_state, fruit, predator


FIXTURES = Path(__file__).parent / "fixtures"


def planner_config(**changes):
    data = load_planner_config(FIXTURES / "global_planner.json").model_dump()
    for key, value in changes.items():
        if isinstance(value, dict):
            data[key].update(value)
        else:
            data[key] = value
    return PlannerConfig.model_validate(data)


def tree(x, y):
    return {"type": "Tree", "distance": math.hypot(x, y), "angle": math.atan2(y, x)}


def sighting(agent_id, x, y, heading=0):
    angle = math.atan2(y, x)
    return {"type": "Agent", "id": agent_id, "distance": math.hypot(x, y),
            "angle": angle, "rel_dir": wrap(angle + math.pi - heading)}


def action(agent_id=7, distance=10, direction=0, turn=0):
    return ActionRequest(agent_id=agent_id, move_distance=distance,
                         move_direction=direction, turn_angle=turn, spawn_agent=False)


class WorldEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = WorldEstimator(planner_config().estimator)

    def test_disconnected_agents_have_separate_frames(self):
        self.estimator.update([agent_state(agent_id=1), agent_state(agent_id=2)], 0)
        self.assertEqual(set(self.estimator.groups), {1, 2})
        self.assertNotEqual(self.estimator.poses[1].group_id, self.estimator.poses[2].group_id)

    def test_agent_sighting_merges_both_position_and_heading_in_either_direction(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.estimator.reset()
                states = [agent_state(agent_id=1), agent_state(agent_id=2)]
                # Agent 2 is at (100, 20), facing +pi/2 in agent 1's frame.
                if reverse:
                    states[1]["observations"] = [sighting(1, -20, 100, -math.pi / 2)]
                else:
                    states[0]["observations"] = [sighting(2, 100, 20, math.pi / 2)]
                self.estimator.update(states, 0)
                self.assertEqual(set(self.estimator.groups), {1})
                np.testing.assert_allclose(self.estimator.poses[2].position, [100, 20], atol=1e-9)
                self.assertAlmostEqual(self.estimator.poses[2].heading, math.pi / 2)

    def test_merging_rotates_existing_landmarks_and_visited_positions(self):
        self.estimator.update([agent_state(agent_id=1), agent_state([tree(10, 0)], agent_id=2)], 0)
        self.estimator.update([
            agent_state([sighting(2, 100, 20, math.pi / 2)], agent_id=1), agent_state(agent_id=2),
        ], 0.1)
        np.testing.assert_allclose(self.estimator.groups[1].trees[0].position, [100, 30], atol=1e-9)
        self.assertTrue(any(np.allclose(point, [100, 20]) for point, _, _ in self.estimator.groups[1].visited.values()))

    def test_odometry_moves_before_turning_and_applies_biome_penalty_once_per_tick(self):
        state = agent_state(biome="swamp")
        self.estimator.update([state], 0)
        self.estimator.remember_actions([action(turn=math.pi / 2)])
        self.estimator.update([state], 0.1)
        np.testing.assert_allclose(self.estimator.poses[7].position, [5, 0], atol=1e-9)
        self.estimator.update([state], 0.1)
        np.testing.assert_allclose(self.estimator.poses[7].position, [5, 0], atol=1e-9)
        self.estimator.remember_actions([action()])
        self.estimator.update([state], 0.2)
        np.testing.assert_allclose(self.estimator.poses[7].position, [5, 5], atol=1e-9)

    def test_two_static_trees_correct_blocked_movement_without_counting_duplicates(self):
        trees = [tree(30, 0), tree(0, 40)]
        self.estimator.update([agent_state(trees)], 0)
        self.estimator.remember_actions([action(distance=10)])
        # The commanded movement was blocked; the sightings remain unchanged.
        self.estimator.update([agent_state(trees)], 0.1)
        np.testing.assert_allclose(self.estimator.poses[7].position, [0, 0], atol=1e-9)
        self.assertEqual(len(self.estimator.groups[7].trees), 2)
        self.assertEqual(self.estimator.poses[7].uncertainty, 1)

    def test_uncertain_agents_stop_extending_the_map_and_dead_maps_are_pruned(self):
        self.estimator.update([agent_state()], 0)
        self.estimator.poses[7].uncertainty = 100
        self.estimator.update([agent_state([tree(40, 0)])], 0.1)
        self.assertEqual(self.estimator.groups[7].trees, [])
        self.estimator.update([agent_state(agent_id=8)], 0.2)
        self.assertEqual(set(self.estimator.poses), {8})
        self.assertEqual(set(self.estimator.groups), {8})

    def test_stale_trees_and_history_are_bounded(self):
        estimator = WorldEstimator(planner_config(estimator={
            "max_trees_per_group": 2, "max_visited_cells_per_group": 1,
        }).estimator)
        estimator.update([agent_state([tree(30, 0), tree(0, 40), tree(-30, 0)])], 0)
        self.assertEqual(len(estimator.groups[7].trees), 2)
        estimator.update([agent_state()], 61)
        self.assertEqual(estimator.groups[7].trees, [])
        self.assertEqual(len(estimator.groups[7].visited), 1)


class PartitionTests(unittest.TestCase):
    def test_dense_food_gets_less_area_with_equal_total_weights(self):
        weights = np.array([[4., 4., 1., 1., 1., 1., 1., 1., 1., 1.]])
        grid = FoodGrid(np.zeros(2), 10, weights, np.ones_like(weights, dtype=bool))
        sections = partition_food(grid, 2)
        self.assertEqual([s.food_weight for s in sections], [8, 8])
        self.assertEqual(sections[0].bounds, (0, 0, 20, 10))
        self.assertEqual(sections[1].bounds, (20, 0, 100, 10))

    def test_sparse_and_zero_food_partitions_cover_each_known_cell_once(self):
        rng = np.random.default_rng(3)
        # This six-cell cycle cannot be split into 3+3 by an axis-aligned cut.
        cycle = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool)
        masks = [cycle] + [rng.random((6, 7)) > 0.6 for _ in range(10)]
        for known in masks:
            for food in (known.astype(float), np.zeros_like(known, dtype=float)):
                grid = FoodGrid(np.zeros(2), 1, food, known)
                for requested in (1, 3, int(known.sum()), 100):
                    sections = partition_food(grid, requested)
                    self.assertEqual(len(sections), min(requested, int(known.sum())))
                    self.assertAlmostEqual(sum(s.food_weight for s in sections), float(food.sum()))
                    for y, x in zip(*np.nonzero(known)):
                        owners = [s for s in sections if s.bounds[0] <= x + 0.5 < s.bounds[2]
                                  and s.bounds[1] <= y + 0.5 < s.bounds[3]]
                        self.assertEqual(len(owners), 1)
                    for section in sections:
                        x, y = np.floor(section.center).astype(int)
                        self.assertTrue(known[y, x])

    def test_grid_uses_observed_food_priors_and_respects_memory_budget(self):
        group = MapGroup(1, trees=[TreeLandmark(np.array([0., 0.]), 0)])
        group.visited[(10, 10)] = (np.array([1000., 1000.]), 0.1, 0)
        grid = food_grid(group, planner_config(max_grid_cells=4))
        self.assertLessEqual(grid.weights.size, 4)
        self.assertAlmostEqual(float(grid.weights.sum()), 1.01)
        self.assertLessEqual(int(grid.known.sum()), 2)
        self.assertIsNone(food_grid(MapGroup(2), planner_config()))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = GlobalPlanner(planner_config())
        self.states = [
            agent_state([tree(0, 10), tree(200, 10), sighting(8, 200, 0)], agent_id=7),
            agent_state(agent_id=8),
        ]

    def test_assignments_survive_motion_and_retries_then_clean_up_on_death_and_reset(self):
        self.planner.instructions(self.states, 0)
        plan = self.planner.plans[7]
        self.assertEqual(len(plan.sections), 2)
        self.assertEqual(len(set(plan.assignments.values())), 2)
        original = dict(plan.assignments)
        # Swap estimated positions while preserving ownership until a replan.
        first, second = self.planner.estimator.poses[7], self.planner.estimator.poses[8]
        first.position, second.position = second.position.copy(), first.position.copy()
        states = [agent_state(agent_id=7), agent_state(agent_id=8)]
        hints = self.planner.instructions(states, 0.1)
        self.assertEqual(plan.assignments, original)
        self.assertEqual(set(hints), {7, 8})
        self.assertEqual(self.planner.instructions(states, 0.1), hints)
        self.planner.instructions(states, 10.1)
        self.assertEqual(self.planner.plans[7].assignments, original)
        self.planner.instructions(states[:1], 10.2)
        self.assertEqual(set(self.planner.plans[7].assignments), {7})
        self.assertEqual(self.planner.hints, {})  # An empty section exerts no push.
        self.planner.instructions(states[:1], 0)
        self.assertEqual(self.planner.estimator.groups[7].trees, [])
        self.planner.instructions([], 0.1)
        self.assertEqual(self.planner.plans, {})
        self.assertEqual(self.planner.estimator.poses, {})

    def test_newborn_is_assigned_immediately_and_ownership_stays_stable(self):
        self.planner.instructions(self.states, 0)
        states = [agent_state([sighting(9, 10, 0)], agent_id=7),
                  agent_state(agent_id=8), agent_state(agent_id=9)]
        self.planner.instructions(states, 0.1)
        self.assertEqual(set(self.planner.plans[7].assignments), {7, 8, 9})
        # Merging a new agent's frame replans immediately; sections remain
        # bounded by available known cells and can be shared when necessary.
        for tick in (0.2, 0.3):
            original = dict(self.planner.plans[7].assignments)
            self.planner.instructions(states, tick)
            self.assertEqual(self.planner.plans[7].assignments, original)

    def test_push_requires_other_occupied_center_to_be_closer_beyond_uncertainty(self):
        sections = [Section(0, (0, 0, 100, 100), np.array([0., 0.]), 1),
                    Section(1, (100, 0, 200, 100), np.array([100., 0.]), 1)]
        plan = GroupPlan(sections, 0, 0, {7: 0, 8: 1})
        pose = EstimatedPose(7, 7, np.array([90., 0.]), heading=math.pi / 2)
        hint = self.planner._hint(pose, plan)
        np.testing.assert_allclose(hint.vector, [0, 90], atol=1e-9)
        self.assertEqual(hint.strength, 0.15)
        for position, uncertainty in ((49, 0), (52, 0), (65, 15), (90, 31)):
            pose.position = np.array([float(position), 0])
            pose.uncertainty = uncertainty
            self.assertIsNone(self.planner._hint(pose, plan))
        pose.position, pose.uncertainty = np.array([90., 0.]), 0
        plan.assignments = {7: 0}
        self.assertIsNone(self.planner._hint(pose, plan))

    def test_disabled_planner_does_no_mapping_and_config_is_validated(self):
        planner = GlobalPlanner(planner_config(enabled=False))
        self.assertEqual(planner.instructions(self.states, 0), {})
        self.assertEqual(planner.estimator.poses, {})
        for field, value in (("push_strength", -0.1), ("push_strength", 1.1),
                             ("grid_cell_size", 0), ("max_sections", 0), ("typo", True)):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                planner_config(**{field: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner.json"
            path.write_text(planner_config(enabled=False).model_dump_json(), encoding="utf-8")
            with patch.dict("os.environ", {"GLOBAL_PLANNER_CONFIG": str(path)}):
                self.assertFalse(load_planner_config().enabled)
                self.assertTrue(load_planner_config(FIXTURES / "global_planner.json").enabled)

    def test_snapshot_contains_only_serializable_estimates(self):
        self.planner.instructions(self.states, 0)
        snapshot = json.loads(json.dumps(self.planner.snapshot(), allow_nan=False))
        self.assertEqual(len(snapshot["agents"]), 2)
        self.assertEqual(len(snapshot["groups"]), 1)

    def test_map_and_assignment_are_independent_of_observation_and_population_order(self):
        other = GlobalPlanner(planner_config())
        for tick in (0, 0.1, 10.1):
            reordered = [dict(state, observations=list(reversed(state["observations"])))
                         for state in reversed(self.states)]
            self.planner.instructions(self.states, tick)
            other.instructions(reordered, tick)
            self.assertEqual(self.planner.snapshot(), other.snapshot())


class PlannerPolicyTests(unittest.TestCase):
    def policy(self):
        return ExpertPolicy(load_config(FIXTURES / "expert_policy.json"), planner_config())

    def test_gentle_push_blends_movement_and_uses_resulting_turn_for_reproduction_cost(self):
        policy = self.policy()
        state = agent_state([fruit(100)], energy=400)
        local = policy.action_decision(state)
        pushed = policy.action_decision(state, section_hint=SectionHint((0, 100), 0.15))
        self.assertEqual(local.move_direction, 0)
        self.assertGreater(pushed.move_direction, 0)
        self.assertLess(pushed.move_direction, math.pi / 4)
        self.assertLessEqual(pushed.move_distance, state["speed"])
        self.assertAlmostEqual(pushed.move_distance * math.cos(pushed.move_direction), 8.5)
        self.assertAlmostEqual(pushed.move_distance * math.sin(pushed.move_direction), 1.5)
        self.assertTrue(pushed.spawn_agent)
        self.assertEqual(policy.action_decision(state, section_hint=SectionHint((0, 100), 0)), local)

    def test_visible_and_remembered_escape_override_section_push(self):
        pushed, baseline = self.policy(), self.policy()
        for tick in range(5):
            state = agent_state([predator(20, 0.3)] if tick == 0 else [fruit(10)], age=5 + tick / 10, energy=400)
            expected = baseline.action_decision(state, sim_time=tick / 10)
            actual = pushed.action_decision(state, sim_time=tick / 10, section_hint=SectionHint((100, 0), 1))
            self.assertEqual(actual, expected)
            self.assertFalse(actual.spawn_agent)

    def test_endpoint_runs_planner_from_standard_observations_and_resets_at_game_over(self):
        policy = self.policy()
        states = [agent_state([sighting(8, 100, 0), tree(10, 10)], agent_id=7), agent_state(agent_id=8)]
        request = StepResponse(game_status="ok", score=0, sim_time=0.1,
                               n_agents=2, agent_status=states)
        with patch("agent_server.policy", policy):
            response = predict(request)
            self.assertEqual(predict(request), response)
            self.assertEqual(set(policy.planner.plans[7].assignments), {7, 8})
            predict(request.model_copy(update={"game_status": "game_over"}))
        self.assertEqual(policy.planner.plans, {})


if __name__ == "__main__":
    unittest.main()

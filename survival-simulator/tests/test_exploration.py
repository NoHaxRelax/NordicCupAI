import json
import math
from pathlib import Path
import unittest

import numpy as np

from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.exploration import avoid_edges
from src.utils.controllers.global_planner import GlobalPlanner
from src.utils.controllers.policy_inputs import ExplorationHint
from src.utils.controllers.population import TraitRating
from src.utils.controllers.world_estimator import EdgeLandmark, TreeLandmark, rotate
from test_expert_policy import agent_state, fruit, predator
from test_global_planner import action, planner_config, sighting


FIXTURES = Path(__file__).parent / "fixtures"


def mapping_config(**values):
    defaults = dict(enabled=False, mapping_enabled=True, exploration={"enabled": True})
    defaults.update(values)
    return planner_config(**defaults)


def connected_states(count=5, energy=250):
    return [agent_state([sighting(other, 15 * other, 0) for other in range(2, count + 1)],
                        agent_id=1, energy=energy)] + [
        agent_state(agent_id=index, energy=energy) for index in range(2, count + 1)]


def accelerated_config(**changes):
    exploration = dict(enabled=True, trait_roles_enabled=True, faster_mapping_enabled=True)
    exploration.update(changes)
    return mapping_config(exploration=exploration)


def ratings(count=5):
    return {agent_id: TraitRating(1.3 if agent_id == 1 else 0.7 if agent_id == count else 1.0,
                                 agent_id == 1, agent_id == count,
                                 1.0 if agent_id == 1 else 0.0 if agent_id == count else 0.5, {})
            for agent_id in range(1, count + 1)}


class ExplorationTests(unittest.TestCase):
    def test_mapping_can_run_without_section_planning_or_exploration(self):
        planner = GlobalPlanner(mapping_config(exploration={"enabled": False}))
        self.assertEqual(planner.instructions([agent_state()], 0), {})
        self.assertIn(7, planner.estimator.poses)
        self.assertEqual(planner.exploration_hints, {})
        planner.remember_actions([action()])
        planner.instructions([agent_state()], 1)
        np.testing.assert_allclose(planner.estimator.poses[7].position, [10, 0])
        self.assertEqual(planner.plans, {})

    def test_disconnected_agents_each_scout_in_their_own_frame(self):
        planner = GlobalPlanner(mapping_config())
        planner.instructions([agent_state(agent_id=1), agent_state(agent_id=2)], 0)
        self.assertEqual(set(planner.estimator.groups), {1, 2})
        self.assertEqual({d.role for d in planner.exploration.duties.values()}, {"scout"})
        for hint in planner.exploration_hints.values():
            np.testing.assert_allclose(hint.vector, [8.5, 0], atol=1e-8)

    def test_roles_persist_between_duty_intervals_and_adjust_for_energy(self):
        planner = GlobalPlanner(mapping_config())
        states = connected_states()
        planner.instructions(states, 0)
        roles = {key: duty.role for key, duty in planner.exploration.duties.items()}
        self.assertEqual(sum(role == "scout" for role in roles.values()), 2)
        richer = [{**state, "energy": 450 if state["agent_id"] == 5 else 250} for state in states]
        planner.instructions(richer, 1)
        self.assertEqual({key: duty.role for key, duty in planner.exploration.duties.items()}, roles)
        planner.instructions(richer, 31)
        self.assertEqual(planner.exploration.duties[5].role, "scout")
        tired = [{**state, "energy": 20 if state["agent_id"] == 5 else 250} for state in richer]
        planner.instructions(tired, 32)
        self.assertEqual(planner.exploration.duties[5].role, "resident")

    def test_scout_heading_is_fixed_despite_facing_changes_and_resident_returns_home(self):
        planner = GlobalPlanner(mapping_config())
        states = connected_states(3)
        planner.instructions(states, 0)
        scout = planner.exploration.duties[1]
        desired_heading = scout.heading
        planner.estimator.poses[1].heading = math.pi / 2
        hints = planner.exploration.instructions(states, planner.estimator, 1)
        relative = math.atan2(hints[1].vector[1], hints[1].vector[0])
        self.assertAlmostEqual(math.cos(relative + math.pi / 2), math.cos(desired_heading))
        self.assertAlmostEqual(math.sin(relative + math.pi / 2), math.sin(desired_heading))
        resident = planner.exploration.duties[2]
        pose = planner.estimator.poses[2]
        pose.position = resident.home + np.array([200.0, 0])
        hints = planner.exploration.instructions(states, planner.estimator, 2)
        world_vector = rotate(hints[2].vector, pose.heading)
        self.assertLess(world_vector[0], 0)
        self.assertAlmostEqual(world_vector[1], 0)

    def test_transform_discards_stale_homes_and_frontier_targets(self):
        planner = GlobalPlanner(mapping_config())
        states = connected_states(3)
        planner.instructions(states, 0)
        group = planner.estimator.groups[1]
        old = planner.exploration.duties[2].home.copy()
        group.frame_revision += 1
        for pose in planner.estimator.poses.values():
            pose.position += np.array([400., 200.])
        planner.exploration.instructions(states, planner.estimator, 1)
        np.testing.assert_allclose(planner.exploration.duties[2].home, old + [400, 200])
        self.assertEqual(planner.exploration.duties[2].frame_revision, group.frame_revision)

    def test_anchored_scouts_get_separate_unknown_targets_without_waiting_for_other_groups(self):
        planner = GlobalPlanner(mapping_config())
        states = connected_states() + [agent_state(agent_id=10)]
        planner.instructions(states, 0)
        group = planner.estimator.groups[1]
        group.anchored, group.world_size = True, (600., 600.)
        hints = planner.exploration.instructions(states, planner.estimator, 1)
        targets = [d.target for d in planner.exploration.duties.values() if d.group_id == 1 and d.role == "scout"]
        self.assertEqual(len(targets), 2)
        self.assertFalse(np.array_equal(targets[0], targets[1]))
        self.assertTrue(all(np.all(target >= 0) and np.all(target < 600) for target in targets))
        self.assertEqual(hints[10].objective, "connect clusters and find boundaries")
        self.assertEqual(hints[1].objective, "explore unknown cells")

    def test_uncertain_pose_does_not_use_absolute_target(self):
        planner = GlobalPlanner(mapping_config())
        states = [agent_state()]
        planner.instructions(states, 0)
        group = planner.estimator.groups[7]
        group.anchored, group.world_size = True, (600., 600.)
        planner.estimator.poses[7].uncertainty = 100
        hints = planner.exploration.instructions(states, planner.estimator, 1)
        self.assertIsNone(planner.exploration.duties[7].target)
        self.assertEqual(hints[7].objective, "seek landmarks")

    def test_unreached_frontier_target_is_temporarily_skipped(self):
        planner = GlobalPlanner(mapping_config())
        states = [agent_state()]
        planner.instructions(states, 0)
        group = planner.estimator.groups[7]
        group.anchored, group.world_size = True, (600., 600.)
        planner.exploration.instructions(states, planner.estimator, 1)
        original = planner.exploration.duties[7].target.copy()
        planner.exploration.instructions(states, planner.estimator, 62)
        self.assertFalse(np.array_equal(planner.exploration.duties[7].target, original))
        self.assertIn(tuple(original), planner.exploration.duties[7].failed_targets)

    def test_hungry_scout_heads_towards_known_food(self):
        planner = GlobalPlanner(mapping_config())
        states = [agent_state(energy=40)]
        planner.instructions(states, 0)
        planner.estimator.groups[7].trees.append(TreeLandmark(np.array([0., 100.]), 0))
        hints = planner.exploration.instructions(states, planner.estimator, 1)
        self.assertGreater(hints[7].vector[1], 0)
        self.assertAlmostEqual(hints[7].vector[0], 0)
        self.assertEqual(hints[7].objective, "find food near remembered tree")
        planner.estimator.poses[7].position = np.array([0., 100.])
        hints = planner.exploration.instructions(states, planner.estimator, 2)
        self.assertGreater(math.hypot(*hints[7].vector), 0)

    def test_wall_turn_is_safe_and_persists_after_wall_leaves_vision(self):
        planner = GlobalPlanner(mapping_config())
        states = [agent_state([{"type": "Edge", "coords": [[5, -100], [5, 100]]}])]
        planner.instructions(states, 0)
        hint = planner.exploration_hints[7]
        self.assertLessEqual(hint.vector[0], 1.01)
        heading = planner.exploration.duties[7].heading
        planner.instructions([agent_state()], 1)
        self.assertEqual(planner.exploration.duties[7].heading, heading)
        np.testing.assert_allclose(planner.exploration_hints[7].vector, hint.vector)
        # A starting point close to a wall can move parallel or away from it.
        direction, distance = avoid_edges(math.pi, 10, (((1, -100), (1, 100)),), 4, 7)
        self.assertEqual(distance, 10)
        self.assertAlmostEqual(abs(direction), math.pi)

    def test_roles_are_deterministic_and_retry_reset_and_death_safe(self):
        first, second = GlobalPlanner(mapping_config()), GlobalPlanner(mapping_config())
        states = connected_states()
        for now in (0, 1):
            first.instructions(states, now)
            second.instructions(list(reversed(states)), now)
            self.assertEqual(first.snapshot(), second.snapshot())
        before = json.dumps(first.snapshot(), sort_keys=True, allow_nan=False)
        first.instructions(states, 1)
        self.assertEqual(json.dumps(first.snapshot(), sort_keys=True, allow_nan=False), before)
        first.instructions(states[:1], 2)
        self.assertEqual(set(first.exploration.duties), {1})
        first.instructions([agent_state(agent_id=1, age=0)], 0)
        np.testing.assert_array_equal(first.exploration.duties[1].home, [0, 0])
        first.instructions([], 1)
        self.assertEqual(first.exploration.duties, {})
        self.assertEqual(first.exploration_hints, {})


class ExplorationPolicyTests(unittest.TestCase):
    def policy(self):
        return ExpertPolicy(load_config(FIXTURES / "expert_policy.json"), mapping_config())

    def test_visible_food_and_visible_or_remembered_escape_override_duties(self):
        first, baseline = self.policy(), self.policy()
        hint = ExplorationHint((0, 8), "scout", "connect clusters")
        food = agent_state([fruit(3, -0.4)])
        self.assertEqual(first.action_decision(food, exploration_hint=hint), baseline.action_decision(food))
        for tick in range(5):
            state = agent_state([predator(20, 0.3)] if tick == 0 else [fruit(10)],
                                age=5 + tick / 10, energy=400)
            actual = first.action_decision(state, sim_time=tick / 10, exploration_hint=hint)
            expected = baseline.action_decision(state, sim_time=tick / 10)
            self.assertEqual(actual, expected)
            self.assertFalse(actual.spawn_agent)

    def test_exploration_never_sprints_and_production_step_turns_at_wall(self):
        policy = self.policy()
        oversized = ExplorationHint((100, 0), "scout", "test")
        action = policy.action_decision(agent_state(energy=5), exploration_hint=oversized)
        self.assertEqual(action.move_distance, 10)
        states = [agent_state([{"type": "Edge", "coords": [[5, -100], [5, 100]]}])]
        action = policy.actions_for_step(states, 0)[0]
        self.assertLessEqual(action.move_distance * math.cos(action.move_direction), 1.01)
        self.assertGreater(abs(action.turn_angle), 0)
        self.assertEqual(policy.actions_for_step(states, 0)[0], action)
        policy.actions_for_step(states, 1, "game_over")
        self.assertEqual(policy.planner.exploration.duties, {})


class FasterMappingTests(unittest.TestCase):
    def test_healthy_low_rank_scouts_leave_elite_at_food_and_more_agents_map_early(self):
        planner = GlobalPlanner(accelerated_config())
        states = connected_states()
        planner.instructions(states, 0, trait_ratings=ratings())
        duties = planner.exploration.duties
        self.assertEqual(duties[1].role, "resident")
        self.assertEqual(duties[5].role, "scout")
        self.assertTrue(duties[5].dispersing)
        self.assertEqual(sum(duty.role == "scout" for duty in duties.values()), 3)
        planner.estimator.groups[1].trees.append(TreeLandmark(np.array([20., 30.]), 0))
        planner.exploration.instructions(states, planner.estimator, 1, ratings())
        np.testing.assert_allclose(duties[1].home, [20, 30])
        self.assertEqual(duties[1].objective, "forage near breeding area")

    def test_equal_founder_traits_do_not_trigger_forced_dispersal(self):
        planner = GlobalPlanner(accelerated_config())
        tied = {agent_id: TraitRating(1.0, False, False, 0.5, {}) for agent_id in range(1, 6)}
        planner.instructions(connected_states(), 0, trait_ratings=tied)
        self.assertFalse(any(duty.dispersing for duty in planner.exploration.duties.values()))
        self.assertEqual(sum(duty.role == "scout" for duty in planner.exploration.duties.values()), 3)

    def test_low_energy_low_rank_forages_and_does_not_start_dispersal(self):
        planner = GlobalPlanner(accelerated_config())
        states = connected_states()
        states[-1]["energy"] = 40
        planner.instructions(states, 0, trait_ratings=ratings())
        self.assertEqual(planner.exploration.duties[5].role, "resident")
        self.assertFalse(planner.exploration.duties[5].dispersing)
        planner.estimator.groups[1].trees.append(TreeLandmark(np.array([80., 100.]), 0))
        hints = planner.exploration.instructions(states, planner.estimator, 1, ratings())
        self.assertEqual(hints[5].objective, "find food near remembered tree")
        self.assertLessEqual(math.hypot(*hints[5].vector), states[-1]["speed"])

    def test_dispersal_remembers_birth_origin_across_reassignments_until_distance_met(self):
        planner = GlobalPlanner(accelerated_config())
        states = connected_states()
        planner.instructions(states, 0, trait_ratings=ratings())
        duty = planner.exploration.duties[5]
        birth = planner.estimator.poses[5].origin.copy()
        duty.home += np.array([100., 100.])
        planner.exploration.instructions(states, planner.estimator, 31, ratings())
        np.testing.assert_allclose(duty.origin, birth)
        self.assertTrue(duty.dispersing)
        planner.estimator.poses[5].position = birth + np.array([260., 0.])
        planner.exploration.instructions(states, planner.estimator, 32, ratings())
        self.assertTrue(duty.dispersed)
        self.assertFalse(duty.dispersing)
        planner.estimator.poses[5].position = birth.copy()
        planner.exploration.instructions(states, planner.estimator, 33, ratings())
        self.assertFalse(duty.dispersing)

    def test_newborn_join_preserves_existing_scout_target_and_station(self):
        planner = GlobalPlanner(accelerated_config())
        states = connected_states()
        planner.instructions(states, 0, trait_ratings=ratings())
        duty = planner.exploration.duties[5]
        target = None if duty.target is None else duty.target.copy()
        heading, home, frame = duty.heading, duty.home.copy(), duty.frame_revision
        birth = connected_states(6)
        birth[-1]["energy"] = 75
        updated = ratings()
        updated[6] = TraitRating(1.0, False, False, 0.5, {})
        planner.instructions(birth, 0.1, trait_ratings=updated)
        current = planner.exploration.duties[5]
        self.assertIs(current, duty)
        self.assertEqual(current.frame_revision, frame)
        self.assertEqual(current.heading, heading)
        np.testing.assert_allclose(current.home, home)
        if target is not None:
            np.testing.assert_allclose(current.target, target)

    def test_frontiers_are_adjacent_cached_and_reject_routes_through_known_edges(self):
        planner = GlobalPlanner(accelerated_config())
        states = [agent_state()]
        planner.instructions(states, 0)
        group, pose = planner.estimator.groups[7], planner.estimator.poses[7]
        plan = planner.exploration.frontiers[7]
        self.assertEqual(set(map(tuple, plan.cells)), {(-1, 0), (0, -1), (0, 1), (1, 0)})
        planner.exploration.instructions(states, planner.estimator, 1)
        self.assertIs(planner.exploration.frontiers[7], plan)
        group.anchored, group.world_size = True, (600., 600.)
        pose.position = np.array([60., 60.])
        group.visited = {(0, 0): (pose.position.copy(), 0.05, 1)}
        group.edges = [EdgeLandmark(np.array([120., 0.]), np.array([120., 500.]), 1)]
        duty = planner.exploration.duties[7]
        duty.heading, duty.target, duty.next_target_at = 0.0, None, 0.0
        planner.exploration.instructions(states, planner.estimator, 2)
        self.assertIsNot(planner.exploration.frontiers[7], plan)
        np.testing.assert_allclose(duty.target, [60., 180.])
        self.assertTrue(planner.exploration._clear_route(planner.exploration.frontiers[7], pose.position, duty.target))

    def test_distinct_frontier_targets_walking_cap_and_trait_snapshot(self):
        planner = GlobalPlanner(accelerated_config())
        planner.population_snapshot = {"elite_ids": [1]}
        states = connected_states()
        planner.instructions(states, 0, trait_ratings=ratings())
        snapshot = json.loads(json.dumps(planner.snapshot(), allow_nan=False))
        self.assertEqual(snapshot["population"]["elite_ids"], [1])
        elite = next(agent for agent in snapshot["agents"] if agent["agent_id"] == 1)
        self.assertEqual(elite["trait_score"], 1.3)
        self.assertTrue(elite["elite"])
        targets = [tuple(duty.target) for duty in planner.exploration.duties.values() if duty.target is not None]
        self.assertEqual(len(targets), len(set(targets)))
        for state in states:
            hint = planner.exploration_hints[state["agent_id"]]
            self.assertLessEqual(math.hypot(*hint.vector), min(state["speed"], state["sprint_speed"]))
        planner.reset()
        self.assertEqual(planner.trait_ratings, {})
        self.assertEqual(planner.population_snapshot, {})
        self.assertEqual(planner.exploration.frontiers, {})


if __name__ == "__main__":
    unittest.main()

"""Exercise sprint decisions at the final action/energy boundary."""

import math
from pathlib import Path
import unittest

from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from src.utils.controllers.policy_inputs import ExplorationHint, ReproductionHint, SectionHint
from test_expert_policy import agent_state, fruit, predator


class MappingSprintTests(unittest.TestCase):
    def policy(self, **sections):
        data = load_config(Path(__file__).parent / "fixtures" / "expert_policy.json").model_dump()
        data["crowd"]["enabled"] = False
        for name, values in sections.items():
            data[name].update(values)
        return ExpertPolicy(ExpertConfig.model_validate(data))

    def hint(self, **values):
        defaults = dict(vector=(20.0, 0.0), role="scout", objective="survey",
                        max_speed=20.0, minimum_energy_reserve=100.0)
        defaults.update(values)
        return ExplorationHint(**defaults)

    def test_legacy_hint_walks_but_explicit_sprint_uses_mutated_cap(self):
        policy = self.policy()
        walking = policy.action_decision(agent_state(), exploration_hint=self.hint(max_speed=None))
        sprinting = policy.action_decision(agent_state(sprint_speed=16), exploration_hint=self.hint())
        self.assertEqual(walking.move_distance, 10)
        self.assertEqual(sprinting.move_distance, 16)

    def test_sprint_preserves_reserve_after_movement_and_turn(self):
        policy = self.policy()
        action = policy.action_decision(agent_state(energy=102),
                                       exploration_hint=self.hint(look_direction=math.pi / 4))
        self.assertAlmostEqual(action.move_distance, 12.75)
        cost = 10 * .05 + (action.move_distance - 10) * .5 + abs(action.turn_angle) / math.tau
        self.assertAlmostEqual(102 - cost, 100)
        low = policy.action_decision(agent_state(energy=99), exploration_hint=self.hint())
        self.assertEqual(low.move_distance, 10)

    def test_zero_sprint_cost_is_supported(self):
        policy = self.policy(mechanics={"sprinting_energy_per_unit": 0})
        action = policy.action_decision(agent_state(), exploration_hint=self.hint())
        self.assertEqual(action.move_distance, 20)

    def test_stale_observations_cannot_authorize_sprints_and_retries_are_stable(self):
        policy = self.policy(movement={"food_sprint_enabled": True})
        state = agent_state([fruit(30)], energy=180)
        first = policy.action_decision(state, sim_time=1)
        self.assertEqual(first.move_distance, 20)
        self.assertEqual(policy.action_decision(state, sim_time=1), first)
        stale = policy.action_decision(state, sim_time=1.1)
        self.assertEqual(stale.move_distance, 10)
        self.assertEqual(policy.action_decision(state, sim_time=1.1), stale)
        fresh = policy.action_decision({**state, "age": 5.1}, sim_time=1.2)
        self.assertEqual(fresh.move_distance, 20)
        policy.reset()
        self.assertEqual(policy.action_decision(state, sim_time=1).move_distance, 20)

    def test_final_heading_and_visibility_gate_sprinting(self):
        policy = self.policy()
        turned = policy.action_decision(agent_state(), exploration_hint=self.hint(),
                                       section_hint=SectionHint(vector=(0, 100), strength=1.0))
        self.assertLessEqual(turned.move_distance, 10)
        blind = policy.action_decision(agent_state(vision_range=22), exploration_hint=self.hint())
        self.assertEqual(blind.move_distance, 10)
        sideways = policy.action_decision(agent_state(),
                                         exploration_hint=self.hint(vector=(0, 20)))
        self.assertEqual(sideways.move_distance, 10)

    def test_visible_stone_prevents_fast_collision(self):
        policy = self.policy()
        action = policy.action_decision(agent_state([
            {"type": "Edge", "coords": [[18, -100], [18, 100]]},
        ]), exploration_hint=self.hint())
        self.assertLessEqual(action.move_distance, 10)

    def test_repeated_rays_do_not_change_obstacle_avoidance(self):
        policy = self.policy()
        wall = {"type": "Edge", "coords": [[18, -100], [18, 100]]}
        single = policy.action_decision(agent_state([wall]), exploration_hint=self.hint())
        repeated = policy.action_decision(agent_state([wall] * 80), exploration_hint=self.hint())
        self.assertEqual(single, repeated)

    def test_survey_turn_is_independent_of_motion_and_food_keeps_priority(self):
        policy = self.policy()
        hint = self.hint(max_speed=None, look_direction=.7)
        survey = policy.action_decision(agent_state(), exploration_hint=hint)
        self.assertEqual(survey.move_direction, 0)
        self.assertAlmostEqual(survey.turn_angle, .7)
        food = policy.action_decision(agent_state([fruit(5, -.3)]), exploration_hint=hint)
        self.assertAlmostEqual(food.move_direction, -.3)
        self.assertAlmostEqual(food.turn_angle, -.3)
        self.assertEqual(food.move_distance, 5)

    def test_directed_scout_passes_distant_food_but_collects_nearby_food(self):
        policy = self.policy()
        hint = self.hint(max_speed=None, food_distance_limit=25)
        far = policy.action_decision(agent_state([fruit(80, -.3)]), exploration_hint=hint)
        self.assertEqual(far.move_direction, 0)
        self.assertEqual(far.move_distance, 10)
        near = policy.action_decision(agent_state([fruit(5, -.3)]), exploration_hint=hint)
        self.assertAlmostEqual(near.move_direction, -.3)
        self.assertEqual(near.move_distance, 5)
        ordinary = policy.action_decision(agent_state([fruit(80, -.3)]),
                                         exploration_hint=self.hint(max_speed=None))
        self.assertAlmostEqual(ordinary.move_direction, -.3)

    def test_food_dash_requires_nearby_visible_clear_route_and_energy(self):
        policy = self.policy(movement={"food_sprint_enabled": True})
        allowed = policy.action_decision(agent_state([fruit(30)], energy=180))
        self.assertEqual(allowed.move_distance, 20)
        for state in (agent_state([fruit(80)], energy=180),
                      agent_state([fruit(30)], energy=169),
                      agent_state([fruit(30, math.pi / 2)], energy=180),
                      agent_state([fruit(30), {"type": "Edge", "coords": [[15, -20], [15, 20]]}], energy=180)):
            with self.subTest(state=state):
                self.assertLessEqual(policy.action_decision(state).move_distance, 10)

    def test_heard_food_can_be_collected_while_scanning_and_growth_stops_food_dashes(self):
        policy = self.policy(movement={"scan_nearby_food": True, "food_sprint_enabled": True})
        hint = self.hint(look_direction=.7)
        action = policy.action_decision(agent_state([fruit(30, -.2)], energy=180), exploration_hint=hint)
        self.assertAlmostEqual(action.move_direction, -.2)
        self.assertAlmostEqual(action.turn_angle, .7)
        self.assertEqual(action.move_distance, 20)
        far = policy.action_decision(agent_state([fruit(80, -.2)], energy=180), exploration_hint=hint)
        self.assertAlmostEqual(far.turn_angle, -.2)
        policy.planner.population_phase = True
        settled = policy.action_decision(agent_state([fruit(30)], energy=180), exploration_hint=hint)
        self.assertEqual(settled.move_distance, 10)

    def test_reproduction_accounts_for_sprint_and_escape_keeps_priority(self):
        policy = self.policy(reproduction={"minimum_energy_reserve": 75})
        hint = self.hint(look_direction=.7)
        breeding = ReproductionHint(energy_threshold=170, allowed=True)
        action = policy.action_decision(agent_state(energy=180), exploration_hint=hint,
                                       reproduction_hint=breeding)
        self.assertGreater(action.move_distance, 10)
        self.assertFalse(action.spawn_agent)
        escape = policy.action_decision(agent_state([predator(20)], energy=120),
                                       exploration_hint=self.hint(minimum_energy_reserve=200))
        self.assertEqual(escape.move_distance, 20)
        self.assertAlmostEqual(abs(escape.move_direction), math.pi)
        self.assertLess(escape.turn_angle, 0)


if __name__ == "__main__":
    unittest.main()

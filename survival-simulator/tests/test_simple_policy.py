import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from src.utils.DTOs import ActionRequest
from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.global_planner import GlobalPlanner, load_planner_config
from src.utils.controllers.population import PopulationTracker, TRAITS
from src.utils.controllers.navigation import NavigationResult
from src.utils.controllers.policy_inputs import HarvestHint, ReproductionHint
from src.utils.controllers.simple_policy import SimpleConfig, SimpleCoordinator
from src.utils.controllers.world_estimator import BiomeSample, EstimatedPose, MapGroup
from test_expert_policy import agent_state, fruit


class SimplePolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.policy = SimpleCoordinator(SimpleConfig())
        self.population = PopulationTracker(self.config.reproduction.selection)
        self.planner = GlobalPlanner()
        self.planner.population_phase = True
        self.group = MapGroup(1, anchored=True, world_size=(1000., 1000.))
        self.planner.estimator.groups = {1: self.group}

    def step(self, states, now):
        for state in states:
            self.planner.estimator.poses.setdefault(state["agent_id"],
                EstimatedPose(state["agent_id"], 1, np.array([100., 100.])))
        living = {s["agent_id"] for s in states}
        self.planner.estimator.poses = {i: p for i, p in self.planner.estimator.poses.items() if i in living}
        self.population.update(states, now)
        return self.policy.update(states, now, self.planner, self.population, self.config.mechanics, 350.)

    def test_population_schedule_best_eligible_genes_and_global_cooldown(self):
        states = [agent_state(agent_id=i, energy=250.) for i in range(3)]
        self.step(states, 0.)
        for trait in TRAITS:
            states[2][trait] *= 1.2
        _, births = self.step(states, 1.)
        self.assertEqual([i for i, h in births.items() if h.allowed], [2])
        self.policy.remember_actions([ActionRequest(agent_id=2, move_distance=0., move_direction=0., turn_angle=0., spawn_agent=True)], 1.)
        self.assertFalse(any(h.allowed for h in self.step(states, 2.)[1].values()))
        self.step(states, 900.)
        self.assertEqual(self.policy.population_target, 6)
        self.step(states, 1800.)
        self.assertEqual(self.policy.population_target, 3)
        self.assertFalse(any(h.allowed for h in self.policy.cached[1].values()))

    def test_birth_target_counts_only_under_40_and_keeps_parent_reserve(self):
        states = [agent_state(agent_id=i, energy=300., age=70.) for i in range(12)]
        self.assertEqual(sum(h.allowed for h in self.step(states, 1800.)[1].values()), 1)
        states.extend(agent_state(agent_id=i, age=0., energy=75.) for i in range(20, 23))
        self.assertFalse(any(h.allowed for h in self.step(states, 1801.)[1].values()))
        self.assertEqual(self.policy.population_plan["young"], 3)
        states[-1]["age"] = 40.
        self.assertTrue(any(h.allowed for h in self.step(states, 1802.)[1].values()))
        self.assertEqual(self.policy.population_plan["young"], 2)
        states = [agent_state(agent_id=5, energy=150., age=70.)]
        self.assertFalse(any(h.allowed for h in self.step(states, 1803.)[1].values()))

    def test_birth_spacing_can_replenish_each_counted_age_window(self):
        states = [agent_state(agent_id=1)]
        for now, target in ((0., 12), (900., 6), (1800., 3)):
            self.step(states, now)
            self.assertEqual(self.policy.population_target, target)
            self.assertLess(self.policy.population_plan["birth_interval_seconds"], 40. / target)

    def age_scouts(self, states):
        for state in states:
            state.update(age=80., energy=200.)
        self.step(states, 0.)
        for tick in (1, 2):
            self.policy.remember_actions([ActionRequest(agent_id=s["agent_id"], move_distance=0., move_direction=0.,
                turn_angle=0., spawn_agent=False) for s in states], (tick - 1) / 10)
            for state in states:
                state["age"] = 80. + tick / 10
                state["energy"] -= .1 + .01 * state["age"]
            result = self.step(states, tick / 10)
        return result

    def test_confirmed_aging_scouts_release_food_and_homes_and_choose_distinct_frontiers(self):
        states = [agent_state([fruit(30)], agent_id=i) for i in (1, 2)]
        hints, births = self.age_scouts(states)
        self.assertEqual(self.policy.retired, {1, 2})
        self.assertFalse(self.policy.assignments)
        self.assertFalse(self.policy.homes)
        self.assertFalse(any(h.allowed for h in births.values()))
        self.assertTrue(all(h.retired and not h.allow_local_food for h in hints.values()))
        self.assertTrue(all(math.hypot(*h.vector) > 0 for h in hints.values()))
        goals = list(self.policy.scout_goals.values())
        self.assertEqual(len(goals), 2)
        self.assertGreater(np.linalg.norm(goals[0] - goals[1]), 50.)
        self.assertTrue(all("explore unseen" in task["kind"] for task in self.policy.tasks.values()))
        json.dumps(self.policy.snapshot(), allow_nan=False)

    def test_normal_old_agents_are_not_retired_from_age_or_action_costs_alone(self):
        state = agent_state(agent_id=1, age=80., energy=350.)
        self.step([state], 0.)
        for tick in (1, 2):
            self.policy.remember_actions([ActionRequest(agent_id=1, move_distance=10., move_direction=0.,
                turn_angle=math.pi, spawn_agent=tick == 1)], (tick - 1) / 10)
            state["age"] += .1
            state["energy"] -= 1.1 + (100. if tick == 1 else 0.)
            self.step([state], tick / 10)
        self.assertFalse(self.policy.retired)
        self.assertIn(1, self.policy.homes)

    def test_aging_survives_meals_but_not_episode_reset(self):
        state = agent_state(agent_id=1)
        self.age_scouts([state])
        self.policy.remember_actions([ActionRequest(agent_id=1, move_distance=0., move_direction=0.,
            turn_angle=0., spawn_agent=False)], .2)
        state.update(age=80.3, energy=250.)
        self.step([state], .3)
        self.assertIn(1, self.policy.retired)
        state.update(age=0., energy=75.)
        self.step([state], .4)
        self.assertFalse(self.policy.retired)
        self.assertFalse(self.policy.scout_goals)

    def test_scout_avoids_food_even_when_hungry_without_sprinting(self):
        expert = ExpertPolicy(self.config.model_copy(update={"policy_mode": "simple"}))
        state = agent_state([fruit(15)], agent_id=1, energy=25., age=80.)
        hint = HarvestHint((100., 0.), None, 0., False, retired=True, allow_local_food=False)
        action = expert.action_decision(state, sim_time=0., harvest_hint=hint,
                                       reproduction_hint=ReproductionHint(177., False))
        self.assertGreater(action.move_distance, 0.)
        self.assertLessEqual(action.move_distance, state["speed"])
        x, y = action.move_distance * math.cos(action.move_direction), action.move_distance * math.sin(action.move_direction)
        self.assertGreaterEqual(math.hypot(15. - x, y), 12.)
        self.assertFalse(action.spawn_agent)

    def test_scout_avoids_all_visible_fruits_not_just_five_nearest(self):
        expert = ExpertPolicy(self.config.model_copy(update={"policy_mode": "simple"}))
        observations = [fruit(14., math.pi)] * 5 + [fruit(15.)]
        state = agent_state(observations, agent_id=1, energy=25., age=80.)
        hint = HarvestHint((100., 0.), None, 0., False, retired=True, allow_local_food=False)
        action = expert.action_decision(state, sim_time=0., harvest_hint=hint,
                                       reproduction_hint=ReproductionHint(177., False))
        x, y = action.move_distance * math.cos(action.move_direction), action.move_distance * math.sin(action.move_direction)
        self.assertGreaterEqual(math.hypot(15. - x, y), 12.)
        self.assertGreaterEqual(math.hypot(-14. - x, y), 12.)

    def test_food_is_shared_reserved_once_and_collected_without_waiting(self):
        states = [agent_state([fruit(30)], agent_id=i) for i in (1, 2)]
        hints, _ = self.step(states, 0.)
        self.assertEqual(len(self.policy.tracks), 1)
        self.assertEqual(len(self.policy.assignments), 1)
        i = next(iter(self.policy.assignments))
        self.assertGreater(hints[i].vector[0], 0.)
        self.assertFalse(hints[i].waiting)
        # A fresh empty hearing circle clears eaten food immediately.
        for state in states:
            state.update(observations=[], age=6.)
        self.step(states, .1)
        self.assertFalse(self.policy.tracks)
        self.assertFalse(self.policy.assignments)

    def test_stale_and_duplicate_frames_do_not_invent_fruit(self):
        state = agent_state([fruit(150)], agent_id=1)
        first = self.step([state], 0.)
        state["observations"] = [fruit(500)]
        self.assertEqual(self.step([state], 0.), first)
        self.step([state], .1)
        self.assertEqual(len(self.policy.tracks), 1)
        self.assertEqual(next(iter(self.policy.tracks.values())).last_seen, 0.)
        self.step([state], 9.)
        self.assertFalse(self.policy.tracks)

    def test_homes_spread_over_productive_land_and_remain_stable(self):
        points = [((100., 100.), "forest"), ((800., 800.), "forest"),
                  ((900., 100.), "river")]
        for i, (point, biome) in enumerate(points):
            self.group.biomes[i] = BiomeSample(np.array(point), biome, 0., 0.)
        states = [agent_state(agent_id=i) for i in (1, 2)]
        self.step(states, 0.)
        homes = {i: p.tolist() for i, p in self.policy.homes.items()}
        self.assertEqual(sorted(homes.values()), [[100., 100.], [800., 800.]])
        self.planner.estimator.poses[1].position[:] = 700.
        self.step(states, 1.)
        self.assertEqual(homes, {i: p.tolist() for i, p in self.policy.homes.items()})
        self.step(states, 31.)
        self.assertEqual(homes, {i: p.tolist() for i, p in self.policy.homes.items()})
        json.dumps(self.policy.snapshot(), allow_nan=False)

    def test_frame_change_drops_old_food_coordinates(self):
        state = agent_state([fruit(150)], agent_id=1)
        self.step([state], 0.)
        self.group.frame_revision += 1
        state.update(age=6., observations=[])
        self.step([state], 1.)
        self.assertFalse(self.policy.tracks)
        self.assertFalse(self.policy.assignments)

    def test_unreachable_food_does_not_restart_pathfinding_each_tick(self):
        state = agent_state([fruit(30)], agent_id=1)
        with patch.object(self.policy.navigator, "steer", return_value=NavigationResult(None, True, math.inf, "blocked")) as steer:
            for tick in range(10):
                state["age"] = 5. + tick * .5
                self.step([state], tick * .5)
            fruit_routes = [call for call in steer.call_args_list if np.allclose(call.args[2], [130., 100.])]
            self.assertEqual(len(fruit_routes), 1)

    def test_idle_scans_complete_a_sweep_then_rest(self):
        state = agent_state(agent_id=1, energy=50.)
        for tick in range(8):
            hint = self.step([state], tick / 10)[0][1]
            self.assertTrue(hint.scan_while_stationary)
            self.assertEqual(hint.vector, (0., 0.))
        self.assertFalse(self.step([state], .8)[0][1].scan_while_stationary)
        self.assertTrue(self.step([state], 4.1)[0][1].scan_while_stationary)

    def test_declining_activity_and_invalid_schedule(self):
        cfg = self.policy.config
        self.assertEqual(cfg.interpolate(0., cfg.map_interval_seconds, cfg.late_map_interval_seconds), .5)
        self.assertEqual(cfg.interpolate(3000., cfg.map_interval_seconds, cfg.late_map_interval_seconds), 2.)
        with self.assertRaises(ValidationError):
            SimpleConfig(population_late=20)
        with self.assertRaises(ValidationError):
            SimpleConfig(late_seconds=500.)

    def test_selecting_simple_bypasses_old_harvest_coordinator(self):
        expert = ExpertPolicy(self.config.model_copy(update={"policy_mode": "simple"}))
        self.assertIsInstance(expert.harvest, SimpleCoordinator)
        self.assertEqual(expert.planner.biome_estimator.config.refit_interval_seconds, 10.)
        actions = expert.actions_for_step([agent_state([fruit(20)], agent_id=1)], 0.)
        self.assertEqual(len(actions), 1)
        expert.reset()
        self.assertIsNone(expert.harvest.last_time)


class ThrottledMapTests(unittest.TestCase):
    def setUp(self):
        config = load_planner_config()
        config = config.model_copy(update={"exploration": config.exploration.model_copy(update={"enabled": False})})
        self.planner = GlobalPlanner(config)
        self.estimator = self.planner.estimator
        self.estimator.groups = {1: MapGroup(1, anchored=True, world_size=(1000., 1000.))}
        self.estimator.poses = {1: EstimatedPose(1, 1, np.array([100., 100.]), biome="grassland")}
        self.planner.population_phase = True

    def test_every_motion_and_turn_is_integrated_between_sensing_updates(self):
        with patch.object(self.estimator, "_correct_from_edges", wraps=self.estimator._correct_from_edges) as correct:
            for tick in range(6):
                self.planner.instructions([agent_state(agent_id=1, age=5. + tick / 10)], tick / 10,
                                          territory_policy=True, map_interval_seconds=.5)
                self.estimator.remember_actions([ActionRequest(agent_id=1, move_distance=10., move_direction=0.,
                    turn_angle=math.pi / 2 if tick == 0 else 0., spawn_agent=False)])
            self.assertEqual(correct.call_count, 2)
            np.testing.assert_allclose(self.estimator.poses[1].position, [110., 140.])
            self.assertAlmostEqual(self.estimator.poses[1].heading, math.pi / 2)
            # A same-time retry must not predict a second movement.
            self.planner.instructions([agent_state(agent_id=1, age=5.5)], .5, map_interval_seconds=.5)
            np.testing.assert_allclose(self.estimator.poses[1].position, [110., 140.])

    def test_newborn_forces_sensing_and_rewind_restores_initial_mapping(self):
        self.planner.instructions([agent_state(agent_id=1)], 10., map_interval_seconds=2.)
        self.planner.instructions([agent_state(agent_id=1, age=5.1), agent_state(agent_id=2)],
                                  10.1, map_interval_seconds=2.)
        self.assertEqual(self.planner.last_map_observation, 10.1)
        self.assertIn(2, self.estimator.poses)
        self.planner.instructions([agent_state(agent_id=1, age=0.)], 0., map_interval_seconds=2.)
        self.assertFalse(self.planner.population_phase)
        self.assertEqual(self.planner.last_map_observation, 0.)
        self.assertNotIn(2, self.estimator.poses)


if __name__ == "__main__":
    unittest.main()

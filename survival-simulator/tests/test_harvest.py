import math
from types import SimpleNamespace
import unittest

import numpy as np
from pydantic import ValidationError

from src.core import SimulationCore
from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.harvest import FruitTrack, HarvestConfig, HarvestCoordinator
from src.utils.controllers.policy_inputs import HarvestHint
from src.utils.controllers.population import PopulationTracker, TRAITS
from src.utils.controllers.world_estimator import EstimatedPose, MapGroup, TreeLandmark
from test_expert_policy import agent_state, fruit, predator


class HarvestTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True, coverage={"enabled": False}))
        self.population = PopulationTracker(self.cfg.reproduction.selection)
        self.group = MapGroup(1, anchored=True, world_size=(1000, 1000))
        self.poses = {}
        self.planner = SimpleNamespace(population_phase=True,
            estimator=SimpleNamespace(poses=self.poses, groups={1: self.group}))

    def step(self, states, now):
        for state in states:
            state["age"] = now + 5
            self.poses.setdefault(state["agent_id"], EstimatedPose(state["agent_id"], 1, np.array([100., 100.])))
        self.population.update(states, now)
        self.population.population_phase = True
        return self.harvest.update(states, now, self.planner, self.population, self.cfg.mechanics, 350)

    def test_shared_sightings_deduplicate_and_do_not_reset_age(self):
        states = [agent_state([fruit(35)], agent_id=i) for i in (1, 2)]
        hints, _ = self.step(states, 10)
        self.assertEqual(len(self.harvest.tracks), 1)
        self.assertEqual(sum(h.track_id is not None for h in hints.values()), 1)
        track = next(iter(self.harvest.tracks.values()))
        self.assertEqual(self.harvest.energy(track, 10), 30)
        self.step(states, 20)
        self.assertEqual(len(self.harvest.tracks), 1)
        self.assertEqual(track.first_seen, 10)
        self.assertEqual(self.harvest.energy(track, 30), 60)

    def test_two_close_fruits_in_one_observation_stay_distinct(self):
        states = [agent_state([fruit(30), fruit(32)], agent_id=i) for i in (1, 2)]
        self.step(states, 0)
        self.step(states, .1)
        self.assertEqual(len(self.harvest.tracks), 2)
        self.assertEqual(len(set(self.harvest.assignments.values())), 2)

    def test_previously_empty_view_brackets_birth(self):
        self.step([agent_state([], agent_id=1)], 10)
        self.step([agent_state([fruit(30)], agent_id=1)], 10.1)
        track = next(iter(self.harvest.tracks.values()))
        self.assertTrue(track.birth_bracketed)
        self.assertAlmostEqual(track.estimated_birth, 10.05)
        self.assertAlmostEqual(self.harvest.energy(track, 10.1), 20.1)

    def test_missing_fruit_is_removed_only_where_observation_was_reliable(self):
        self.step([agent_state([fruit(30), fruit(100)], agent_id=1)], 0)
        self.poses[1].heading = math.pi
        self.step([agent_state([], agent_id=1)], .1)
        self.assertEqual(len(self.harvest.tracks), 1)  # Far fruit is behind the observer.
        track = next(iter(self.harvest.tracks.values()))
        self.assertEqual(track.position[0], 200)
        self.step([agent_state([], agent_id=1)], 21)
        self.assertFalse(self.harvest.tracks)

    def test_waits_outside_pickup_radius_and_hungry_agents_eat_early(self):
        state = agent_state([], energy=300, agent_id=1)
        self.step([state], 0)
        state["observations"] = [fruit(20)]
        hints, _ = self.step([state], 1.)
        self.assertTrue(hints[1].waiting)
        self.assertAlmostEqual(hints[1].vector[0], 2)
        state["energy"] = 40
        hints, _ = self.step([state], 1.1)
        self.assertFalse(hints[1].waiting)
        self.assertGreater(hints[1].vector[0], 0.)
        self.assertLess(20. - hints[1].vector[0], 10.)  # Inside fresh-fruit pickup reach.

    def test_only_known_fresh_fruit_allows_a_safe_ripening_wait(self):
        state = agent_state(agent_id=1, age=10., energy=75.)
        pose = EstimatedPose(1, 1, np.array([100., 100.]))
        # Both tracks are first seen at100s. The unknown fruit could already
        # be47s old and rot in3s, before the previous13s wait could finish.
        unknown = FruitTrack(1, np.array([118., 100.]), 100., 100., 95., False, 0.)
        fresh = FruitTrack(2, unknown.position.copy(), 100., 100., 99.95, True, 0.)
        utility, _, wait = self.harvest._candidate(state, pose, unknown, 100., .05)
        self.assertGreater(utility, 0.)
        self.assertEqual(wait, 0.)
        utility, _, wait = self.harvest._candidate(state, pose, fresh, 100., .05)
        self.assertGreater(utility, 0.)
        self.assertGreater(wait, 10.)
        self.assertGreater(state["energy"] - .6 - (wait + .1) * self.harvest._metabolism(state),
                           self.harvest.config.survival_reserve)
        state["energy"] = 40.
        self.assertEqual(self.harvest._candidate(state, pose, fresh, 100., .05)[2], 0.)

    def test_absolute_map_required_frame_changes_and_rewind_forget_tracks(self):
        state = agent_state([fruit(30)], agent_id=1)
        self.planner.population_phase = False
        self.assertEqual(self.step([state], 0), ({}, {}))
        self.assertFalse(self.harvest.tracks)
        self.planner.population_phase = True
        self.step([state], 10)
        self.group.frame_revision += 1
        self.step([state], 20)
        self.assertEqual(next(iter(self.harvest.tracks.values())).first_seen, 20)
        self.step([state], 0)
        self.assertEqual(next(iter(self.harvest.tracks.values())).first_seen, 0)

    def test_stale_observations_and_same_time_retries_do_not_create_evidence(self):
        state = agent_state([fruit(30)], agent_id=1)
        hints, breeding = self.step([state], 1)
        result = self.harvest.update([state], 1, self.planner, self.population, self.cfg.mechanics, 350)
        self.assertEqual(result, (hints, breeding))
        state["observations"] = [fruit(100)]
        self.harvest.update([state], 2, self.planner, self.population, self.cfg.mechanics, 350)
        self.assertEqual(len(self.harvest.tracks), 1)
        self.assertEqual(next(iter(self.harvest.tracks.values())).last_seen, 1)

    def test_global_assignment_avoids_duplicate_targets_and_prefers_near_agents(self):
        states = [agent_state([fruit(30)], agent_id=1), agent_state([fruit(30, math.pi)], agent_id=2)]
        self.poses[2] = EstimatedPose(2, 1, np.array([300., 100.]))
        hints, _ = self.step(states, 0)
        self.assertEqual(len(set(self.harvest.assignments.values())), 2)
        self.assertGreater(hints[1].vector[0], 0)
        self.assertLess(hints[2].vector[0], 0)

    def test_best_genes_get_breeding_slot_without_exhausting_parent(self):
        self.harvest.config = self.harvest.config.model_copy(update={"survival_population": 12})
        states = [agent_state(agent_id=i, energy=400) for i in range(3)]
        self.step(states, 0)
        for trait in TRAITS:
            states[2][trait] *= 1.2
        _, hints = self.step(states, 1)
        self.assertEqual([i for i, hint in hints.items() if hint.allowed], [2])
        states[2]["energy"] = 125
        _, hints = self.step(states, 2)
        self.assertFalse(hints[2].allowed)
        self.assertEqual(sum(h.allowed for h in hints.values()), 1)

    def test_population_cap_and_birth_rate_limit(self):
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True, minimum_population=2, maximum_population=2))
        states = [agent_state(agent_id=i, energy=400) for i in range(2)]
        _, hints = self.step(states, 0)
        self.assertFalse(any(h.allowed for h in hints.values()))
        states = states[:1]
        _, hints = self.step(states, 1)
        self.assertTrue(hints[0].allowed)
        self.harvest.remember_actions([SimpleNamespace(agent_id=0, spawn_agent=True,
                                                       move_distance=0., turn_angle=0.)], 1)
        _, hints = self.step(states, 1.1)
        self.assertFalse(hints[0].allowed)

    def test_waiting_overrides_greedy_food_but_predator_escape_still_wins(self):
        policy = ExpertPolicy(self.cfg)
        hint = HarvestHint((0., 0.), 1, 30, True)
        action = policy.action_decision(agent_state([fruit(10)], energy=200), harvest_hint=hint)
        self.assertEqual(action.move_distance, 0)
        action = policy.action_decision(agent_state([fruit(10), predator(20)], energy=200), harvest_hint=hint)
        self.assertEqual(action.move_distance, 20)
        self.assertFalse(action.spawn_agent)

    def test_gene_backup_requires_spare_food_instead_of_expanding_a_starving_colony(self):
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True, minimum_population=2))
        states = [agent_state(agent_id=i, energy=400) for i in range(2)]
        self.step(states, 0)
        states[1]["speed"] *= 1.5
        _, hints = self.step(states, 40)
        self.assertFalse(any(hint.allowed for hint in hints.values()))
        self.assertEqual(self.harvest.gene_backup_ids, set())
        for i in (2, 3):
            child = dict(states[1], agent_id=i, age=5)
            states.append(child)
            self.poses[i] = EstimatedPose(i, 1, np.array([100., 100.]))
        self.population.update(states, 41)
        hints = self.harvest._breeders(states, self.poses, 41, self.population, self.cfg.mechanics, 350)
        self.assertFalse(any(h.allowed for h in hints.values()))

    def test_population_budget_shrinks_when_food_disappears(self):
        states = [agent_state(agent_id=i, energy=400) for i in range(12)]
        _, hints = self.step(states, 0)
        self.assertEqual(self.harvest.population_target, 6)
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_old_parent_renews_lineage_with_a_lower_reserve_and_weak_genes(self):
        state = agent_state(agent_id=1, energy=160)
        self.step([state], 0)
        for trait in TRAITS:
            state[trait] *= .8
        _, hints = self.step([state], 65)
        self.assertTrue(hints[1].allowed)
        self.assertTrue(hints[1].preserve_lineage)
        self.assertEqual(hints[1].minimum_energy_reserve, 20.)
        self.assertLess(hints[1].energy_threshold, state["energy"])

    def test_two_young_carriers_prevent_unnecessary_age_replacement(self):
        self.harvest.config = self.harvest.config.model_copy(update={"minimum_population": 2})
        states = [agent_state(agent_id=i, energy=400) for i in range(3)]
        self.step(states, 0)
        states[0]["age"] = 70.
        states[1]["age"] = states[2]["age"] = 10.
        self.population.update(states, 1)
        hints = self.harvest._breeders(states, self.poses, 1, self.population, self.cfg.mechanics, 350)
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_aging_agent_does_not_wait_to_ripen_a_meal(self):
        self.step([agent_state([], agent_id=1, energy=300)], 59.)
        hints, _ = self.step([agent_state([fruit(20)], agent_id=1, energy=300)], 60)
        self.assertFalse(hints[1].waiting)
        self.assertGreater(hints[1].vector[0], 0.)
        self.assertLess(20. - hints[1].vector[0], 10.)

    def test_senescent_parent_cannot_take_the_only_meal_from_a_young_carrier(self):
        states = [agent_state([fruit(30)], agent_id=1, energy=20),
                  agent_state([fruit(30)], agent_id=2, energy=80)]
        self.step(states, 0)
        states[0]["age"], states[1]["age"] = 100., 10.
        self.harvest._assign(states, self.poses, .1, self.population, self.cfg.mechanics)
        self.assertEqual(set(self.harvest.assignments), {2})

    def test_healthy_agent_rests_when_there_is_no_food_to_collect(self):
        hints, _ = self.step([agent_state(agent_id=1, energy=300)], 0)
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertEqual(self.harvest.tasks[1]["kind"], "rest with food reserve")

    def test_agent_watches_recent_tree_in_hearing_range_without_pacing(self):
        self.group.trees = [TreeLandmark(np.array([115., 100.]), 0)]
        hints, _ = self.step([agent_state(agent_id=1, energy=150)], 0)
        self.assertEqual(hints[1].vector, (0., 0.))
        self.assertEqual(self.harvest.tasks[1]["kind"], "watch orchard")
        hints, _ = self.step([agent_state(agent_id=1, energy=140)], 7)
        self.assertEqual(self.harvest.tasks[1]["kind"], "local exploration")

    def test_recent_tree_does_not_trigger_an_unaffordable_trip(self):
        self.group.trees = [TreeLandmark(np.array([900., 100.]), 0)]
        self.step([agent_state(agent_id=1, energy=30)], 0)
        self.assertEqual(self.harvest.tasks[1]["kind"], "local exploration")

    def test_three_agents_do_not_all_wait_at_the_same_empty_tree(self):
        self.group.trees = [TreeLandmark(np.array([115., 100.]), 0)]
        self.step([agent_state(agent_id=i, energy=150) for i in range(3)], 0)
        self.assertFalse(any(task["kind"] == "watch orchard" for task in self.harvest.tasks.values()))

    def test_hungry_camper_leaves_an_unproductive_tree(self):
        tree = TreeLandmark(np.array([115., 100.]), 0)
        self.group.trees = [tree]
        state = agent_state(agent_id=1, energy=40)
        self.step([state], 0)
        self.assertEqual(self.harvest.tasks[1]["kind"], "watch orchard")
        tree.last_seen = 10.1
        self.step([state], 10.1)
        self.assertEqual(self.harvest.tasks[1]["kind"], "local exploration")
        tree.last_seen = 10.2
        self.step([state], 10.2)
        self.assertEqual(self.harvest.tasks[1]["kind"], "local exploration")

    def test_stationary_camper_scans_once_per_six_seconds(self):
        tree = TreeLandmark(np.array([115., 100.]), 0)
        self.group.trees = [tree]
        state = agent_state(agent_id=1, energy=150)
        scan_turns = []
        for step in range(60):
            now = step / 10.
            tree.last_seen = now
            hints, _ = self.step([state], now)
            self.assertEqual(hints[1].vector, (0., 0.))
            if hints[1].scan_while_stationary:
                scan_turns.append(hints[1].look_direction)
        self.assertAlmostEqual(sum(scan_turns), math.tau)

    def test_doomed_trip_cannot_reserve_food_needed_by_another_agent(self):
        states = [agent_state([fruit(100)], agent_id=1, energy=4),
                  agent_state([fruit(100)], agent_id=2, energy=100)]
        self.step(states, 0)
        self.assertNotIn(1, self.harvest.assignments)
        self.assertIn(2, self.harvest.assignments)

    def test_long_food_trip_accounts_for_age_drain(self):
        pose = EstimatedPose(1, 1, np.array([100., 100.]))
        track = FruitTrack(1, np.array([400., 100.]), 0., 0., 0., True, 0.)
        old = agent_state(agent_id=1, age=100., energy=20)
        young = agent_state(agent_id=2, age=10., energy=150)
        self.assertLess(self.harvest._candidate(old, pose, track, 0., .05)[0], 0.)
        self.assertGreater(self.harvest._candidate(young, pose, track, 0., .05)[0], 0.)

    def test_weak_genes_only_breed_during_small_population_recovery(self):
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True, minimum_population=2))
        states = [agent_state(agent_id=i, energy=400) for i in range(2)]
        self.step(states, 0)
        for trait in TRAITS:
            states[1][trait] *= .8
        _, hints = self.step(states, 1)
        self.assertFalse(hints[1].allowed)
        self.poses.pop(0)
        _, hints = self.step(states[1:], 2)
        self.assertTrue(hints[1].allowed)

    def test_uncertain_poses_keep_local_foraging(self):
        self.poses[1] = EstimatedPose(1, 1, np.array([100., 100.]), uncertainty=10)
        hints, _ = self.step([agent_state([fruit(30)], agent_id=1)], 0)
        self.assertNotIn(1, hints)
        self.assertFalse(self.harvest.tracks)

    def test_older_fruit_beats_a_nearer_new_discovery(self):
        # Deliberately wait outside pickup range with enough energy to ripen;
        # a hungry agent held stationary here should trigger stuck recovery.
        state = agent_state([], agent_id=1, energy=300)
        self.step([state], 0)
        state["observations"] = [fruit(45)]
        self.step([state], .1)
        old_id = next(iter(self.harvest.tracks))
        for step in range(2, 120):
            self.poses[1].position[0] = 100 + min(27, step * 9)
            state["observations"] = [fruit(145 - self.poses[1].position[0])]
            self.step([state], step / 10)
        state["observations"] = [fruit(18), fruit(12)]
        hints, _ = self.step([state], 12)
        self.assertEqual(hints[1].track_id, old_id)

    def test_small_colony_can_harvest_early_to_fund_growth(self):
        state = agent_state([fruit(20)], energy=40, agent_id=1)
        hints, _ = self.step([state], 0)
        self.assertFalse(hints[1].waiting)
        self.assertGreater(hints[1].vector[0], 0.)
        self.assertLess(20. - hints[1].vector[0], 10.)

    def test_assignment_accounts_for_observed_biome_travel_cost(self):
        self.planner.estimator.config = SimpleNamespace(biome_movement_factors={"river": .3})
        states = [agent_state([fruit(30)], agent_id=1, biome="river"),
                  agent_state([fruit(30)], agent_id=2, biome="grassland")]
        hints, _ = self.step(states, 0)
        self.assertIsNone(hints[1].track_id)
        self.assertIsNotNone(hints[2].track_id)

    def test_old_scout_role_does_not_override_territory_food_policy(self):
        self.planner.exploration_hints = {1: SimpleNamespace(role="scout")}
        state = agent_state([], agent_id=1, energy=300)
        self.step([state], 0)
        state["observations"] = [fruit(30)]
        hints, _ = self.step([state], 1.)
        self.assertIsNotNone(hints[1].track_id)
        self.assertTrue(hints[1].waiting)
        state["energy"] = 40
        hints, _ = self.step([state], 1.1)
        self.assertIsNotNone(hints[1].track_id)
        self.assertFalse(hints[1].waiting)

    def test_config_rejects_impossible_ripening_and_population_limits(self):
        for kwargs in ({"ripe_energy": 70}, {"minimum_population": 70}, {"lifetime_seconds": 5},
                       {"population_cap_middle": 20}, {"population_cap_late": 5},
                       {"population_middle_seconds": 1800.}, {"population_late_seconds": 800.}):
            with self.assertRaises(ValidationError):
                HarvestConfig(**kwargs)


class GlobalReproductionTests(unittest.TestCase):
    setUp = HarvestTests.setUp

    def plan(self, states, now=0.):
        for state in states:
            agent_id = state["agent_id"]
            self.poses.setdefault(agent_id, EstimatedPose(agent_id, 1, np.array([100., 100.])))
        self.population.update(states, now)
        self.population.population_phase = True
        return self.harvest._breeders(states, self.poses, now, self.population, self.cfg.mechanics, 350.)

    def add_food(self, count, position=(100., 100.), now=0.):
        for track_id in range(count):
            self.harvest.tracks[track_id] = FruitTrack(track_id, np.asarray(position), now, now,
                                                      now - 20., True, 0.)

    def observe_metabolism(self, states, senescent_ids):
        """Confirm upkeep from three public states and known zero-cost actions."""
        for index in range(3):
            samples = []
            for state in states:
                remaining = 2 - index
                age = state["age"] - remaining * .1
                loss = sum(.1 + (age + (tick + 1) * .1) * .01
                           if state["agent_id"] in senescent_ids else .1
                           for tick in range(remaining))
                samples.append(dict(state, age=age, energy=state["energy"] + loss))
            self.harvest.metabolic_tracker.update(samples, index * .1, .1)
            self.harvest.metabolic_tracker.remember_actions([
                SimpleNamespace(agent_id=state["agent_id"], spawn_agent=False,
                                move_distance=0., turn_angle=0.) for state in samples])

    def test_scouting_floor_declines_with_the_world_instead_of_observation_count(self):
        states = [agent_state(agent_id=i, age=10., energy=400) for i in range(12)]
        for now, expected in ((0., 6), (900., 4), (1800., 2)):
            self.plan(states, now)
            self.assertEqual(self.harvest.population_target, expected)
            self.assertEqual(self.harvest.population_plan["scouting_floor"], expected)

    def test_abundant_food_cannot_override_later_population_caps(self):
        states = [agent_state(agent_id=i, age=10., energy=400) for i in range(12)]
        self.add_food(500)
        for now, cap, phase in ((899.9, 12, "early"), (900., 4, "middle"),
                                (1799.9, 4, "middle"), (1800., 2, "late")):
            hints = self.plan(states, now)
            self.assertGreater(self.harvest.population_supply / self.harvest.config.food_budget_per_agent_second, cap)
            self.assertEqual(self.harvest.population_target, cap)
            self.assertEqual(self.harvest.population_plan["target_cap"], cap)
            self.assertEqual(self.harvest.population_plan["population_phase"], phase)
            if phase != "early":
                self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_population_schedule_is_configurable_and_respects_global_limit(self):
        self.harvest.config = HarvestConfig(enabled=True, maximum_population=12,
            population_cap_early=20, population_cap_middle=8, population_cap_late=3,
            population_middle_seconds=50., population_late_seconds=100.)
        states = [agent_state(agent_id=0, age=10., energy=400)]
        self.add_food(500)
        for now, expected in ((0., 12), (50., 8), (100., 3)):
            self.plan(states, now)
            self.assertEqual(self.harvest.population_target, expected)

    def test_late_population_cap_preserves_last_lineage_replacement(self):
        hints = self.plan([agent_state(agent_id=0, age=75., energy=125.)], 2500.)
        self.assertEqual(self.harvest.population_target, 2)
        self.assertTrue(hints[0].allowed)
        self.assertTrue(hints[0].preserve_lineage)
        self.assertEqual(hints[0].minimum_energy_reserve, 20.)

    def test_food_supply_estimate_does_not_collapse_after_one_empty_observation(self):
        states = [agent_state(agent_id=0, age=10., energy=400)]
        self.add_food(100)
        self.plan(states)
        initial = self.harvest.population_target
        self.assertGreater(initial, 6)
        self.harvest.tracks.clear()
        self.plan(states, 1.)
        self.assertGreater(self.harvest.population_target, initial / 2.)
        self.plan(states, 180.)
        self.assertEqual(self.harvest.population_target, 6)

    def test_aging_cohort_reproduces_before_the_current_population_falls(self):
        states = [agent_state(agent_id=i, age=85., energy=180.) for i in range(6)]
        hints = self.plan(states)
        plan = self.harvest.population_plan
        self.assertEqual(plan["population"], plan["target"])
        self.assertLess(plan["viable_in_30_seconds"], plan["population"])
        self.assertTrue(plan["replacement_needed"])
        selected = [hint for hint in hints.values() if hint.allowed]
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0].preserve_lineage)

    def test_real_food_cluster_beats_better_genes_in_a_barren_location(self):
        states = [agent_state(agent_id=i, age=10., energy=350.) for i in range(4)]
        self.plan(states)
        self.poses[1].position = np.array([500., 100.])
        for trait in TRAITS:
            states[0][trait] *= 1.2
            states[1][trait] *= .8
        self.add_food(2, (500., 100.), now=1.)
        hints = self.plan(states, 1.)
        self.assertEqual([i for i, hint in hints.items() if hint.allowed], [1])
        self.assertGreater(self.harvest.population_plan["parent_local_food"], 100.)

    def test_growth_avoids_barren_births_when_existing_agents_are_viable(self):
        states = [agent_state(agent_id=i, age=10., energy=350.) for i in range(4)]
        hints = self.plan(states)
        self.assertGreater(self.harvest.population_target, len(states))
        self.assertFalse(self.harvest.population_plan["critical"])
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_last_aging_parent_can_leave_a_child_before_energy_is_too_low(self):
        hints = self.plan([agent_state(agent_id=0, age=75., energy=125.)])
        self.assertTrue(hints[0].allowed)
        self.assertTrue(hints[0].preserve_lineage)
        self.assertEqual(hints[0].minimum_energy_reserve, 20.)

    def test_replacement_overlap_has_a_population_cap(self):
        states = [agent_state(agent_id=i, age=85., energy=300.) for i in range(9)]
        states[0]["age"] = 10.
        hints = self.plan(states)
        self.assertTrue(self.harvest.population_plan["replacement_needed"])
        self.assertEqual(self.harvest.population_plan["overlap_cap"], 9)
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_healthy_growth_uses_a_global_spacing_interval(self):
        states = [agent_state(agent_id=i, age=10., energy=350.) for i in range(4)]
        self.add_food(4)
        hints = self.plan(states)
        parent = next(i for i, hint in hints.items() if hint.allowed)
        self.assertEqual(self.harvest.planned_birth_interval, 2.5)
        self.harvest.active = True
        self.harvest.remember_actions([SimpleNamespace(agent_id=parent, spawn_agent=True)], 0.)
        hints = self.plan(states, 1.)
        self.assertFalse(any(hint.allowed for hint in hints.values()))
        self.assertEqual(self.harvest.population_plan["reason"], "spacing generations")

    def test_nearby_food_is_not_counted_in_full_for_every_competing_parent(self):
        states = [agent_state(agent_id=i, age=10., energy=200.) for i in range(4)]
        self.plan(states)
        self.add_food(1)
        shares = self.harvest._breeding_food(states, self.poses, 0.)
        self.assertAlmostEqual(sum(shares.values()), 60.)
        self.assertTrue(all(value == 15. for value in shares.values()))

    def test_age_forecast_uses_the_public_onset_distribution(self):
        self.assertEqual(self.harvest._expected_age_drain(0., 60.), 0.)
        self.assertAlmostEqual(self.harvest._expected_age_drain(120., 10.), 125.)
        self.assertGreater(self.harvest._expected_age_drain(70., 30.), 0.)

    def test_pessimistic_forecast_does_not_create_barren_births_among_hungry_young(self):
        states = [agent_state(agent_id=0, age=70., energy=300.)]
        states += [agent_state(agent_id=i, age=10., energy=75.) for i in range(1, 6)]
        hints = self.plan(states)
        self.assertTrue(self.harvest.population_plan["critical"])
        self.assertFalse(self.harvest.population_plan["barren_birth_allowed"])
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_last_aging_cohort_can_replace_above_its_overlap_cap(self):
        states = [agent_state(agent_id=i, age=90., energy=180.) for i in range(10)]
        self.observe_metabolism(states, set(range(10)))
        hints = self.plan(states)
        plan = self.harvest.population_plan
        self.assertGreater(plan["population"], plan["overlap_cap"])
        self.assertTrue(plan["last_cohort_rescue"])
        self.assertEqual(sum(hint.allowed for hint in hints.values()), 1)
        self.assertEqual(plan["reason"], "transfer senescent energy to successors")

    def test_funded_cohort_slots_stop_at_the_desired_healthy_successors(self):
        states = [agent_state(agent_id=i, age=90., energy=180.) for i in range(10)]
        self.observe_metabolism(states, set(range(10)))
        self.assertTrue(any(hint.allowed for hint in self.plan(states).values()))
        states.append(agent_state(agent_id=10, age=0., energy=75.))
        hints = self.plan(states, .1)
        self.assertFalse(self.harvest.population_plan["last_cohort_rescue"])
        self.assertEqual(self.harvest.population_plan["healthy_young"], 1)
        self.assertTrue(any(hint.allowed for hint in hints.values()))
        states.extend(agent_state(agent_id=i, age=0., energy=75.) for i in range(11, 14))
        hints = self.plan(states, .2)
        self.assertEqual(self.harvest.population_plan["healthy_young"], 4)
        self.assertFalse(self.harvest.population_plan["funded_successor_transfer"])
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_last_cohort_exception_respects_the_hard_population_limit(self):
        self.harvest.config = self.harvest.config.model_copy(update={
            "minimum_population": 6, "maximum_population": 10})
        states = [agent_state(agent_id=i, age=90., energy=180.) for i in range(10)]
        self.observe_metabolism(states, set(range(10)))
        hints = self.plan(states)
        self.assertTrue(self.harvest.population_plan["critical"])
        self.assertFalse(self.harvest.population_plan["last_cohort_rescue"])
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_dying_parents_transfer_energy_to_four_successors_before_reserves_vanish(self):
        states = [agent_state(agent_id=i, age=100., energy=150.) for i in range(9)]
        self.observe_metabolism(states, set(range(9)))
        self.harvest.active = True
        chosen = []
        for now in range(4):
            hints = self.plan(states, float(now))
            parent = next(i for i, hint in hints.items() if hint.allowed)
            chosen.append(parent)
            self.assertEqual(hints[parent].minimum_energy_reserve, 5.)
            self.assertEqual(self.harvest.population_plan["successor_cap"], 13)
            self.assertEqual(self.harvest.planned_birth_interval, 1.)
            state = next(state for state in states if state["agent_id"] == parent)
            state["energy"] -= 100.
            states.append(agent_state(agent_id=9 + now, age=0., energy=75.))
            action = SimpleNamespace(agent_id=parent, spawn_agent=True,
                                     move_distance=0., turn_angle=0.)
            self.population.remember_actions([action], float(now))
            self.harvest.remember_actions([action], float(now))
            self.assertFalse(any(h.allowed for h in self.plan(states, now + .5).values()))
            for state in states:
                state["age"] += 1.
                state["energy"] -= 11. if state["agent_id"] < 9 else 1.
        self.assertEqual(len(set(chosen)), 4)
        # Keep unused parents well funded: actual successors, not exhausted
        # parent reserves or pessimistic forecasts, must close the exception.
        for state in states:
            if state["agent_id"] < 9 and state["agent_id"] not in chosen:
                state["energy"] = 300.
        hints = self.plan(states, 4.)
        self.assertEqual(self.harvest.population_plan["healthy_young"], 4)
        self.assertEqual(len(states), 13)
        self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_last_confirmed_senescent_parent_can_transfer_110_energy(self):
        states = [agent_state(agent_id=0, age=100., energy=110.)]
        self.observe_metabolism(states, {0})
        hint = self.plan(states)[0]
        self.assertTrue(hint.allowed)
        self.assertTrue(hint.preserve_lineage)
        self.assertEqual(hint.minimum_energy_reserve, 5.)
        self.assertEqual(hint.energy_threshold, 107.)

    def test_age_alone_cannot_open_funded_successor_slots(self):
        for age in (55., 100.):
            with self.subTest(age=age):
                self.setUp()
                states = [agent_state(agent_id=i, age=age, energy=130.) for i in range(9)]
                self.observe_metabolism(states, set())
                hints = self.plan(states)
                self.assertTrue(self.harvest.population_plan["critical"])
                self.assertFalse(self.harvest.population_plan["funded_successor_transfer"])
                self.assertFalse(any(hint.allowed for hint in hints.values()))

    def test_extra_successor_slot_cannot_select_a_healthy_better_gene_parent(self):
        states = [agent_state(agent_id=i, age=100., energy=180.) for i in range(9)]
        self.observe_metabolism(states, {0})
        self.plan(states)
        for trait in TRAITS:
            states[1][trait] *= 1.5
        self.add_food(6, now=1.)
        hints = self.plan(states, 1.)
        self.assertTrue(self.harvest.population_plan["critical"])
        self.assertEqual([i for i, hint in hints.items() if hint.allowed], [0])

    def test_forecast_replacement_can_still_use_a_real_food_cluster(self):
        states = [agent_state(agent_id=0, age=70., energy=300.)]
        states += [agent_state(agent_id=i, age=10., energy=75.) for i in range(1, 6)]
        self.plan(states)
        self.poses[0].position = np.array([500., 100.])
        self.add_food(2, (500., 100.), 1.)
        hints = self.plan(states, 1.)
        self.assertFalse(self.harvest.population_plan["barren_birth_allowed"])
        self.assertEqual([i for i, hint in hints.items() if hint.allowed], [0])

    def test_poor_future_forecast_does_not_spend_young_parents_travel_reserves(self):
        states = [agent_state(agent_id=i, age=40., energy=180.) for i in range(6)]
        self.add_food(4)
        hints = self.plan(states)
        self.assertLess(self.harvest.population_plan["viable_in_60_seconds"],
                        self.harvest.population_plan["desired_young"])
        self.assertTrue(self.harvest.population_plan["replacement_needed"])
        self.assertFalse(any(hint.allowed for hint in hints.values()))
        states[0]["energy"] = 300.
        hints = self.plan(states, .1)
        self.assertTrue(hints[0].allowed)
        self.assertFalse(hints[0].preserve_lineage)
        self.assertEqual(hints[0].minimum_energy_reserve, 100.)
        self.assertGreaterEqual(hints[0].energy_threshold, 220.)

    def test_aging_parents_can_spend_the_lower_reserve_for_the_same_forecast(self):
        states = [agent_state(agent_id=i, age=55., energy=180.) for i in range(6)]
        self.add_food(4)
        hints = self.plan(states)
        chosen = [hint for hint in hints.values() if hint.allowed]
        self.assertEqual(len(chosen), 1)
        self.assertTrue(chosen[0].preserve_lineage)
        self.assertEqual(chosen[0].minimum_energy_reserve, 40.)

    def test_actual_near_extinction_can_still_use_a_young_parent(self):
        hints = self.plan([agent_state(agent_id=0, age=10., energy=150.)])
        self.assertTrue(self.harvest.population_plan["critical"])
        self.assertTrue(hints[0].allowed)
        self.assertTrue(hints[0].preserve_lineage)
        self.assertEqual(hints[0].minimum_energy_reserve, 20.)


class PredatorFreeTests(unittest.TestCase):
    def test_disables_initial_later_and_explicit_spawns_and_survives_reset(self):
        sim = SimulationCore(env_width=300, env_height=240, starting_agents=0,
                             starting_predators=3, starting_trees=0, seed=1, predators_enabled=False)
        self.assertFalse(sim.env.predators)
        self.assertIsNone(sim.env.spawn_predator())
        sim.env.time = 1e9  # The normal random spawn check will always fire.
        sim.env.non_agent_step(.1)
        self.assertFalse(sim.env.predators)
        sim.reset()
        self.assertFalse(sim.env.predators_enabled)
        self.assertFalse(sim.env.predators)


if __name__ == "__main__":
    unittest.main()

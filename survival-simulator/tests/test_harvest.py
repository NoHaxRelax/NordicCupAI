import math
from types import SimpleNamespace
import unittest

import numpy as np
from pydantic import ValidationError

from src.core import SimulationCore
from src.utils.controllers.expert_policy import ExpertPolicy, load_config
from src.utils.controllers.harvest import HarvestConfig, HarvestCoordinator
from src.utils.controllers.policy_inputs import HarvestHint
from src.utils.controllers.population import PopulationTracker, TRAITS
from src.utils.controllers.world_estimator import EstimatedPose, MapGroup
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
        state = agent_state([fruit(20)], energy=300, agent_id=1)
        hints, _ = self.step([state], 0)
        self.assertTrue(hints[1].waiting)
        self.assertAlmostEqual(hints[1].vector[0], 2)
        state["energy"] = 40
        hints, _ = self.step([state], .1)
        self.assertFalse(hints[1].waiting)
        self.assertAlmostEqual(hints[1].vector[0], 20)

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
        states = [agent_state(agent_id=i, energy=400) for i in range(3)]
        self.step(states, 0)
        for trait in TRAITS:
            states[2][trait] *= 1.2
        _, hints = self.step(states, 1)
        self.assertEqual([i for i, hint in hints.items() if hint.allowed], [2])
        states[2]["energy"] = 190
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
        self.harvest.remember_actions([SimpleNamespace(agent_id=0, spawn_agent=True)], 1)
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

    def test_gene_backup_can_exceed_food_target_but_stops_with_young_carriers(self):
        self.harvest = HarvestCoordinator(HarvestConfig(enabled=True, minimum_population=2))
        states = [agent_state(agent_id=i, energy=400) for i in range(2)]
        self.step(states, 0)
        states[1]["speed"] *= 1.5
        _, hints = self.step(states, 40)
        self.assertTrue(hints[1].allowed)  # Food target is already met.
        self.assertEqual(self.harvest.gene_backup_ids, {1})
        for i in (2, 3):
            child = dict(states[1], agent_id=i, age=5)
            states.append(child)
            self.poses[i] = EstimatedPose(i, 1, np.array([100., 100.]))
        self.population.update(states, 41)
        hints = self.harvest._breeders(states, self.poses, 41, self.population, self.cfg.mechanics, 350)
        self.assertFalse(any(h.allowed for h in hints.values()))

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
        state = agent_state([fruit(45)], agent_id=1, energy=300)
        self.step([state], 0)
        old_id = next(iter(self.harvest.tracks))
        for step in range(1, 120):
            self.poses[1].position[0] = 100 + min(27, step * 9)
            state["observations"] = [fruit(145 - self.poses[1].position[0])]
            self.step([state], step / 10)
        state["observations"] = [fruit(18), fruit(12)]
        hints, _ = self.step([state], 12)
        self.assertEqual(hints[1].track_id, old_id)

    def test_small_colony_can_harvest_early_to_fund_growth(self):
        state = agent_state([fruit(20)], energy=150, agent_id=1)
        hints, _ = self.step([state], 0)
        self.assertFalse(hints[1].waiting)
        self.assertEqual(hints[1].vector, (20., 0.))

    def test_assignment_accounts_for_observed_biome_travel_cost(self):
        self.planner.estimator.config = SimpleNamespace(biome_movement_factors={"river": .3})
        states = [agent_state([fruit(30)], agent_id=1, biome="river"),
                  agent_state([fruit(30)], agent_id=2, biome="grassland")]
        hints, _ = self.step(states, 0)
        self.assertIsNone(hints[1].track_id)
        self.assertIsNotNone(hints[2].track_id)

    def test_old_scout_role_does_not_override_territory_food_policy(self):
        self.planner.exploration_hints = {1: SimpleNamespace(role="scout")}
        state = agent_state([fruit(30)], agent_id=1, energy=300)
        hints, _ = self.step([state], 0)
        self.assertIsNotNone(hints[1].track_id)
        self.assertTrue(hints[1].waiting)
        state["energy"] = 40
        hints, _ = self.step([state], 1)
        self.assertIsNotNone(hints[1].track_id)
        self.assertFalse(hints[1].waiting)

    def test_config_rejects_impossible_ripening_and_population_limits(self):
        for kwargs in ({"ripe_energy": 70}, {"minimum_population": 70}, {"lifetime_seconds": 5}):
            with self.assertRaises(ValidationError):
                HarvestConfig(**kwargs)


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

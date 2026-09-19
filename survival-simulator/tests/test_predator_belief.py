import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.elements.predator import Predator
from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from src.utils.controllers.predator_belief import (
    EMPTY_OBSTACLES, PREDATOR_MAX_ENERGY, PREDATOR_SIZE, PREDATOR_SPRINT, PREDATOR_WALK,
    Obstacles, PredatorBeliefConfig, PredatorTracker, blocked, occupied, predator_sightings,
    segments_clear, wrap,
)
from src.utils.controllers.world_estimator import EdgeLandmark, EstimatedPose, MapGroup
from test_expert_policy import agent_state, fruit
from test_global_planner import planner_config


FIXTURES = Path(__file__).parent / "fixtures"
# compute_visibility indexes the corner array unconditionally, so a scene needs
# at least one edge. The real environment always encloses the world; put the
# walls outside every vision radius so they cannot steer these scenarios.
ARENA = (((0., 0.), (2000., 0.)), ((2000., 0.), (2000., 2000.)),
         ((2000., 2000.), (0., 2000.)), ((0., 2000.), (0., 0.)))
CENTRE = 600.


class Stand:
    """Minimal stand-in for an Agent as the predator's sensors read one."""

    def __init__(self, x, y, direction=0., size=5., agent_id=7):
        self.x, self.y, self.direction, self.size = float(x), float(y), float(direction), size
        self.agent_id = agent_id


def step_engine(creature, agents, edges=ARENA):
    """Apply Predator.step the way Environment does, without terrain or rocks.

    Mirrors update_entity_position/update_entity_direction for the no-obstacle,
    multiplier-one case: move first on the heading the tick started with, then
    turn. Resting is driven by Environment's loop, not by step, so it is left
    to the caller.
    """
    signals = creature.step(creature.observe(agents=agents, edges=list(edges)))
    if "move" in signals:
        distance = min(max(signals["move"], 0.), creature.sprint_speed)
        if creature.energy < creature.max_energy / 5 and distance > creature.speed:
            distance = creature.speed
        creature.energy -= (distance * .05 if distance <= creature.speed
                            else creature.speed * .05 + (distance - creature.speed) * .5)
        direction = (creature.direction if signals["direction"] is None
                     else creature.direction + signals["direction"])
        creature.x += distance * math.cos(direction)
        creature.y += distance * math.sin(direction)
    if "turn" in signals:
        creature.direction += signals["turn"]
        creature.energy -= min(math.pi, abs(signals["turn"])) / math.tau
    return signals


def make_predator(x, y, direction=0., energy=PREDATOR_MAX_ENERGY):
    creature = Predator(x, y, rng=random.Random(0))
    creature.direction, creature.energy, creature.resting = float(direction), energy, False
    return creature


def tracker(**changes):
    """A noise-free tracker on flat terrain, so propagation is deterministic."""
    defaults = dict(enabled=True, particles=8, position_noise=0., heading_noise=0.,
                    measurement_uncertainty=1e-9, heading_uncertainty=1e-9)
    defaults.update(changes)
    return PredatorTracker(PredatorBeliefConfig(**defaults), movement_factors={"grassland": 1.})


def seeded(instance, position, heading, now=0., group=None, group_id=1):
    """Start a track from a known pose with every particle at full energy."""
    with patch.object(instance, "_sample_energy",
                      side_effect=lambda count: np.full(count, PREDATOR_MAX_ENERGY)):
        track = instance._new_track(group_id, np.asarray(position, dtype=float), heading,
                                    now, group, {7})
    instance.tracks[track.track_id] = track
    return track


def pose(agent_id, position, heading=0., group_id=1):
    return EstimatedPose(agent_id, group_id, np.asarray(position, dtype=float), heading)


def state(agent_id, observations=(), **stats):
    return agent_state(observations, agent_id=agent_id, **stats)


def rock(*segments, rectangles=()):
    """Mapped rock in the belief's own frame, as the tracker consumes it."""
    edges = (np.asarray(segments, dtype=float).reshape(-1, 2, 2) if segments
             else np.empty((0, 2, 2)))
    boxes = (np.asarray(rectangles, dtype=float).reshape(-1, 4) if len(rectangles)
             else np.empty((0, 4)))
    return Obstacles(edges, boxes)


def wall(x, y0, y1):
    """A single vertical face, the way spawn_obstacle emits one: toward +y."""
    return ((x, y0), (x, y1))


def sighting(distance, angle=0., rel_dir=None):
    observation = {"type": "Predator", "distance": distance, "angle": angle}
    if rel_dir is not None:
        observation["rel_dir"] = rel_dir
    return observation


class ForwardModelTests(unittest.TestCase):
    """The belief is only worth having if it reproduces Predator.step."""

    def assert_tracks_engine(self, creature, agents, ticks=8, tolerance=1e-6):
        instance = tracker()
        track = seeded(instance, (creature.x, creature.y), creature.direction)
        positions = np.array([[agent.x, agent.y] for agent in agents], dtype=float).reshape(-1, 2)
        headings = np.array([agent.direction for agent in agents], dtype=float).reshape(-1)
        for tick in range(ticks):
            step_engine(creature, agents)
            instance._advance(track, positions, headings, EMPTY_OBSTACLES, None)
            with self.subTest(tick=tick):
                offsets = track.positions - np.array([creature.x, creature.y])
                self.assertLess(float(np.max(np.hypot(offsets[:, 0], offsets[:, 1]))), tolerance)
                self.assertLess(float(np.max(np.abs(wrap(track.headings - creature.direction)))),
                                tolerance)
        return instance, track

    def test_direct_chase_matches_engine_when_the_agent_looks_away(self):
        # Agent facing +x with the predator behind it: |looking| > pi/2 chases.
        agent = Stand(CENTRE + 60, CENTRE, direction=0.)
        creature = make_predator(CENTRE, CENTRE, direction=0.)
        self.assert_tracks_engine(creature, [agent])
        self.assertGreater(creature.x, CENTRE)

    def test_offset_chase_reproduces_the_clamped_turn(self):
        # Broadside inside the hearing radius, which is not cone-limited, so
        # the bearing is large enough for the +-0.3 turn clamp to bind.
        agent = Stand(CENTRE, CENTRE + 55, direction=0.)
        creature = make_predator(CENTRE, CENTRE, direction=0.)
        self.assert_tracks_engine(creature, [agent], ticks=10)
        self.assertGreater(abs(creature.direction), .3)

    def test_watched_pivot_matches_engine_when_the_agent_looks_back(self):
        # Agent looking back at a predator 120 away: outside the 90-unit
        # hearing override it circles instead of charging. The gaze is offset
        # so that -np.sign is unambiguous; exact zero is covered separately.
        agent = Stand(CENTRE + 120, CENTRE, direction=math.pi - .5)
        creature = make_predator(CENTRE, CENTRE, direction=0.)
        self.assert_tracks_engine(creature, [agent], ticks=6)
        self.assertGreater(abs(creature.y - CENTRE), 1.)

    def test_an_exactly_zero_gaze_keeps_both_pivot_directions_alive(self):
        # -np.sign(0) leaves the native pivot side undetermined and roundoff
        # decides it, so the cloud has to carry both instead of picking one.
        agent = Stand(CENTRE + 120, CENTRE, direction=math.pi)
        instance = tracker(particles=64, measurement_uncertainty=2.)
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        instance._advance(track, np.array([[agent.x, agent.y]]),
                          np.array([agent.direction]), EMPTY_OBSTACLES, None)
        above = track.positions[:, 1] > CENTRE + 5.
        below = track.positions[:, 1] < CENTRE - 5.
        self.assertTrue(bool(above.any()) and bool(below.any()))

    def test_closest_agent_wins_exactly_as_the_engine_chooses_it(self):
        near = Stand(CENTRE + 60, CENTRE, 0., agent_id=7)
        far = Stand(CENTRE, CENTRE + 110, 0., agent_id=8)
        creature = make_predator(CENTRE, CENTRE, direction=0.)
        self.assert_tracks_engine(creature, [far, near], ticks=6)

    def test_an_unsensed_agent_does_not_attract_the_belief(self):
        # Behind the predator and past its hearing radius, so it is invisible
        # and the predator wanders. The wander draw comes from the engine's own
        # RNG and cannot be reproduced, so only the branch is asserted here.
        agent = Stand(CENTRE - 200, CENTRE, direction=0.)
        instance = tracker()
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        for _ in range(4):
            instance._advance(track, np.array([[agent.x, agent.y]]),
                              np.array([agent.direction]), EMPTY_OBSTACLES, None)
        # Walking, not sprinting, and away from the agent rather than onto it.
        self.assertGreater(float(track.mean[0]), CENTRE + 3 * PREDATOR_WALK)
        self.assertLess(float(track.mean[0]), CENTRE + 4 * PREDATOR_WALK + 1.)

    def test_wander_keeps_the_heading_and_walks(self):
        instance = tracker()
        track = seeded(instance, (CENTRE, CENTRE), .4)
        for _ in range(5):
            instance._advance(track, np.empty((0, 2)), np.empty(0), EMPTY_OBSTACLES, None)
        travelled = np.hypot(*(track.positions - np.array([CENTRE, CENTRE])).T)
        # Five walking ticks of 11 units, less the small wander turns.
        self.assertTrue(np.all(travelled > 45.))
        self.assertTrue(np.all(travelled <= 5 * PREDATOR_WALK))

    def test_low_energy_particles_fall_back_to_walking_speed(self):
        instance = tracker()
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        # Below a fifth of maximum the engine caps the step at walking speed,
        # so one cloud carries both hypotheses until a sighting settles it.
        track.energies = np.array([PREDATOR_MAX_ENERGY] * 4 + [10.] * 4)
        instance._advance(track, np.array([[CENTRE + 50., CENTRE]]), np.array([0.]),
                          EMPTY_OBSTACLES, None)
        np.testing.assert_allclose(track.positions[:4, 0], CENTRE + PREDATOR_SPRINT)
        np.testing.assert_allclose(track.positions[4:, 0], CENTRE + PREDATOR_WALK)

    def test_exhausted_particles_rest_and_recover_without_moving(self):
        instance = tracker()
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        track.energies = np.full(8, .2)
        agents, headings = np.array([[CENTRE + 10., CENTRE]]), np.array([0.])
        instance._advance(track, agents, headings, EMPTY_OBSTACLES, None)
        self.assertTrue(bool(track.resting.all()))
        resting_at = track.positions.copy()
        instance._advance(track, agents, headings, EMPTY_OBSTACLES, None)
        np.testing.assert_allclose(track.positions, resting_at)
        self.assertTrue(np.all(track.energies > .2))

    def test_particles_do_not_walk_into_mapped_rock(self):
        face = rock(wall(CENTRE + 20., CENTRE - 40., CENTRE + 40.))
        instance = tracker()
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        instance._advance(track, np.array([[CENTRE + 50., CENTRE]]), np.array([0.]), face, None)
        self.assertFalse(bool(occupied(track.positions, face, PREDATOR_SIZE).any()))

    def test_edge_avoidance_turns_an_idle_predator_off_the_wall(self):
        face = rock(wall(CENTRE + 80., CENTRE - 40., CENTRE + 40.))
        instance = tracker()
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        instance._advance(track, np.empty((0, 2)), np.empty(0), face, None)
        # Engine: turn(+-pi / distance) away from the nearest face point.
        self.assertGreater(float(np.min(np.abs(wrap(track.headings)))), .04)


class RockGeometryTests(unittest.TestCase):
    def test_line_of_sight_stops_at_a_face_but_not_beside_it(self):
        face = rock(wall(100., 0., 200.))
        blockedline = segments_clear(np.array([[50., 100.]]), np.array([[150., 100.]]), face.edges)
        beside = segments_clear(np.array([[50., 300.]]), np.array([[150., 300.]]), face.edges)
        self.assertFalse(bool(blockedline[0]))
        self.assertTrue(bool(beside[0]))

    def test_a_segment_ending_on_a_face_is_not_blocked_by_it(self):
        # Otherwise a predator standing against a wall could never be seen.
        face = rock(wall(100., 0., 200.))
        touching = segments_clear(np.array([[50., 100.]]), np.array([[100., 100.]]), face.edges)
        self.assertTrue(bool(touching[0]))

    def test_a_rock_interior_is_forbidden_not_just_the_strip_beside_a_face(self):
        # Environment._in_obstacle tests the filled rectangle, so a point deep
        # inside a wide rock is blocked even though no face is within a radius.
        box = rock(rectangles=[(0., 0., 200., 200.)])
        self.assertTrue(bool(occupied(np.array([[100., 100.]]), box, PREDATOR_SIZE)[0]))
        self.assertFalse(bool(occupied(np.array([[260., 100.]]), box, PREDATOR_SIZE)[0]))
        # The clearance test alone would have called the interior clear.
        faces = rock(wall(0., 0., 200.), wall(200., 0., 200.))
        self.assertFalse(bool(blocked(np.array([[100., 100.]]), faces.edges, PREDATOR_SIZE)[0]))

    def test_the_inflated_rectangle_keeps_a_predator_a_radius_clear(self):
        box = rock(rectangles=[(0., 0., 100., 100.)])
        self.assertTrue(bool(occupied(np.array([[105., 50.]]), box, PREDATOR_SIZE)[0]))
        self.assertFalse(bool(occupied(np.array([[115., 50.]]), box, PREDATOR_SIZE)[0]))


class OcclusionTests(unittest.TestCase):
    """A particle must not chase an agent it could not have seen."""

    def _path_length(self, obstacles, ticks=4):
        """Total distance the belief travels over `ticks`.

        Step length is the sharp discriminator between the branches: a chase
        requests the 15-unit sprint, while every branch that has not sensed an
        agent walks at 11. Closing distance is not, because the engine's
        edge-avoidance branch also walks forward while it turns.
        """
        agent = np.array([[CENTRE + 150., CENTRE]])
        instance = tracker(particles=8)
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        travelled, previous = 0., track.mean.copy()
        for _ in range(ticks):
            instance._advance(track, agent, np.array([0.]), obstacles, None)
            travelled += float(np.hypot(*(track.mean - previous)))
            previous = track.mean.copy()
        return travelled

    def test_a_clear_line_sprints_and_a_blocked_one_only_walks(self):
        # Agent 150 ahead and facing away, so with sight the predator charges.
        ticks = 4
        open_ground = self._path_length(EMPTY_OBSTACLES, ticks)
        hidden = self._path_length(rock(wall(CENTRE + 70., CENTRE - 60., CENTRE + 60.)), ticks)
        self.assertAlmostEqual(open_ground, ticks * PREDATOR_SPRINT, delta=1.)
        self.assertAlmostEqual(hidden, ticks * PREDATOR_WALK, delta=1.)

    def test_hearing_is_never_occluded(self):
        # Creature.observe appends everything inside the hearing radius before
        # any visibility test, so a wall between does not hide it.
        agent = np.array([[CENTRE + 55., CENTRE]])
        instance = tracker(particles=8)
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        instance._advance(track, agent, np.array([0.]),
                          rock(wall(CENTRE + 30., CENTRE - 60., CENTRE + 60.)), None)
        # It moved, and it moved toward the agent rather than wandering off.
        self.assertGreater(float(track.mean[0]), CENTRE + PREDATOR_WALK)

    def test_switching_occlusion_off_restores_seeing_through_rock(self):
        agent = np.array([[CENTRE + 150., CENTRE]])
        face = rock(wall(CENTRE + 70., CENTRE - 60., CENTRE + 60.))
        instance = tracker(particles=8, occlusion=False, collision_attempts=0)
        track = seeded(instance, (CENTRE, CENTRE), 0.)
        instance._advance(track, agent, np.array([0.]), face, None)
        self.assertGreater(float(track.mean[0]), CENTRE + PREDATOR_WALK)

    def test_a_sighting_rules_out_particles_the_observer_could_not_see(self):
        # A face at x=60 up to y=40 leaves a gap above it. The observer at the
        # origin saw the predator by sight, so a clear line existed: particles
        # reachable only through the face are impossible, not merely unlikely.
        instance = tracker(particles=8)
        face = rock(wall(60., -300., 40.))
        track = seeded(instance, (120., 100.), 0.)
        track.positions = np.array([[120., 100.]] * 4 + [[120., -100.]] * 4, dtype=float)
        track.weights = np.full(8, 1. / 8)
        kept = instance._reweight(track, np.array([120., 100.]), None,
                                  np.array([0., 0.]), face)
        self.assertTrue(kept)
        self.assertAlmostEqual(float(track.weights[4:].sum()), 0., places=9)
        self.assertAlmostEqual(float(track.weights[:4].sum()), 1., places=9)

    def test_negative_evidence_spares_particles_hidden_behind_rock(self):
        # The agent is looking straight at the belief but a face is between.
        for hidden, expected in ((True, 1), (False, 0)):
            with self.subTest(hidden=hidden):
                instance = tracker(particles=32, negative_evidence_margin=0.)
                seeded(instance, (150., 100.), 0.)
                instance.last_time = 0.
                group = MapGroup(1)
                if hidden:
                    # Between the agent at x=100 and the belief at x=150.
                    group.edges = [EdgeLandmark(np.array([120., 0.]), np.array([120., 200.]), 0.)]
                    group._edge_revision += 1
                instance.update({1: group}, {7: pose(7, (100., 100.))},
                                [state(7, [fruit(3.)])], .1)
                self.assertEqual(len(instance.tracks), expected)


class SightingTests(unittest.TestCase):
    def test_rel_dir_recovers_the_predator_heading_in_the_group_frame(self):
        instance = tracker()
        observer = pose(7, (100., 100.), heading=.5)
        # A predator 50 ahead of the observer, facing straight back at it.
        offset = np.array([50. * math.cos(.5), 50. * math.sin(.5)])
        expected = wrap(math.atan2(-offset[1], -offset[0]))
        instance.update({1: MapGroup(1)}, {7: observer},
                        [state(7, [sighting(50., 0., rel_dir=0.)])], 0.)
        track = next(iter(instance.tracks.values()))
        np.testing.assert_allclose(track.mean, observer.position + offset, atol=1e-3)
        self.assertTrue(track.heading_known)
        self.assertLess(abs(wrap(float(track.headings.mean()) - expected)), 1e-3)

    def test_a_sighting_without_rel_dir_starts_heading_blind(self):
        instance = tracker(particles=64)
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                        [state(7, [sighting(50.)])], 0.)
        track = next(iter(instance.tracks.values()))
        self.assertFalse(track.heading_known)
        # Headings drawn across the full circle, not collapsed onto one guess.
        self.assertGreater(float(track.headings.std()), 1.)

    def test_sightings_past_the_reactive_radius_are_kept(self):
        # The 100-unit predator_danger_radius gate dropped these outright.
        kept = predator_sightings([sighting(180.), sighting(300.)], 260.)
        self.assertEqual([round(row[0]) for row in kept], [180])

    def test_malformed_and_non_finite_observations_are_ignored(self):
        rows = predator_sightings([
            {"type": "Predator", "distance": "x", "angle": 0.},
            {"type": "Predator", "angle": 0.},
            {"type": "Predator", "distance": float("nan"), "angle": 0.},
            {"type": "Fruit", "distance": 5., "angle": 0.},
            sighting(20., .1, rel_dir=float("inf")),
        ], 260.)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][2])

    def test_two_agents_seeing_one_predator_keep_a_single_track(self):
        instance = tracker()
        poses = {7: pose(7, (100., 100.)), 8: pose(8, (100., 160.))}
        instance.update({1: MapGroup(1)}, poses, [
            state(7, [sighting(30., math.pi / 2)]),
            state(8, [sighting(30., -math.pi / 2)]),
        ], 0.)
        self.assertEqual(len(instance.tracks), 1)
        track = next(iter(instance.tracks.values()))
        self.assertEqual(track.observers, {7, 8})
        np.testing.assert_allclose(track.mean, [100., 130.], atol=1e-3)

    def test_two_separate_predators_keep_two_tracks(self):
        instance = tracker()
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                        [state(7, [sighting(60., 0.), sighting(60., math.pi)])], 0.)
        self.assertEqual(len(instance.tracks), 2)

    def test_a_correcting_sighting_pulls_the_belief_onto_the_measurement(self):
        instance = tracker(measurement_uncertainty=4., position_noise=.5)
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(160., 0., rel_dir=math.pi)])], 0.)
        first = next(iter(instance.tracks.values())).track_id
        # A tick of motion, then a fresh bearing on the same predator.
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(172., 0., rel_dir=math.pi)])], .1)
        self.assertEqual(len(instance.tracks), 1)
        track = next(iter(instance.tracks.values()))
        self.assertEqual(track.track_id, first)
        self.assertEqual(track.sightings, 2)
        np.testing.assert_allclose(track.mean, [172., 0.], atol=4.)

    def test_a_jump_beyond_reach_starts_a_second_track(self):
        # No predator covers 400 units in a tick, so this cannot be the same one.
        # Absence is switched off so only the gate can separate them.
        instance = tracker(negative_evidence=False)
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(150., 0., rel_dir=math.pi)])], 0.)
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(150., math.pi, rel_dir=math.pi)])], .1)
        self.assertEqual(len(instance.tracks), 2)


class NegativeEvidenceTests(unittest.TestCase):
    def test_silence_inside_the_hearing_radius_removes_particles(self):
        # Hearing is reported before any visibility test, so silence is exact.
        instance = tracker(particles=32, negative_evidence_margin=0.)
        seeded(instance, (100., 100.), 0.)
        instance.last_time = 0.
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                        [state(7, [fruit(3.)])], .1)
        self.assertEqual(instance.tracks, {})

    def test_a_live_sighting_does_not_erase_its_own_track(self):
        instance = tracker(particles=32, negative_evidence_margin=0.)
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                        [state(7, [sighting(20., 0., rel_dir=math.pi)])], 0.)
        self.assertEqual(len(instance.tracks), 1)
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                        [state(7, [sighting(20., 0., rel_dir=math.pi)])], .1)
        self.assertEqual(len(instance.tracks), 1)

    def test_belief_outside_the_hearing_radius_survives_a_quiet_tick(self):
        instance = tracker(particles=32)
        seeded(instance, (400., 400.), 0.)
        instance.last_time = 0.
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                        [state(7, [fruit(3.)])], .1)
        self.assertEqual(len(instance.tracks), 1)

    def test_an_agent_reporting_nothing_at_all_is_not_treated_as_silence(self):
        # An empty observation list can mean a skipped refresh, not quiet.
        instance = tracker(particles=32, negative_evidence_margin=0.)
        seeded(instance, (100., 100.), 0.)
        instance.last_time = 0.
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))}, [state(7)], .1)
        self.assertEqual(len(instance.tracks), 1)

    def test_vision_cone_silence_is_ignored_unless_it_is_switched_on(self):
        # A predator behind a rock inside the cone is unobserved but present.
        for enabled, expected in ((False, 1), (True, 0)):
            with self.subTest(vision_negative_evidence=enabled):
                instance = tracker(particles=32, negative_evidence_margin=0.,
                                   vision_negative_evidence=enabled)
                seeded(instance, (250., 100.), 0.)
                instance.last_time = 0.
                instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.))},
                                [state(7, [fruit(3.)])], .1)
                self.assertEqual(len(instance.tracks), expected)


class LifecycleTests(unittest.TestCase):
    def test_a_stale_track_expires(self):
        instance = tracker(max_unseen_seconds=.3)
        seeded(instance, (400., 400.), 0.)
        instance.last_time = 0.
        for now in (.1, .2, .3, .4):
            instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))}, [state(7)], now)
        self.assertEqual(instance.tracks, {})

    def test_a_dispersed_track_is_dropped_rather_than_reported_vaguely(self):
        instance = tracker(max_spread=5., particles=32, position_noise=8.)
        seeded(instance, (400., 400.), 0.)
        instance.last_time = 0.
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))}, [state(7)], .1)
        self.assertEqual(instance.tracks, {})

    def test_track_count_is_capped(self):
        instance = tracker(max_tracks=2)
        for index in range(5):
            seeded(instance, (500. + 200. * index, 500.), 0.)
        instance.last_time = 0.
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))}, [state(7)], .1)
        self.assertLessEqual(len(instance.tracks), 2)

    def test_a_repeated_timestamp_does_not_propagate_twice(self):
        instance = tracker()
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(160., 0., rel_dir=math.pi)])], .1)
        before = next(iter(instance.tracks.values())).positions.copy()
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(160., 0., rel_dir=math.pi)])], .1)
        np.testing.assert_allclose(before, next(iter(instance.tracks.values())).positions)

    def test_a_rewound_clock_and_an_empty_population_clear_the_belief(self):
        for boundary in ("rewind", "empty"):
            with self.subTest(boundary=boundary):
                instance = tracker()
                instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                                [state(7, [sighting(160.)])], 10.)
                self.assertEqual(len(instance.tracks), 1)
                if boundary == "rewind":
                    instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))}, [state(7)], 1.)
                else:
                    instance.update({}, {}, [], 10.1)
                self.assertEqual(instance.tracks, {})

    def test_a_gap_in_the_clock_propagates_the_missed_ticks(self):
        instance = tracker()
        seeded(instance, (CENTRE, CENTRE), 0.)
        instance.last_time = 0.
        # One second of chasing is ten engine ticks, not one.
        instance.update({1: MapGroup(1)}, {7: pose(7, (CENTRE + 200., CENTRE))}, [state(7)], 1.)
        track = next(iter(instance.tracks.values()))
        self.assertGreater(float(track.mean[0]), CENTRE + 9 * PREDATOR_SPRINT)

    def test_disabled_is_inert(self):
        instance = tracker(enabled=False)
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(30.)])], 0.)
        self.assertEqual(instance.tracks, {})
        self.assertEqual(instance.threats, {})

    def test_snapshot_is_json_compatible(self):
        instance = tracker()
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(160., 0., rel_dir=math.pi)])], 0.)
        payload = json.loads(json.dumps(instance.snapshot(1)))
        self.assertEqual(len(payload["tracks"]), 1)
        self.assertIsNotNone(payload["tracks"][0]["heading"])
        self.assertEqual(json.loads(json.dumps(instance.snapshot(2)))["tracks"], [])


class ThreatTests(unittest.TestCase):
    def test_threat_reports_the_escape_bearing_in_the_agent_facing_frame(self):
        instance = tracker(particles=32, threat_radius=120.)
        # Predator dead ahead of the agent, so escape points behind it.
        instance.update({1: MapGroup(1)}, {7: pose(7, (100., 100.), heading=.6)},
                        [state(7, [sighting(40., 0., rel_dir=math.pi)])], 0.)
        threat = instance.threats[7]
        self.assertAlmostEqual(threat.angle, 0., places=2)
        self.assertAlmostEqual(abs(threat.escape_direction), math.pi, places=2)
        self.assertGreater(threat.probability, .9)
        self.assertAlmostEqual(threat.distance, 40., places=1)
        self.assertEqual(threat.seconds_unseen, 0.)

    def test_a_distant_belief_raises_no_threat(self):
        instance = tracker(particles=32, threat_radius=50.)
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(200., 0., rel_dir=math.pi)])], 0.)
        self.assertEqual(instance.threats, {})
        self.assertEqual(len(instance.tracks), 1)

    def test_danger_field_peaks_on_the_belief(self):
        instance = tracker(particles=32, threat_radius=40.)
        instance.update({1: MapGroup(1)}, {7: pose(7, (0., 0.))},
                        [state(7, [sighting(100., 0., rel_dir=math.pi)])], 0.)
        near, far = instance.danger(1, [[100., 0.], [400., 0.]])
        self.assertGreater(near, .9)
        self.assertLess(far, 1e-9)
        self.assertEqual(float(instance.danger(2, [[100., 0.]])[0]), 0.)


class PolicyIntegrationTests(unittest.TestCase):
    """The belief must reach the flee branch without changing the defaults."""

    def policy(self, memory=None, belief=None, enabled=True):
        data = load_config(FIXTURES / "expert_policy.json").model_dump()
        data["memory"].update(memory or {})
        settings = PredatorBeliefConfig(enabled=enabled, particles=48).model_dump()
        settings.update(belief or {})
        return ExpertPolicy(ExpertConfig.model_validate(data),
                            planner_config(predator_belief=settings))

    def run_chase(self, policy, ticks, start=(CENTRE, CENTRE), agent=(CENTRE + 60., CENTRE)):
        """Drive a real Predator at a stationary agent and feed back sightings.

        The agent is only told about the predator when it is genuinely within
        its own hearing radius or vision cone, so later ticks exercise nothing
        but the belief.
        """
        creature = make_predator(*start, direction=0.)
        stand = Stand(*agent, direction=0.)
        actions = []
        for tick in range(ticks):
            offset = np.array([creature.x - stand.x, creature.y - stand.y])
            distance = float(np.hypot(*offset))
            bearing = wrap(math.atan2(offset[1], offset[0]) - stand.direction)
            visible = distance <= 50. or (distance <= 200. and abs(bearing) <= math.pi / 6)
            observations = [fruit(2.)] if not visible else [sighting(
                distance, bearing,
                rel_dir=wrap(math.atan2(-offset[1], -offset[0]) - creature.direction))]
            actions.append(policy.actions_for_step(
                [agent_state(observations, agent_id=7, energy=400., age=5. + tick * .1)],
                sim_time=tick * .1)[0])
            step_engine(creature, [stand])
        return creature, actions

    def test_running_the_belief_alone_changes_no_action(self):
        believing, blind = self.policy(), self.policy(enabled=False)
        self.assertFalse(believing.config.memory.belief_escape_enabled)
        self.assertEqual(believing.config.memory.belief_flee_seconds, 0.)
        _, with_belief = self.run_chase(believing, 30)
        _, without = self.run_chase(blind, 30)
        self.assertEqual([action.model_dump() for action in with_belief],
                         [action.model_dump() for action in without])
        # The tracker really did run; it simply had no authority.
        self.assertTrue(believing.planner.predator_tracker.threats
                        or believing.planner.predator_tracker.tracks)
        self.assertEqual(blind.planner.predator_tracker.tracks, {})

    def test_belief_escape_steers_by_a_bearing_that_keeps_moving(self):
        policy = self.policy(memory=dict(belief_escape_enabled=True))
        seen = []
        original = ExpertPolicy.action_decision

        def record(self, agent_state_, **kwargs):
            seen.append(kwargs.get("predator_threat"))
            return original(self, agent_state_, **kwargs)

        with patch.object(ExpertPolicy, "action_decision", record):
            self.run_chase(policy, 25)
        threats = [threat for threat in seen if threat is not None]
        self.assertTrue(threats, "the belief never produced a threat for the policy")
        # A frozen sighting bearing cannot move; a propagated one must.
        bearings = [threat.escape_direction for threat in threats]
        self.assertGreater(max(bearings) - min(bearings), 1e-6)
        self.assertTrue(any(threat.seconds_unseen > 0 for threat in threats),
                        "no threat outlived its sighting")

    def test_belief_flee_seconds_extends_danger_past_the_sighting_timeout(self):
        memory = dict(predator_escape_seconds=.2, belief_escape_enabled=True,
                      belief_flee_seconds=4., belief_flee_probability=.05)
        # A wide threat radius keeps this test about the timeout. At the
        # default radius the fleeing agent simply outruns the belief, which is
        # the system working rather than the wiring being exercised.
        believing = self.policy(memory=memory, belief=dict(threat_radius=600.))
        blind = self.policy(memory=memory, enabled=False)
        # One sighting inside the reactive radius, then silence. At 80 units it
        # is outside hearing, so no negative evidence contradicts the belief.
        moves = {}
        for name, policy in (("belief", believing), ("blind", blind)):
            policy.actions_for_step([agent_state(
                [sighting(80., 0., rel_dir=math.pi)], agent_id=7, energy=400., age=5.)],
                sim_time=0.)
            moves[name] = [policy.actions_for_step([agent_state(
                [fruit(2.)], agent_id=7, energy=400., age=5. + tick * .1)],
                sim_time=tick * .1)[0].move_distance for tick in range(1, 9)]
        # Fleeing sprints; taking the fruit instead moves only two units. The
        # sighting-only policy gives up after 0.2 seconds.
        self.assertEqual(moves["blind"][2:], [2.] * 6, moves["blind"])
        self.assertTrue(all(move > 2. for move in moves["belief"]), moves["belief"])

    def test_a_far_sighting_can_open_a_danger_state_the_reactive_gate_drops(self):
        memory = dict(predator_escape_seconds=2., belief_escape_enabled=True,
                      belief_flee_seconds=3., belief_flee_probability=.05)
        believing = self.policy(memory=memory, belief=dict(threat_radius=600.))
        blind = self.policy(memory=memory, enabled=False)
        radius = believing.config.perception.predator_danger_radius
        distance = radius + 60.
        self.assertGreater(distance, radius)
        moves = {}
        for name, policy in (("belief", believing), ("blind", blind)):
            first = policy.actions_for_step([agent_state(
                [sighting(distance, 0., rel_dir=math.pi)], agent_id=7, energy=400.)],
                sim_time=0.)[0]
            moves[name] = [first.move_distance] + [policy.actions_for_step([agent_state(
                [fruit(2.)], agent_id=7, energy=400., age=5. + tick * .1)],
                sim_time=tick * .1)[0].move_distance for tick in range(1, 4)]
        # prepare_inputs drops the sighting outright, so the reactive policy
        # forages and explores straight through it. Only the belief flees, and
        # fleeing is the one branch that spends the full sprint speed.
        sprint = 20.
        self.assertEqual(moves["belief"], [sprint] * 4, moves["belief"])
        self.assertTrue(all(move < sprint for move in moves["blind"]), moves["blind"])
        self.assertEqual(len(believing.planner.predator_tracker.tracks), 1)

    def test_planner_snapshot_carries_the_belief(self):
        policy = self.policy()
        policy.actions_for_step([agent_state(
            [sighting(40., 0., rel_dir=math.pi)], agent_id=7, energy=400.)], sim_time=0.)
        groups = policy.planner.snapshot()["groups"]
        self.assertTrue(groups)
        self.assertTrue(all("predator_belief" in group for group in groups))


if __name__ == "__main__":
    unittest.main()

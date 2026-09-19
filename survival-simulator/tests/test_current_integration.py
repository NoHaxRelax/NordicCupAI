"""Public-observation regressions for branch integration; no native game needed."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from models.core import EntrapmentPolicy, Track
from models.entrapment.bystander_avoidance import avoid_predators, DEFAULT_CONFIG
from models.entrapment import my_guide
from models.entrapment.guide_pathfinding import navigation_plan
from models.exploration.world_estimator import EstimatedPose
from models.experiment_config import defaults
from models.experimental_policy import ExperimentalEntrapment
from src.utils.DTOs import ActionRequest

EDGES = [((-400., -400.), (400., -400.)), ((400., -400.), (400., 400.)),
         ((400., 400.), (-400., 400.)), ((-400., 400.), (-400., -400.))]


class CurrentIntegrationTests(unittest.TestCase):
    def test_displaced_guide_rejoins_predator_width_lane(self):
        walls = EDGES + [((-6., -300.), (-6., 300.))]
        plan = navigation_plan((100., 0.), walls, (200., 0.), {})
        self.assertIsNotNone(plan)
        self.assertEqual(plan['mode'], 'rejoin_predator_lane')
        self.assertGreater(plan['dist_to_point'], 0.)

    def test_corner_pockets_extend_regular_sites_only_when_enabled(self):
        regular, corner = dict(site_kind='gap'), dict(site_kind='corner_pocket')
        def sites(_static, *, corner_only=False):
            return [corner] if corner_only else [regular]
        with patch('models.core.our_sites', side_effect=sites):
            self.assertEqual(EntrapmentPolicy()._candidate_sites({}), [regular])
            self.assertEqual(EntrapmentPolicy(include_corner_pockets=True)._candidate_sites({}), [regular, corner])

    def test_corner_fallback_and_occupied_corner_survive_new_regular_site(self):
        regular, corner = dict(site_kind='gap'), dict(site_kind='corner_pocket')
        policy = EntrapmentPolicy()
        with patch('models.core.our_sites', side_effect=lambda _, corner_only=False: [corner] if corner_only else []):
            self.assertEqual(policy._candidate_sites({}), [corner])
        policy.site = corner
        with patch('models.core.our_sites', side_effect=lambda _, corner_only=False: [corner] if corner_only else [regular]):
            self.assertEqual(policy._candidate_sites({}), [regular, corner])

    def test_shared_predators_require_fresh_same_frame_observations(self):
        policy = EntrapmentPolicy()
        policy.now = 2.
        policy.estimator.poses[1] = EstimatedPose(1, 0, np.array((0., 0.)))
        policy.estimator.poses[2] = EstimatedPose(2, 0, np.array((100., 0.)))
        sighting = dict(type='Predator', distance=20., angle=math.pi, rel_dir=0.)
        policy.tracks[0] = Track(0, 0, np.array((80., 0.)), 2., observers={2:sighting})
        shared = policy._shared_predators(1)
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]['distance'], 80.)
        self.assertAlmostEqual(abs(shared[0]['rel_dir']), math.pi)
        self.assertEqual(policy._shared_predators(2), [])
        policy.now = 2.2
        self.assertEqual(policy._shared_predators(1), [])
        policy.now = 2.
        policy.tracks[0].group = 1
        self.assertEqual(policy._shared_predators(1), [])

    def test_fresh_held_predator_releases_guide_after_ten_seconds(self):
        policy = EntrapmentPolicy()
        policy.now, policy.site_group, policy.bait = 10., 0, 1
        policy.site = dict(goal=(0., 0.))
        policy.estimator.poses[1] = EstimatedPose(1, 0, np.array((0., 0.)))
        policy.estimator.poses[2] = EstimatedPose(2, 0, np.array((70., 0.)))
        track = Track(0, 0, np.array((20., 0.)), 9.9, guide_id=2, held_since=0.)
        policy.tracks[0] = track
        policy._track_predators({1:dict(observations=[dict(type='Predator', distance=20., angle=0.)]),
                                 2:dict(observations=[])})
        self.assertIsNone(track.guide_id)
        self.assertEqual(policy.metrics['guide_releases'], 1)
        self.assertNotIn(2, policy.roles)

    def test_all_additions_off_preserves_current_predator_avoidance(self):
        state = dict(agent_id=1, observations=[dict(type='Predator', distance=80., angle=0., rel_dir=0.)],
                     speed=10., sprint_speed=20., biome='grassland', energy=150., max_energy=500.,
                     hearing_radius=50., vision_range=200., vision_angle=math.pi/3, age=0.)
        baseline = EntrapmentPolicy(seed=0)
        configurable = ExperimentalEntrapment(defaults(), seed=0)
        expected = [(aid, a.model_dump()) for aid, a in baseline([copy.deepcopy(state)], 0.)]
        actual = [(aid, a.model_dump()) for aid, a in configurable([copy.deepcopy(state)], 0.)]
        self.assertEqual(actual, expected)

    def test_coordinator_enables_guide_motion_association(self):
        policy = EntrapmentPolicy()
        policy.now = 1.
        policy.site = dict(goal=(100., 0.), handoff=(70., 0.), mouth=(95., 0.))
        policy.estimator.poses[1] = EstimatedPose(1, 0, np.array((0., 0.)))
        observation = dict(type='Predator', distance=100., angle=math.pi)
        track = SimpleNamespace(guide_id=1, observers={1: observation}, edges=EDGES, memory={}, completed_at=None)
        with patch('models.core.guide', return_value=dict(move_distance=0., move_direction=0., turn_angle=0.)) as guide:
            policy._guide_action(track, dict(observations=[observation]))
        context = guide.call_args.args[3]
        self.assertTrue(context['associate_target'])
        self.assertIs(context['target_predator'], observation)

    def test_held_crowd_hint_cannot_replace_tracked_newcomer(self):
        first = dict(type='Predator', distance=100., angle=math.pi)
        held = dict(type='Predator', distance=100., angle=0.)
        memory = {}
        my_guide._select_target((100., 0.), EDGES, dict(observations=[held, first]),
                                dict(tick=0, target_predator=first), memory)
        next_sighting = dict(type='Predator', distance=90., angle=math.pi)
        with patch.object(my_guide, '_guide', return_value={}) as inner, \
             patch.object(my_guide, 'prioritize', return_value={}):
            my_guide.guide((100., 0.), EDGES, dict(observations=[held, next_sighting]),
                           dict(tick=1, handoff=(70., 0.), target_predator=held, associate_target=True), memory)
        self.assertIs(inner.call_args.args[3]['target_predator'], next_sighting)

    def test_measured_lucas_default_holds_at_55_with_confirmed_following(self):
        predator = dict(type='Predator', distance=100., angle=math.pi)
        state = dict(observations=[predator], speed=10., sprint_speed=20.,
                     biome='grassland', energy=500., max_energy=500.)
        memory = {'_following_debug': {}}
        with patch.object(my_guide, 'predator_is_not_following', return_value=False):
            my_guide._guide((55., 0.), EDGES, state,
                            dict(tick=0, handoff=(25., 0.), target_predator=predator), memory)
        self.assertEqual(memory['debug']['mode'], 'hold_at_delivery')

    def test_bystander_moves_away_from_trap_without_hidden_predator_state(self):
        state = dict(observations=[], speed=10., sprint_speed=20.,
                     biome='grassland', energy=50., max_energy=500.)
        action = ActionRequest(agent_id=1, move_distance=10., move_direction=0., turn_angle=0., spawn_agent=True)
        safe, avoiding = avoid_predators(action, state, bait=(80., 0.))
        self.assertTrue(avoiding)
        self.assertLess(safe.move_distance*math.cos(safe.move_direction), 0.)
        self.assertLessEqual(safe.move_distance, state['speed'])
        self.assertFalse(safe.spawn_agent)
        unchanged, avoiding = avoid_predators(action, state, (80., 0.), dict(DEFAULT_CONFIG, enabled=False))
        self.assertIs(unchanged, action)
        self.assertFalse(avoiding)


if __name__ == '__main__':
    unittest.main()

"""Contracts for the observation-only colony integration."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.entrapment_policy import EntrapmentPolicy, remaining_life
from models.observed_trap_sites import observed_rectangles, our_sites
from models.nikolaj.world_estimator import MapGroup, EstimatedPose, EdgeLandmark
from models.entrapment_sites import enumerate_sites
from models.my_guide import guide


def edges(rect):
    x, y, w, h = rect
    points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    return [EdgeLandmark(np.array(a, dtype=float), np.array(b, dtype=float), 0.)
            for a, b in zip(points, points[1:]+points[:1])]


class IntegrationTests(unittest.TestCase):


    def test_lifetime_is_conservative_under_earliest_native_senescence(self):
        for age in (0., 55., 59.9, 60., 90., 130.):
            for energy in (3., 75., 200., 500.):
                e, a, t = energy, age, 0.
                while e > 0:
                    a += .1; t += .1; e -= .1
                    if a > 60.: e -= .01*a
                self.assertLessEqual(remaining_life(energy, age), t)

    def test_incomplete_interior_obstacle_is_not_invented(self):
        group = MapGroup(0, anchored=True, world_size=(1000., 800.))
        group.edges = edges((200., 200., 100., 60.))[:3]
        self.assertEqual(observed_rectangles(group)['obstacles'], [])
        group.edges += edges((200., 200., 100., 60.))[3:]
        self.assertEqual(observed_rectangles(group)['obstacles'], [(200., 200., 100., 60.)])

    def test_unobserved_boundaries_are_not_supplied(self):
        group = MapGroup(0, anchored=True, world_size=(1000., 800.))
        group.edges = [EdgeLandmark(np.array((0., 30.)), np.array((1000., 30.)), 0.)]
        self.assertEqual(observed_rectangles(group)['obstacles'], [(0., 0., 1000., 30.)])
        group.world_size = None
        self.assertIsNone(observed_rectangles(group))


    def test_our_short_gap_detector_and_bait_depth_are_used(self):
        static = dict(width=1200., height=900., obstacles=[(450., 400., 80., 100.), (545., 400., 80., 100.)])
        sites = our_sites(static)
        self.assertTrue(sites)
        for s in sites:
            self.assertEqual(s['gap'], 15.)
            self.assertEqual(s['bait_depth'], 5.)
            self.assertTrue(s['second_access_clear'])
            self.assertLessEqual(math.dist(s['goal'], s['handoff']), 44.)
        self.assertEqual({tuple(s['goal']) for s in sites},
                         {tuple(s['goal']) for s in enumerate_sites(static, min_gap=10.1, min_overlap=10.3)})

    def test_replacement_overlaps_instead_of_evicting_current_bait(self):
        p = EntrapmentPolicy()
        p.site = dict(goal=[100., 100.], replacement_entry=[100., 150.])
        p.site_group = 0; p.bait = 1; p.incoming = 2; p.now = 20.
        p.estimator.groups[0] = MapGroup(0)
        p.estimator.poses = {aid: EstimatedPose(aid, 0, np.array((100., 100.))) for aid in (1, 2)}
        p.navigator = Mock()
        states = {aid: dict(energy=200., age=70.) for aid in (1, 2)}
        with patch.object(p, '_select_bait', return_value=None): p._bait_roles(states)
        self.assertEqual(p.bait, 2)
        self.assertIn(1, p.retired_baits)
        self.assertEqual(p.roles[1], 'retired_bait')
        self.assertEqual(p.metrics['overlapping_replacements'], 1)
        self.assertEqual(p._bait_action(1, states[1]).move_distance, 0.)

    def test_two_observers_do_not_assign_two_guides_to_same_sighting(self):
        p = EntrapmentPolicy()
        p.now = 1.; p.site = dict(goal=[0., 0.]); p.site_group = 0; p.bait = 0
        p.estimator.groups[0] = MapGroup(0)
        p.estimator.poses = {a: EstimatedPose(a, 0, np.array(pos, dtype=float))
                            for a, pos in ((0, (0, 0)), (1, (100, 0)), (2, (110, 0)))}
        p.orchard.minds = {a: SimpleNamespace(old=False) for a in (0, 1, 2)}
        states = {a: dict(energy=150., age=10., observations=[]) for a in (0, 1, 2)}
        for a, distance in ((1, 100.), (2, 90.)):
            states[a]['observations'] = [dict(type='Predator', distance=distance, angle=0., rel_dir=0.)]
        p.roles[0] = 'bait'
        p._track_predators(states)
        self.assertEqual(len(p.tracks), 1)
        self.assertEqual(p.metrics['guide_assignments'], 1)
        self.assertEqual(sum(role == 'guide' for role in p.roles.values()), 1)

    def test_guide_can_select_associated_visible_predator(self):
        target = dict(type='Predator', distance=40., angle=.4)
        other = dict(type='Predator', distance=20., angle=-.7)
        state = dict(observations=[other, target], energy=300., max_energy=500., biome='grassland', speed=10., sprint_speed=20.)
        geometry = [(tuple(e.start), tuple(e.end)) for e in edges((-400., -400., 800., 800.))]
        action = guide((30., 0.), geometry, state, dict(tick=0, handoff=(0., 0.), target_predator=target), {})
        self.assertEqual(action['turn_angle'], .4)


if __name__ == '__main__': unittest.main()

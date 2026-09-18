"""Priority conflicts for observation-only local guiding."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.guide_steering import prioritize

EDGES = [((-400,-400),(400,-400)), ((400,-400),(400,400)),
         ((400,400),(-400,400)), ((-400,400),(-400,-400))]


class SteeringTests(unittest.TestCase):
    def steer(self, distance, rel_dir, direction, length=20., edges=EDGES, energy=500.):
        predator = dict(type='Predator', distance=distance, angle=0., rel_dir=rel_dir)
        agent = dict(observations=[predator], speed=10., sprint_speed=20.,
                     biome='grassland', energy=energy, max_energy=500.)
        memory = {}
        result = prioritize(dict(move_distance=length, move_direction=direction, turn_angle=0.),
                            (300.,0.), edges, agent, memory)
        q = (result['move_distance'] * math.cos(result['move_direction']),
             result['move_distance'] * math.sin(result['move_direction']))
        return result, q, memory['debug']['steering']

    def test_survival_overrides_hearing_and_destination(self):
        action, q, debug = self.steer(25., math.pi, 0.)
        self.assertLess(q[0], -19.)
        self.assertGreaterEqual(math.dist(q, (25.,0.)), 44.9)

    def test_hearing_overrides_destination_when_behind_predator(self):
        action, q, debug = self.steer(54., math.pi, math.pi)
        self.assertTrue(debug['safe'])
        self.assertTrue(debug['detectable'])
        self.assertLessEqual(math.dist(q, (54.,0.)), 55.)

    def test_vision_allows_progress_outside_hearing(self):
        action, q, debug = self.steer(120., 0., math.pi)
        self.assertTrue(debug['safe'])
        self.assertTrue(debug['detectable'])
        self.assertAlmostEqual(q[0], -20.)

    def test_wall_blocks_vision(self):
        wall = [((60.,-300.),(60.,300.))]
        action, q, debug = self.steer(120., 0., math.pi, edges=EDGES+wall)
        self.assertFalse(debug['detectable'])
        self.assertGreater(q[0], 0.)  # Approach hearing rather than walk away.

    def test_low_energy_cannot_request_sprint(self):
        action, q, debug = self.steer(25., math.pi, 0., energy=50.)
        self.assertLessEqual(action['move_distance'], 10.)

    def test_gaze_accounts_for_translation(self):
        action, q, debug = self.steer(120., 0., math.pi/2)
        self.assertAlmostEqual(action['turn_angle'], math.atan2(-q[1],120.-q[0]))

    def test_no_observation_preserves_recovery_action(self):
        action = dict(move_distance=10., move_direction=1., turn_angle=0.)
        self.assertEqual(prioritize(action, (300.,0.), EDGES,
                                   dict(observations=[]), {}), action)

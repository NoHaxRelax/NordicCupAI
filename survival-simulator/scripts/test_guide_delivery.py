"""Delivery holds must survive the outer safety-steering wrapper."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.entrapment.my_guide import guide

EDGES = [((-400,-400),(400,-400)), ((400,-400),(400,400)),
         ((400,400),(-400,400)), ((-400,400),(-400,-400))]


class DeliveryTests(unittest.TestCase):
    def act(self, handoff, memory, observed=True, bait=(30.,0.)):
        agent = dict(observations=[dict(type='Predator', distance=20., angle=.4)] if observed else [],
                     speed=10., sprint_speed=20., biome='grassland', energy=500., max_energy=500.)
        return guide(bait, EDGES, agent, dict(tick=0, dt=.1, handoff=handoff), memory)

    def test_close_predator_does_not_push_guide_off_delivery(self):
        memory={}
        action=self.act((.5,0.),memory)
        self.assertEqual(action['move_distance'],0.)
        self.assertEqual(action['turn_angle'],.4)
        self.assertEqual(memory['debug']['mode'],'hold_at_delivery')

    def test_before_arrival_safety_still_applies(self):
        memory={}
        self.assertGreater(self.act((20.,0.),memory,bait=(100.,0.))['move_distance'],0.)
        self.assertIn('steering',memory['debug'])

    def test_stops_at_hearing_radius_minus_five(self):
        memory={}
        self.assertEqual(self.act((25.,0.),memory,bait=(55.,0.))['move_distance'],0.)
        self.assertEqual(memory['debug']['mode'],'hold_at_delivery')

    def test_not_following_uses_recovery_even_at_delivery(self):
        memory={'_predator_not_following':True}
        self.act((0.,0.),memory)
        self.assertEqual(memory['debug']['mode'],'wait_for_predator')

    def test_lost_contact_exits_delivery_hold(self):
        memory={}
        self.act((0.,0.),memory)
        self.act((0.,0.),memory,observed=False)
        self.assertEqual(memory['debug']['mode'],'wait_for_predator')


if __name__=='__main__':
    unittest.main()

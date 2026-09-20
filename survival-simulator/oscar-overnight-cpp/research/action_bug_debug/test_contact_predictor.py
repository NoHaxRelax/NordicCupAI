"""Public-DTO regression fixtures. Actual game comparisons use the C++ engine."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'nightsim/serve'))
from harvest import Harvester


def frames(t, distance, heading=math.pi, ego=0., peer=False):
    rows=[]
    for aid in range(8):
        obs=[dict(type='Edge',coords=[[-100-ego,-100],[-100-ego,100]]),
             dict(type='Edge',coords=[[-100-ego,-100],[200-ego,-100]])]
        if aid==2:
            obs.append(dict(type='Predator',distance=distance,angle=0.,rel_dir=math.pi-heading))
            if peer: obs.append(dict(type='Agent',id=3,distance=50.,angle=0.,rel_dir=math.pi))
        rows.append(dict(agent_id=aid,energy=100.,max_energy=500.,speed=10.,sprint_speed=20.,
                         biome='grassland',age=t,observations=obs))
    return rows


def actions(step=0.):
    return [dict(agent_id=i,move_distance=step if i==2 else 0.,move_direction=0.,turn_angle=0.,spawn_agent=False)
            for i in range(8)]


class ContactTests(unittest.TestCase):
    def make(self):
        return Harvester(budget=2000,sacrifice_mode='predict_contact',cooldown=0.)

    def test_own_approach_to_sleeping_predator_does_not_fire(self):
        hv=self.make()
        hv.apply(frames(.1,40),.1,actions(10),score=0)
        hv.apply(frames(.2,30,ego=10),.2,actions(),score=.1)
        self.assertEqual(hv.harvests,0)
        self.assertEqual(hv.contact_predictor.stats['stationary'],1)

    def test_straight_active_contact_fires(self):
        hv=self.make()
        hv.apply(frames(.1,40),.1,actions(),score=0)
        out=hv.apply(frames(.2,25),.2,actions(),score=.1)
        self.assertEqual(hv.harvests,1)
        self.assertEqual(sum(a['agent_id']==2 for a in out),2000)
        self.assertLess(hv.log[-1]['predicted_contact'],14.)

    def test_turning_away_does_not_fire(self):
        hv=self.make()
        hv.apply(frames(.1,40,heading=math.pi/2),.1,actions(),score=0)
        hv.apply(frames(.2,25,heading=math.pi/2),.2,actions(),score=.1)
        self.assertEqual(hv.harvests,0)

    def test_slow_terrain_motion_does_not_assume_sprint(self):
        hv=self.make()
        hv.apply(frames(.1,32.5),.1,actions(),score=0)
        hv.apply(frames(.2,29.2),.2,actions(),score=.1)
        self.assertEqual(hv.harvests,0)

    def test_crossing_into_targets_river_uses_slower_bound(self):
        hv=self.make()
        old=frames(.1,40); new=frames(.2,29)
        old[2]['biome']=new[2]['biome']='river'
        hv.apply(old,.1,actions(),score=0)
        hv.apply(new,.2,actions(),score=.1)
        self.assertEqual(hv.harvests,0)

    def test_retains_wall_after_it_leaves_vision(self):
        hv=self.make()
        old=frames(.1,40)
        old[2]['observations'].append(dict(type='Edge',coords=[[15,-5],[15,5]]))
        hv.apply(old,.1,actions(),score=0)
        hv.apply(frames(.2,25),.2,actions(),score=.1)
        self.assertEqual(hv.harvests,0)
        self.assertEqual(len(hv.contact_predictor.previous[2]['walls']),3)

    def test_confirmed_transfer_marks_other_observers_and_reset_clears(self):
        hv=self.make()
        hv.apply(frames(.1,40,peer=True),.1,actions(),score=0)
        hv.apply(frames(.2,25,peer=True),.2,actions(),score=.1)
        self.assertEqual(hv.harvests,1)
        survivors=[s for s in frames(.3,0,peer=True) if s['agent_id'] not in (1,2)]
        hv.apply(survivors,.3,actions(),score=9.2)
        self.assertEqual(hv.contact_predictor.confirmed,1)
        self.assertTrue(hv.contact_predictor.frozen[3])
        hv.reset()
        self.assertIsNone(hv.contact_predictor)

    def test_missing_score_never_claims_confirmed_transfer(self):
        hv=self.make()
        hv.apply(frames(.1,40,peer=True),.1,actions())
        hv.apply(frames(.2,25,peer=True),.2,actions())
        hv.apply([s for s in frames(.3,0) if s['agent_id'] not in (1,2)],.3,actions())
        self.assertEqual(hv.contact_predictor.confirmed,0)

    def test_payout_ceiling_is_respected(self):
        hv=self.make()
        hv.apply(frames(.1,40),.1,actions(),score=0)
        out=hv.apply(frames(.2,25),.2,actions(),score=.1,payout_limit=2.)
        self.assertEqual(sum(a['agent_id']==2 for a in out),600)
        self.assertLessEqual(hv.log[-1]['expected_score'],2.)

    def test_invalid_payout_ceiling_does_not_consume_attempt(self):
        for limit in (0.,-1.,float('nan'),float('inf')):
            hv=self.make()
            hv.apply(frames(.1,40),.1,actions(),score=0)
            out=hv.apply(frames(.2,25),.2,actions(),score=.1,payout_limit=limit)
            self.assertEqual(hv.harvests,0)
            self.assertEqual(len(out),8)


if __name__=='__main__': unittest.main()

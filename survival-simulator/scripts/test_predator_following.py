"""Compare chase hypotheses with the native predator and test DTO timing."""
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.entrapment.guide_pathfinding import RoutePlanner
from models.entrapment.predator_following import possible_follow_moves, predator_is_not_following, wrap
from src.elements.predator import Predator
from src.elements.environment import Environment


def rectangle(x,y,w,h):
    return [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),
            ((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]


class FollowTests(unittest.TestCase):
    def test_predictions_include_native_motion_across_energy_terrain_and_collisions(self):
        for pivot in (False,True):
            for energy in (20.,200.):
                for modifier in (1.,.8,.5,.3):
                    for wall in (False,True):
                        with self.subTest(pivot=pivot,energy=energy,modifier=modifier,wall=wall):
                            p=(180.,130.);g=(320.,130.) if pivot else (260.,130.)
                            ph=0.;gh=math.pi+.25
                            sample=dict(predator=p,guide=g,predator_heading=ph,guide_heading=gh)
                            boxes=[SimpleNamespace(x=195.,y=100.,width=30.,height=80.)] if wall else []
                            edges=rectangle(0,0,400,300)
                            for box in boxes:edges+=rectangle(box.x,box.y,box.width,box.height)
                            geometry=RoutePlanner(edges,clearance=10.)
                            candidates=possible_follow_moves(sample,geometry)
                            predator=Predator(*p,rng=random.Random(1))
                            predator.direction=ph;predator.energy=energy
                            env=Environment.__new__(Environment)
                            env.width=400;env.height=300
                            env.biome_map=np.full((400,300),SimpleNamespace(move_penalty=modifier),dtype=object)
                            obs=dict(type='Agent',distance=math.dist(p,g),
                                     angle=wrap(math.atan2(g[1]-p[1],g[0]-p[0])-ph),
                                     rel_dir=wrap(math.atan2(p[1]-g[1],p[0]-g[0])-gh))
                            signals=predator.step([obs])
                            env.update_entity_position(predator,signals['move'],signals['direction'],boxes)
                            if 'turn' in signals:env.update_entity_direction(predator,signals['turn'])
                            self.assertTrue(any(math.dist(q,(predator.x,predator.y))<1e-6
                                                and abs(wrap(h-predator.direction))<1e-6
                                                for q,h,_ in candidates))

    def feed(self,memory,tick,p,g=(260.,130.),gh=math.pi,ph=0.,visible=True,others=()):
        def local(point):
            dx,dy=point[0]-g[0],point[1]-g[1]
            return dx*math.cos(gh)+dy*math.sin(gh),-dx*math.sin(gh)+dy*math.cos(gh)
        edges=[(local(a),local(b)) for a,b in rectangle(0,0,400,300)]
        q=local(p)
        pred=dict(type='Predator',distance=math.hypot(*q),angle=math.atan2(q[1],q[0]),
                  rel_dir=wrap(math.atan2(g[1]-p[1],g[0]-p[0])-ph)) if visible else None
        observations=[pred] if pred else []
        for other in others:
            q=local(other)
            observations.append(dict(type='Predator',distance=math.hypot(*q),angle=math.atan2(q[1],q[0]),
                                     rel_dir=wrap(math.atan2(g[1]-other[1],g[0]-other[0]))))
        return predator_is_not_following(pred,local((355.,245.)),edges,
                                          dict(observations=observations),dict(tick=tick),memory)

    def test_stale_dto_compares_against_previous_guide_pose(self):
        memory={}
        self.assertFalse(self.feed(memory,0,(180.,130.)))
        # Tick 1 repeats the fixture predator pose while the guide has moved.
        self.assertFalse(self.feed(memory,1,(180.,130.),g=(275.,130.)))
        # Both previous guide and predator face along the same pursuit axis;
        # allow the exact-zero watched-pivot native special case.
        self.assertFalse(self.feed(memory,2,(195.,130.),g=(290.,130.),gh=math.pi/2))
        self.assertEqual(memory['_following_debug']['consecutive_mismatches'],0)

    def test_two_impossible_moves_trigger_and_matching_move_clears(self):
        memory={}
        self.assertFalse(self.feed(memory,1,(165.,130.)))
        self.assertFalse(self.feed(memory,2,(180.,130.)))
        # Moving backward while retaining a heading directly toward us is not
        # a possible chase on any native terrain or energy cap in open space.
        self.assertFalse(self.feed(memory,3,(175.,130.)))
        self.assertTrue(self.feed(memory,4,(170.,130.)))
        self.assertTrue(self.feed(memory,5,(170.,130.)))  # Rest cannot clear True.
        self.assertFalse(self.feed(memory,6,(185.,130.)))

    def test_rest_and_reacquisition_are_not_false_negatives(self):
        memory={}
        for tick in range(1,5):self.assertFalse(self.feed(memory,tick,(180.,130.)))
        self.assertEqual(memory['_following_debug']['status'],'stationary_or_resting')
        self.assertFalse(self.feed(memory,5,(195.,130.)))
        self.assertFalse(self.feed(memory,6,(195.,130.)))  # Rest also retains False.
        self.assertFalse(self.feed(memory,7,(195.,130.),visible=False))
        self.assertFalse(self.feed(memory,8,(210.,140.)))
        self.assertEqual(memory['_following_debug']['status'],'need_consecutive_observations')

    def test_impossible_identity_jump_is_unknown(self):
        memory={}
        self.feed(memory,1,(100.,130.))
        self.assertFalse(self.feed(memory,2,(180.,130.)))
        self.assertEqual(memory['_following_debug']['status'],'possible_predator_switch')
        memory['_predator_not_following']=True
        self.assertTrue(self.feed(memory,3,(100.,130.)))

    def test_distant_crowd_does_not_hide_a_lost_or_following_target(self):
        memory={}; other=[(100.,250.)]
        self.assertFalse(self.feed(memory,1,(165.,130.),others=other))
        self.assertFalse(self.feed(memory,2,(180.,130.),others=other))
        self.assertNotEqual(memory['_following_debug']['status'],'ambiguous_identity_or_missing_heading')
        self.assertFalse(self.feed(memory,3,(175.,130.),others=other))
        self.assertTrue(self.feed(memory,4,(170.,130.),others=other))
        self.assertFalse(self.feed(memory,5,(185.,130.),others=other))

    def test_overlapping_predators_retain_the_latch(self):
        memory={'_predator_not_following':True}
        self.assertTrue(self.feed(memory,1,(180.,130.),others=[(185.,130.)]))
        self.assertTrue(self.feed(memory,2,(195.,130.),others=[(200.,130.)]))
        self.assertEqual(memory['_following_debug']['status'],'ambiguous_identity_or_missing_heading')


if __name__=='__main__':
    unittest.main()

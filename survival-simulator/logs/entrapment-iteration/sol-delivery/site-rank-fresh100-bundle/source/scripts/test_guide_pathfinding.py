"""Geometry and rotating-frame regressions for the user-editable guide."""
import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.entrapment.guide_pathfinding import RoutePlanner, navigation_plan
from models.entrapment.my_guide import guide, LOST_WAIT_TICKS


def rectangle(x,y,w,h):
    return [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),
            ((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]


class PathfindingTests(unittest.TestCase):
    def test_lost_predator_waits_returns_and_resumes_after_turning(self):
        memory={'_predator_not_following':False}  # Previously established pursuit.
        world_edges=rectangle(0,0,400,300)
        def call(position,heading,seen,tick):
            c,s=math.cos(heading),math.sin(heading)
            def local(p):
                x,y=p[0]-position[0],p[1]-position[1]
                return (x*c+y*s,-x*s+y*c)
            p=local((140,140))
            agent=dict(observations=([dict(type='Predator',distance=math.hypot(*p),angle=math.atan2(p[1],p[0]))] if seen else []),
                       speed=10.,sprint_speed=20.,biome='grassland')
            action=guide(local((365,160)),[(local(a),local(b)) for a,b in world_edges],
                         agent,dict(handoff=local((340,160)),tick=tick,dt=.1),memory)
            return action,p
        call((80,140),0.,True,0)
        for tick in range(1,LOST_WAIT_TICKS+1):
            action,_=call((90,150),math.pi/2,False,tick)
            self.assertEqual(action['move_distance'],0.)
            self.assertEqual(action['turn_angle'],0.)
            self.assertEqual(memory['debug']['mode'],'wait_for_predator')
        action,p=call((90,150),math.pi/2,False,LOST_WAIT_TICKS+1)
        self.assertEqual(action['turn_angle'],0.)
        self.assertGreater(action['move_distance'],0.)
        self.assertLessEqual(action['move_distance'],10.)
        self.assertAlmostEqual(action['move_direction'],math.atan2(p[1],p[0]))
        self.assertAlmostEqual(memory['debug']['last_predator'][0],p[0])
        self.assertAlmostEqual(memory['debug']['last_predator'][1],p[1])
        self.assertEqual(memory['debug']['mode'],'return_to_predator')
        action,p=call((90,150),math.pi/2,True,LOST_WAIT_TICKS+2)
        dx=action['move_distance']*math.cos(action['move_direction'])
        dy=action['move_distance']*math.sin(action['move_direction'])
        self.assertAlmostEqual(action['turn_angle'],math.atan2(p[1]-dy,p[0]-dx))
        self.assertEqual(memory['_lost_ticks'],0)
        self.assertEqual(memory['_reacquisitions'],1)
        self.assertNotIn('_recovery_navigation',memory)

    def test_arrived_at_missing_predator_holds_heading(self):
        edges=rectangle(0,0,400,300)
        bait=(365,160)
        from models.entrapment.guide_pathfinding import fixed_frame
        to_fixed,_=fixed_frame(bait,edges)
        memory=dict(_last_predator_fixed=to_fixed((5,0)),
                    _last_contact_position_fixed=to_fixed((0,0)),_lost_ticks=LOST_WAIT_TICKS)
        action=guide(bait,edges,dict(observations=[],speed=10.,sprint_speed=20.,biome='grassland'),
                     dict(handoff=(340,160)),memory)
        self.assertEqual(action['move_distance'],0.)
        self.assertEqual(action['turn_angle'],0.)
        self.assertEqual(memory['debug']['mode'],'wait_at_last_predator_position')

    def test_routes_around_obstacle_with_predator_clearance(self):
        planner=RoutePlanner(rectangle(0,0,200,200)+rectangle(80,60,40,80))
        start,goal=(30,100),(170,100)
        route=planner.plan(start,goal)
        self.assertGreater(len(route),1)
        self.assertEqual(route[-1],goal)
        for a,b in zip([start]+route,route):
            self.assertTrue(planner.clear(a,b))
        self.assertGreater(sum(math.dist(a,b) for a,b in zip([start]+route,route)),140)

    def test_agent_only_gap_is_rejected(self):
        edges=rectangle(0,0,200,200)+rectangle(90,0,20,91)+rectangle(90,109,20,91)
        self.assertEqual(RoutePlanner(edges).plan((40,100),(160,100)),[])

    def test_predator_width_gap_is_usable(self):
        edges=rectangle(0,0,200,200)+rectangle(90,0,20,87)+rectangle(90,113,20,87)
        self.assertEqual(RoutePlanner(edges).plan((40,100),(160,100)),[(160,100)])

    def test_plan_survives_rotation_translation_and_edge_reordering(self):
        world_edges=rectangle(0,0,200,200)+rectangle(80,60,40,80)
        bait=(180,145)
        target=(170,100)
        memory={}
        def at(position,heading):
            c,s=math.cos(heading),math.sin(heading)
            def local(p):
                x,y=p[0]-position[0],p[1]-position[1]
                return (x*c+y*s,-x*s+y*c)
            edges=[(local(a),local(b)) for a,b in world_edges]
            return navigation_plan(local(bait),edges[::-1],local(target),memory)
        first=at((30,100),0)
        planner=memory['_route_planner']
        rotated=at((30,100),1.1)
        self.assertIs(planner,memory['_route_planner'])
        self.assertAlmostEqual(first['dist'],rotated['dist'])
        self.assertAlmostEqual(math.atan2(math.sin(first['dir']-1.1),math.cos(first['dir']-1.1)),rotated['dir'])
        heading=first['dir']
        moved=at((30+3*math.cos(heading),100+3*math.sin(heading)),.7)
        self.assertIsNotNone(moved)
        self.assertLess(moved['dist'],first['dist'])
        self.assertEqual(moved['replans'],1)


if __name__=='__main__':
    unittest.main()

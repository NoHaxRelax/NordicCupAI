"""Small observation-only comparators; exactly one finite action per living agent.

No map, positions, hidden fruit ages or predator energies are available to policy.
"""
import math
import random
from src.utils.DTOs import ActionRequest
from src.utils.controllers.dummy_agent_policy import action_decision


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def edge_point(coords):
    (x,y),(u,v)=coords
    dx,dy=u-x,v-y
    t=max(0.,min(1.,-(x*dx+y*dy)/(dx*dx+dy*dy or 1)))
    return x+t*dx,y+t*dy


class SimplePolicy:
    def __init__(self, mode='nursery', seed=0):
        self.mode=mode
        self.rng=random.Random(seed)
        self.ticks=0

    def __call__(self, states, sim_time):
        self.ticks+=1
        if self.mode=='dummy':
            return [(s['agent_id'],action_decision(s,self.rng)) for s in states]
        scores={s['agent_id']:min(s['speed'],s['sprint_speed']) for s in states}
        best=max(scores.values(),default=10.)
        actions=[]
        slots=max(0,6-len(states))
        ordered=sorted(states,key=lambda s:(-scores[s['agent_id']],s['agent_id']))
        for s in ordered:
            obs=s['observations']; energy=s['energy']; aid=s['agent_id']
            fruits=sorted((o for o in obs if o['type']=='Fruit'),key=lambda o:o['distance'])
            predators=sorted((o for o in obs if o['type']=='Predator'),key=lambda o:o['distance'])
            trees=sorted((o for o in obs if o['type']=='Tree'),key=lambda o:o['distance'])
            edges=[edge_point(o['coords']) for o in obs if o['type']=='Edge']
            walk=min(s['speed'],s['sprint_speed'])
            direction=0.; distance=0.; turn=0.; danger=bool(predators and predators[0]['distance']<95)
            if danger:
                p=predators[0]
                direction=wrap(p['angle']+math.pi)
                distance=walk
                # Sprint only when a predator can reach us in a few steps.
                if p['distance']<55 and energy>s['max_energy']/5+5:
                    distance=min(s['sprint_speed'],max(walk,16.))
                turn=max(-.35,min(.35,p['angle'])) # keep predator visible while strafing away
            elif fruits and (self.mode=='greedy' or energy<s['max_energy']-55):
                f=fruits[0]
                direction=f['angle']; distance=min(walk,max(0.,f['distance']-8))
                turn=max(-.4,min(.4,direction))
            elif self.mode!='greedy' and trees and trees[0]['distance']<75 and energy>50:
                t=trees[0]
                direction=t['angle']; distance=min(walk,max(0.,t['distance']-35))
                turn=.12 if distance==0 else max(-.3,min(.3,direction))
            else:
                distance=walk
                turn=(.015 if aid%2 else -.015)*(1 if (self.ticks//200)%2 else -1)
            # Repel nearby visible walls; keeps rules local and simple.
            near=[(x,y) for x,y in edges if math.hypot(x,y)<30]
            if near and distance:
                vx,vy=math.cos(direction),math.sin(direction)
                for x,y in near:
                    d=max(1.,math.hypot(x,y))
                    if (vx*x+vy*y)>-3:
                        vx-=2*x/d; vy-=2*y/d
                direction=math.atan2(vy,vx)
                if not danger: turn=max(-.4,min(.4,direction))
            if self.mode=='greedy':
                spawn=energy>200 and not danger
            else:
                # Local food and a population ceiling limit demand; replace old agents sooner.
                viable_food=bool(fruits or trees)
                reserve=125 if s['age']>65 or len(states)<3 else 225
                breeder=(self.mode!='speed' or scores[aid]>=.95*best or len(states)<3)
                spawn=(slots>0 and viable_food and energy>reserve and breeder and not danger)
                if spawn: slots-=1
            actions.append((aid,ActionRequest(agent_id=aid,move_distance=distance,
                move_direction=direction,turn_angle=turn,spawn_agent=spawn)))
        return actions

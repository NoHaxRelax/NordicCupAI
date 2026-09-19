"""Reject survival destinations inside observed predator sensing or trap traffic.

All coordinates come from ordinary sightings and the shared observed map.
This filters destinations, not entire paths, and cannot protect against an
unseen predator. The final action-level avoidance remains necessary.
"""
import math
from collections import Counter

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from models.exploration.world_estimator import rotate


class ForagingSafety:
    def __init__(self):
        self.counts = Counter()
        self.predators = {}
        self.corridors = {}
        self.walls = {}
        self.cache = {}
        self.bait = None

    def update(self, policy, states):
        self.policy = policy
        self.predators = {}
        self.corridors = {}
        self.walls = {}
        self.cache = {}
        self.bait = None
        seen = set()
        for aid,state in states.items():
            pose = policy.estimator.poses.get(aid)
            if pose is None or pose.uncertainty > 8.: continue
            for obs in state['observations']:
                if obs['type'] != 'Predator': continue
                angle = pose.heading+obs['angle']
                point = (pose.position[0]+obs['distance']*math.cos(angle),
                         pose.position[1]+obs['distance']*math.sin(angle))
                heading = angle+math.pi-obs['rel_dir'] if 'rel_dir' in obs else None
                key = (pose.group_id,*point,heading)
                if key in seen: continue
                seen.add(key)
                self.predators.setdefault(pose.group_id,[]).append((point,heading))
        if policy.site is not None and any(aid in states and policy._arrival(aid)
                for aid in policy.retired_baits | {policy.bait}):
            self.bait = (policy.site_group,tuple(policy.site['goal']))
        for corridor in policy.guide_corridors:
            group = policy.estimator.groups.get(corridor['group'])
            if (group is None or corridor['guide'] not in states
                    or corridor.get('frame_revision') != group.frame_revision): continue
            if len(corridor['points']) >= 2:
                self.corridors.setdefault(corridor['group'],[]).append(LineString(corridor['points']))

    def _occluded(self, gid, start, end):
        if gid not in self.walls:
            group = self.policy.estimator.groups[gid]
            self.walls[gid] = unary_union([LineString([e.start,e.end]) for e in group.edges])
        return self.walls[gid].intersects(LineString([start,end]))

    def __call__(self, aid, point, kind):
        pose = self.policy.estimator.poses.get(aid)
        if pose is None or pose.uncertainty > 8.: return True
        gid = pose.group_id
        if (gid not in self.predators and gid not in self.corridors
                and (self.bait is None or self.bait[0] != gid)): return True
        key = (aid,tuple(point),kind)
        if key in self.cache: return self.cache[key]
        mind = self.policy.orchard.minds[aid]
        distance,angle = mind.pose.local(point)
        offset = rotate((distance*math.cos(angle),distance*math.sin(angle)),pose.heading)
        target = (pose.position[0]+offset[0],pose.position[1]+offset[1])
        reason = None
        # Match the action filter's existing thresholds, so it does not have
        # to fight an unchanged food/watch destination on every subsequent tick.
        if self.bait is not None and self.bait[0] == gid and math.dist(target,self.bait[1]) < 105.:
            reason = 'occupied_trap'
        if reason is None:
            for predator,heading in self.predators.get(gid,()):
                gap = math.dist(predator,target)
                if gap < 76.:
                    reason = 'hearing'; break
                if gap <= 265.:
                    bearing = math.atan2(target[1]-predator[1],target[0]-predator[0])
                    in_cone = heading is None or abs(math.atan2(math.sin(bearing-heading),math.cos(bearing-heading))) < math.radians(40)
                    if in_cone and not self._occluded(gid,predator,target):
                        reason = 'vision'; break
        if reason is None and any(path.distance(Point(target)) < 76. for path in self.corridors.get(gid,())):
            reason = 'guide_traffic'
        allowed = reason is None
        self.cache[key] = allowed
        if not allowed:
            self.counts[kind+':'+reason] += 1
        return allowed

    def snapshot(self):
        return dict(rejected_destination_checks=dict(self.counts),
                    note='Repeated candidate checks, not counts of saved agents; ordinary observed data only.')

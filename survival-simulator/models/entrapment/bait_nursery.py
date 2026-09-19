"""Small, optional food-gathering nursery behind an observed trap.

Uses observed trees and agent states only. Children enter the ordinary bait
selection pool; food, children and energy are never injected into the world.
"""
import math
import numpy as np
from src.utils.DTOs import ActionRequest
from models.exploration.world_estimator import rotate
from models.survival.oscar_orchard import MOVE_PENALTY


class BaitNursery:
    def __init__(self, size=0):
        if size not in range(5): raise ValueError('nursery size must be 0..4')
        self.size = size
        self.members = set()
        self.center = None
        self.frame = None
        self.last_birth = -math.inf
        self.spawn_parent = None

    def update(self, policy, states, active_life):
        self.spawn_parent = None
        if not self.size or policy.site is None:
            self.members.clear(); self.center = None; return
        group = policy.estimator.groups[policy.site_group]
        frame = (group.group_id, group.frame_revision)
        if self.frame != frame:
            self.members.clear(); self.center = None; self.frame = frame
        goal = np.asarray(policy.site['goal']); rear = np.asarray(policy.site['replacement_entry'])
        inward = np.asarray(policy.site['inward'])
        trees = [t.position for t in group.trees if policy.now-t.last_seen < 20.
                 and 120. < np.linalg.norm(t.position-goal) < 300.
                 and (t.position-goal) @ inward > 30.
                 and np.linalg.norm(t.position-rear) < 220.
                 and policy.navigator._clear(rear, t.position)]
        if self.center is None or not any(np.linalg.norm(p-self.center)<90. for p in trees):
            self.center = (min(trees, key=lambda p: np.linalg.norm(p-rear)
                              -30*sum(np.linalg.norm(p-q)<100. for q in trees)).copy() if trees else None)
        if self.center is None:
            self.members.clear(); return
        unavailable = policy.retired_baits | {policy.bait, policy.incoming} | {t.guide_id for t in policy.tracks.values()}
        eligible = [aid for aid, s in states.items() if aid not in unavailable and s['age'] < 55.
                    and policy.estimator.poses[aid].group_id == group.group_id
                    and policy.estimator.poses[aid].uncertainty <= 8.
                    and np.linalg.norm(policy.estimator.poses[aid].position-self.center) < 350.]
        self.members.intersection_update(eligible)
        candidates = [aid for aid in eligible if aid not in self.members and states[aid]['energy'] >= 175.]
        candidates.sort(key=lambda aid: (np.linalg.norm(policy.estimator.poses[aid].position-self.center),
                                         -states[aid]['energy']))
        self.members.update(candidates[:max(0,self.size-len(self.members))])
        for aid in self.members: policy.roles[aid] = 'nursery_farmer'
        children = sum(s['age'] < 15. and policy.estimator.poses[aid].group_id == group.group_id
                       and np.linalg.norm(policy.estimator.poses[aid].position-self.center)<160.
                       for aid,s in states.items())
        if policy.incoming is None and active_life < 75. and children < 2 and policy.now-self.last_birth >= 15.:
            rich = [aid for aid in self.members if states[aid]['energy'] >= max(300., .7*states[aid]['max_energy'])
                    and np.linalg.norm(policy.estimator.poses[aid].position-self.center) < 100.]
            if rich: self.spawn_parent = max(rich, key=lambda aid: states[aid]['energy'])

    def action(self, policy, aid, state):
        pose = policy.estimator.poses[aid]
        mind = policy.orchard.minds[aid]; group = policy.orchard.groups[mind.group]
        target = self.center
        # Reuse Orchard's ripe-fruit assignment, bounded to the nursery area.
        if mind.fruit in group.fruits:
            fruit = group.fruits[mind.fruit]
            distance, angle = mind.pose.local(fruit.p)
            point = pose.position+rotate((distance*math.cos(angle),distance*math.sin(angle)),pose.heading)
            if np.linalg.norm(point-self.center) <= 160.:
                target = point
            else:
                if fruit.claimed == aid: fruit.claimed = None
                mind.fruit = None
        moving = target is not self.center or np.linalg.norm(pose.position-target) > 30.
        move = direction = 0.; turn = .1
        if moving:
            plan = policy.navigator.steer(aid, pose.position, target, policy.now)
            if plan.waypoint is not None and not plan.blocked:
                offset = rotate(plan.waypoint-pose.position, -pose.heading)
                direction = math.atan2(offset[1],offset[0])
                move = min(state['speed'],float(np.linalg.norm(offset))/MOVE_PENALTY[state['biome']])
                turn = max(-.3,min(.3,direction))
        spawn = aid == self.spawn_parent
        if spawn:
            self.last_birth = policy.now
            policy.event('nursery_birth_requested', agent=aid, energy=state['energy'])
        return ActionRequest(agent_id=aid,move_distance=move,move_direction=direction,
                             turn_angle=turn,spawn_agent=spawn)

    def snapshot(self):
        return dict(members=sorted(self.members), center=None if self.center is None else self.center.tolist())

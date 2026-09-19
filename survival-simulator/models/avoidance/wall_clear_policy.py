"""Public-observation Orchard policy with the tuned wall-aware predator escape."""
from __future__ import annotations

import math

from models.survival.oscar_orchard import (
    OrchardPolicy, add, mul, point_segment, segments_cross, unit, wrap,
)


class WallClearOrchardPolicy(OrchardPolicy):
    def __init__(self, seed=0, *, pred_r=78.9691258405212,
                 pred_face_r=70.0, pred_sprint_r=26.447891933911748,
                 pred_dodge_r=78.9691258405212,
                 pred_dodge_ang=1.3806979798158516, pred_cone=0.5, **kw):
        super().__init__(seed=seed, **kw)
        self.pred = dict(r=pred_r, face_r=pred_face_r,
                         sprint_r=pred_sprint_r, dodge_r=pred_dodge_r,
                         dodge_ang=pred_dodge_ang, cone=pred_cone)

    def _steer_clear(self, m, direction, look=25.0):
        recent = [(a, b) for a, b, seen in m.edges
                  if self.time-seen < 25.0 and point_segment(m.pose.p, a, b) < look+10.0]
        if not recent:
            return direction

        def clear(heading):
            end = add(m.pose.p, mul(unit(heading), look))
            return all(not segments_cross(m.pose.p, end, a, b)
                       and point_segment(end, a, b) >= 7.0 for a, b in recent)

        heading = m.pose.theta + direction
        if clear(heading):
            return direction
        for k in range(1, 12):
            for sign in (1, -1):
                candidate = heading + sign*k*math.pi/12.0
                if clear(candidate):
                    return wrap(candidate-m.pose.theta)
        return direction

    def _act(self, m, state, states):
        threats = []
        for obs in state['observations']:
            if obs.get('type') != 'Predator':
                continue
            distance = float(obs['distance'])
            rel = float(obs.get('rel_dir', math.pi))
            if distance < self.pred['r'] or (abs(rel) < self.pred['cone'] and distance < self.pred['face_r']):
                threats.append((distance, float(obs['angle']), rel))
        if not threats:
            return super()._act(m, state, states)

        nearest = min(threats)
        vx = sum(-math.cos(angle)/max(distance, 15.0) for distance, angle, _ in threats)
        vy = sum(-math.sin(angle)/max(distance, 15.0) for distance, angle, _ in threats)
        away = math.atan2(vy, vx)
        distance, angle, rel = nearest
        if distance < self.pred['dodge_r']:
            away = wrap(angle+math.pi+(self.pred['dodge_ang'] if rel >= 0.0 else -self.pred['dodge_ang']))
        away = self._steer_clear(m, away)
        walk = min(state['speed'], state['sprint_speed'])
        step = state['sprint_speed'] if distance < self.pred['sprint_r'] else walk
        if m.fruit is not None:
            group = self.groups[m.group]
            if m.fruit in group.fruits:
                group.fruits[m.fruit].claimed = None
            m.fruit = None
        return step, away, angle, 'wall-clear-flee'

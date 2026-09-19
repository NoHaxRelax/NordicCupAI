"""Orchard adapter whose map bounds come exclusively from public Edge sightings.

The research policy assumes a 1600 by 1200 arena. This adapter disables those
bounds until one observation contains two perpendicular, full boundary edges.
Their lengths reveal the dimensions; their intersection's equal distance from
the nearest endpoints reveals the wall inset. Until then, maps stay relative.

The longer observed axis is called X. Edge endpoint ordering supplies its
orientation. These are geometric estimates, not queries to the simulator.
Each worker runs one policy instance; the inherited code uses module globals
for dimensions, which are reset from this instance before every decision.
"""
from __future__ import annotations

import math

from models.survival import oscar_orchard as orchard


class ObservedBoundsOrchard(orchard.OrchardPolicy):
    def __init__(self, seed=0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self._observed_bounds = None
        self._bounds_evidence = None
        self._set_dimensions()

    def _set_dimensions(self):
        if self._observed_bounds is None:
            orchard.W = orchard.H = math.inf
        else:
            orchard.W, orchard.H = self._observed_bounds[:2]

    @property
    def bounds_audit(self):
        bounds = self._observed_bounds
        return {
            "inferred": bounds is not None,
            "source": "public Edge observations only",
            "width": bounds[0] if bounds else None,
            "height": bounds[1] if bounds else None,
            "inset": bounds[2] if bounds else None,
            "evidence_count": 2 if bounds else 0,
            "evidence": self._bounds_evidence,
        }

    def __call__(self, states_list, sim_time):
        self._set_dimensions()
        return super().__call__(states_list, sim_time)

    def _infer_bounds(self, m, observations):
        if self._observed_bounds is not None:
            return
        edges = []
        for observation in observations:
            if observation["type"] != "Edge":
                continue
            start, end = map(tuple, observation["coords"])
            delta = orchard.sub(end, start)
            length = orchard.norm(delta)
            # The upstream policy uses this same conservative boundary detector.
            # A single segment, however long, never establishes the missing size.
            if length > 1000 and math.isfinite(length):
                edges.append((start, orchard.mul(delta, 1 / length), length))
        for index, first in enumerate(edges):
            for second in edges[index + 1:]:
                long, short = sorted((first, second), key=lambda edge: -edge[2])
                start, u, width = long
                other, v, height = short
                # Equal lengths cannot establish a common choice of the X axis.
                if math.isclose(width, height, rel_tol=1e-6):
                    continue
                if abs(u[0] * v[0] + u[1] * v[1]) > 1e-6:
                    continue
                cross = u[0] * v[1] - u[1] * v[0]
                offset = orchard.sub(other, start)
                along_u = (offset[0] * v[1] - offset[1] * v[0]) / cross
                along_v = (offset[0] * u[1] - offset[1] * u[0]) / cross
                inset_u = min(along_u, width - along_u)
                inset_v = min(along_v, height - along_v)
                if not (0 < inset_u < height * .1 and 0 < inset_v < height * .1):
                    continue
                if not math.isclose(inset_u, inset_v, rel_tol=1e-6, abs_tol=1e-5):
                    continue
                inset = (inset_u + inset_v) / 2
                short_sign = 1 if cross > 0 else -1
                self._observed_bounds = (width, height, inset, short_sign)
                self._bounds_evidence = {
                    "agent_id": m.aid,
                    "sim_time": self.time,
                    "edge_lengths": [width, height],
                    "endpoint_insets": [inset_u, inset_v],
                    "same_observation": True,
                }
                self._set_dimensions()
                return

    def _observe(self, m, state):
        self._set_dimensions()
        self._infer_bounds(m, state["observations"])
        return super()._observe(m, state)

    def _anchor(self, m, observation):
        if self._observed_bounds is None:
            return
        width, height, inset, short_sign = self._observed_bounds
        (x1, y1), (x2, y2) = observation["coords"]
        length = math.hypot(x2 - x1, y2 - y1)
        phi = math.atan2(y2 - y1, x2 - x1)
        if math.isclose(length, width, rel_tol=1e-6, abs_tol=1e-5):
            theta = orchard.wrap(-phi)
            candidates = [(0., inset), (0., height - inset), (0., 0.), (0., height)]
        elif math.isclose(length, height, rel_tol=1e-6, abs_tol=1e-5):
            theta = orchard.wrap(short_sign * math.pi / 2 - phi)
            start_y = 0. if short_sign > 0 else height
            candidates = [(inset, start_y), (width - inset, start_y),
                          (0., start_y), (width, start_y)]
        else:
            return
        position = None
        for start in candidates:
            candidate = orchard.sub(start, orchard.rot((x1, y1), theta))
            if 4 <= candidate[0] <= width - 4 and 4 <= candidate[1] <= height - 4:
                position = candidate
                break
        if position is None:
            return
        group = self.groups[m.group]
        if not group.anchored:
            old_pose = m.pose
            turn = orchard.wrap(theta - old_pose.theta)
            shift = orchard.sub(position, orchard.rot(old_pose.p, turn))
            self._transform_group(group, turn, shift)
            # _observe holds this object locally; preserve it through anchoring.
            old_pose.p, old_pose.theta = m.pose.p, m.pose.theta
            m.pose = old_pose
            group.anchored = True
            self.metrics["anchors"] += 1
        else:
            error = orchard.norm(orchard.sub(position, m.pose.p))
            if .5 < error < 40:
                m.pose.p = position
                self.metrics["pose_corrections"] += 1

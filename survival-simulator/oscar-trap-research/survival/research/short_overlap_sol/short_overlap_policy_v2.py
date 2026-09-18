"""Frozen observation-only depth-5 gap policy accepting overlap >=10.1."""
from __future__ import annotations

import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1] / "depth5_test"
sys.path.insert(0, str(BASE))
from observed_gap_policy import ObservedGapPolicy, add, sub, mul, dot, unit


class ShortOverlapPolicyV2(ObservedGapPolicy):
    """The established controller with only edge/overlap minima relaxed."""

    MIN_EDGE = 10.1
    MIN_OVERLAP = 10.1

    def _identify(self, group, states):
        live = sorted(group.members & states.keys())
        if not live:
            return None
        observer = live[0]
        point = self.poses[observer].p
        for i, first in enumerate(group.edges):
            length = math.dist(*first)
            if not self.MIN_EDGE <= length <= 100.1:
                continue
            tangent = unit(sub(first[1], first[0]))
            midpoint = mul(add(*first), .5)
            normal = (-tangent[1], tangent[0])
            for second in group.edges[i + 1:]:
                other_length = math.dist(*second)
                if not self.MIN_EDGE <= other_length <= 100.1:
                    continue
                other_tangent = unit(sub(second[1], second[0]))
                if abs(dot(tangent, other_tangent)) < .999:
                    continue
                other_midpoint = mul(add(*second), .5)
                delta = sub(other_midpoint, midpoint)
                gap = abs(dot(delta, normal))
                if not 10.9 <= gap <= 19.1:
                    continue
                first_projection = sorted(dot(p, tangent) for p in first)
                second_projection = sorted(dot(p, tangent) for p in second)
                low = max(first_projection[0], second_projection[0])
                high = min(first_projection[1], second_projection[1])
                overlap = high - low
                if overlap < self.MIN_OVERLAP:
                    continue
                cross = (dot(midpoint, normal) + dot(other_midpoint, normal)) / 2
                ends = [add(mul(tangent, low), mul(normal, cross)),
                        add(mul(tangent, high), mul(normal, cross))]
                mouth = min(ends, key=lambda p: math.dist(point, p))
                far = max(ends, key=lambda p: math.dist(point, p))
                inward = unit(sub(far, mouth))
                goal = add(mouth, mul(inward, self.bait_depth))
                return dict(gap=gap, length=overlap, mouth=mouth, goal=goal,
                            inward=inward, mapped=self.time)
        return None


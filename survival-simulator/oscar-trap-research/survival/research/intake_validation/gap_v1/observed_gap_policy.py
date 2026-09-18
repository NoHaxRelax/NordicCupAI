"""Observation-only controller for a size-selective obstacle passage.

The controller accepts only JSON-compatible native observation DTOs and public
simulation time.  It infers a 11--19 unit passage from observed obstacle edges,
walks to a point 30 units inside the nearest mouth, faces into the passage, and
then remains still.  It has no environment reference or fixture coordinates.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH / "wall_deployment"))
from controller import WallPolicy, add, sub, mul, dot, unit, norm, rot, wrap


class ObservedGapPolicy(WallPolicy):
    """Map a narrow parallel-face passage and hold every agent inside it."""

    def __init__(self):
        super().__init__()
        self.stations = {}

    def _identify(self, group, states):
        live = sorted(group.members & states.keys())
        if not live:
            return None
        observer = live[0]
        point = self.poses[observer].p
        for i, first in enumerate(group.edges):
            length = math.dist(*first)
            if not 54.9 <= length <= 100.1:
                continue
            tangent = unit(sub(first[1], first[0]))
            midpoint = mul(add(*first), .5)
            normal = (-tangent[1], tangent[0])
            for second in group.edges[i + 1:]:
                other_length = math.dist(*second)
                if abs(other_length - length) > .1:
                    continue
                other_tangent = unit(sub(second[1], second[0]))
                if abs(dot(tangent, other_tangent)) < .999:
                    continue
                other_midpoint = mul(add(*second), .5)
                delta = sub(other_midpoint, midpoint)
                gap = abs(dot(delta, normal))
                if not 10.9 <= gap <= 19.1 or abs(dot(delta, tangent)) > .1:
                    continue
                center = mul(add(midpoint, other_midpoint), .5)
                ends = [add(center, mul(tangent, length / 2)),
                        add(center, mul(tangent, -length / 2))]
                mouth = min(ends, key=lambda p: math.dist(point, p))
                far = max(ends, key=lambda p: math.dist(point, p))
                inward = unit(sub(far, mouth))
                goal = add(mouth, mul(inward, 30.))
                return dict(gap=gap, length=length, mouth=mouth, goal=goal,
                            inward=inward, mapped=self.time)
        return None

    def act(self, observations, sim_time):
        self.time = sim_time
        states = {state["agent_id"]: state for state in observations}
        if not self.poses and all(not state["observations"] for state in observations):
            return [dict(agent_id=aid, move_distance=0., move_direction=0.,
                         turn_angle=0., spawn_agent=False) for aid in sorted(states)]
        for aid in sorted(states.keys() & self.poses.keys()):
            self._map(aid, states[aid])
        self._register(states)
        for aid in sorted(states):
            self._map(aid, states[aid])

        actions = []
        self.decisions = {}
        handled = set()
        for group in self.groups:
            live = sorted(group.members & states.keys())
            if not live:
                continue
            key = id(group)
            station = self.stations.get(key)
            if station is None:
                station = self._identify(group, states)
                if station:
                    self.stations[key] = station
                    self._event("mapped_size_selective_gap", gap=round(station["gap"], 3),
                                length=round(station["length"], 3))
            for aid in live:
                handled.add(aid)
                state = states[aid]
                pose = self.poses[aid]
                target = pose.p if station is None else station["goal"]
                delta = sub(target, pose.p)
                modifier = dict(forest=1., grassland=1., swamp=.5,
                                desert=.8, river=.3)[state["biome"]]
                requested = min(state["speed"], norm(delta) / modifier)
                move_direction = (wrap(math.atan2(delta[1], delta[0]) - pose.theta)
                                  if norm(delta) > .01 else 0.)
                desired_heading = (math.atan2(station["inward"][1], station["inward"][0])
                                   if station else pose.theta + .35)
                turn = wrap(desired_heading - pose.theta)
                action = dict(agent_id=aid, move_distance=requested,
                              move_direction=move_direction, turn_angle=turn,
                              spawn_agent=False)
                actions.append(action)
                self.decisions[aid] = {"rule": ("enter_observed_size_selective_gap"
                                                 if station and norm(delta) > .01
                                                 else "hold_observed_gap_refuge"
                                                 if station else "scan_for_gap")}
                pose.p = add(pose.p, rot((requested * modifier, 0.),
                                         pose.theta + move_direction))
                pose.theta = wrap(pose.theta + turn)
        for aid in sorted(states.keys() - handled):
            actions.append(dict(agent_id=aid, move_distance=0., move_direction=0.,
                                turn_angle=.35, spawn_agent=False))
            self.decisions[aid] = {"rule": "scan_for_gap"}
        return sorted(actions, key=lambda action: action["agent_id"])

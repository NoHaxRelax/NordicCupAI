"""Observation-only narrow-gap policy with sacrificial late-arrival guides.

Agents first observed during initial setup enter and hold the mapped refuge.
Agents that first appear later map the same passage from their own observations
and wait 20 units outside its nearest mouth.  This role rule uses only native
DTOs and public time; no fixture coordinates or predator state enter actions.
"""
from __future__ import annotations

import math

from observed_gap_policy import ObservedGapPolicy
from controller import add, sub, mul, norm, rot, wrap


class ObservedGuidePolicy(ObservedGapPolicy):
    def __init__(self, setup_cutoff=3.):
        super().__init__()
        self.setup_cutoff = setup_cutoff
        self.first_seen = {}

    def act(self, observations, sim_time):
        self.time = sim_time
        states = {state["agent_id"]: state for state in observations}
        for aid in states:
            self.first_seen.setdefault(aid, sim_time)
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
                late = self.first_seen[aid] > self.setup_cutoff
                if station is None:
                    target = pose.p
                elif late:
                    target = add(station["mouth"], mul(station["inward"], -20.))
                else:
                    target = station["goal"]
                delta = sub(target, pose.p)
                modifier = dict(forest=1., grassland=1., swamp=.5,
                                desert=.8, river=.3)[state["biome"]]
                requested = min(state["speed"], norm(delta) / modifier)
                move_direction = (wrap(math.atan2(delta[1], delta[0]) - pose.theta)
                                  if norm(delta) > .01 else 0.)
                desired_heading = (math.atan2(station["inward"][1], station["inward"][0])
                                   if station else pose.theta + .35)
                turn = wrap(desired_heading - pose.theta)
                actions.append(dict(agent_id=aid, move_distance=requested,
                                    move_direction=move_direction, turn_angle=turn,
                                    spawn_agent=False))
                self.decisions[aid] = {"rule": ("stage_late_guide_outside_observed_gap"
                                                 if station and late
                                                 else "enter_observed_size_selective_gap"
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

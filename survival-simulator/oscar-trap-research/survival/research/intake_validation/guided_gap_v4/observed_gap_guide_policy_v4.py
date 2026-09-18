"""Observation-only aligned far-mouth staging at a predator-excluding gap.

Each disconnected supplied guide maps the overlapping inner faces itself. It
first centers 125 units outside the observed mouth. Once it senses a predator
behind itself, it runs a 50-unit axial lead-in and holds 75 units outside. Both
points remain beyond held-predator hearing. Fixture geometry is never supplied.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

BASE=Path(__file__).with_name("observed_gap_base_v2_snapshot.py")
spec=importlib.util.spec_from_file_location("observed_gap_policy_pinned",BASE)
base=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(base)


class ObservedGapGuidePolicy(base.ObservedGapPolicy):
    def __init__(self):
        super().__init__();self.phases={};self.first_seen={}

    def act(self,observations,sim_time):
        self.time=sim_time;states={s["agent_id"]:s for s in observations}
        for aid in states:self.first_seen.setdefault(aid,sim_time)
        if not self.poses and all(not s["observations"] for s in observations):
            return [dict(agent_id=aid,move_distance=0.,move_direction=0.,turn_angle=0.,spawn_agent=False) for aid in sorted(states)]
        for aid in sorted(states.keys() & self.poses.keys()):self._map(aid,states[aid])
        self._register(states)
        for aid in sorted(states):self._map(aid,states[aid])
        actions=[];self.decisions={};handled=set()
        for group in self.groups:
            live=sorted(group.members & states.keys())
            if not live:continue
            key=id(group);station=self.stations.get(key)
            if station is None:
                station=self._identify(group,states)
                if station:
                    self.stations[key]=station
                    self._event("mapped_size_selective_gap",gap=round(station["gap"],3),length=round(station["length"],3))
            for aid in live:
                handled.add(aid);state=states[aid];pose=self.poses[aid]
                target=pose.p;rule="scan_for_gap"
                if station:
                    late=self.first_seen[aid]>3.
                    if late:
                        far=base.sub(station["mouth"],base.mul(station["inward"],125.))
                        hold=base.sub(station["mouth"],base.mul(station["inward"],75.))
                        phase=self.phases.setdefault(aid,"center_far")
                        offset=base.sub(pose.p,far);cross=(-station["inward"][1],station["inward"][0])
                        centered=abs(base.dot(offset,cross))<=2.
                        if phase=="center_far" and (math.dist(pose.p,far)<4 or
                           (centered and base.dot(offset,station["inward"])>=0)):
                            phase=self.phases[aid]="wait_follower"
                        behind=[o for o in state["observations"] if o["type"]=="Predator" and abs(o["angle"])>math.pi/2]
                        if phase=="wait_follower" and behind and min(o["distance"] for o in behind)<=50:
                            phase=self.phases[aid]="axial_lead"
                            self._event("observed_follower_axial_lead",guide=aid)
                        if phase=="axial_lead" and (math.dist(pose.p,hold)<4 or
                           base.dot(base.sub(pose.p,hold),station["inward"])>=0):
                            phase=self.phases[aid]="hold_far"
                        target=far if phase in ("center_far","wait_follower") else hold
                        rule={"center_far":"center_125_outside_observed_mouth",
                              "wait_follower":"wait_for_observed_follower_behind",
                              "axial_lead":"lead_follower_along_observed_axis",
                              "hold_far":"hold_75_outside_observed_mouth"}[phase]
                    else:
                        target=station["goal"];rule="sprint_into_observed_gap"
                delta=base.sub(target,pose.p);modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
                requested=min(state["sprint_speed"],base.norm(delta)/modifier)
                direction=base.wrap(math.atan2(delta[1],delta[0])-pose.theta) if base.norm(delta)>.01 else 0.
                heading=math.atan2(station["inward"][1],station["inward"][0]) if station else pose.theta+.35
                turn=base.wrap(heading-pose.theta)
                action=dict(agent_id=aid,move_distance=requested,move_direction=direction,turn_angle=turn,spawn_agent=False)
                actions.append(action);self.decisions[aid]={"rule":rule}
                pose.p=base.add(pose.p,base.rot((requested*modifier,0.),pose.theta+direction));pose.theta=base.wrap(pose.theta+turn)
        for aid in sorted(states.keys()-handled):
            actions.append(dict(agent_id=aid,move_distance=0.,move_direction=0.,turn_angle=.35,spawn_agent=False));self.decisions[aid]={"rule":"scan_for_gap"}
        return sorted(actions,key=lambda a:a["agent_id"])

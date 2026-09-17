"""Observation-only front-side tangential sweep for occupied-wall intake.

Later guides reconstruct the nearest long wall face from native Edge
observations.  Each guide approaches one inboard end of that face at a 25-unit
normal clearance, sprints tangentially across it, then sacrifices itself at the
front face.  The route never crosses an end or enters the protected side.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1] / "wall_funneling" / "observed_policy_experimental.py"
_spec = importlib.util.spec_from_file_location("wall_funneling_staged_tangent_base", BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base)


class ObservedFunnel(_base.ObservedFunnel):
    def __init__(self, capacity=33, gate=True, replenish=False, gather=False,
                 line_clearance=25., sweep_halfspan=36.):
        super().__init__(capacity=capacity, gate=gate, replenish=replenish, gather=gather)
        self.line_clearance = line_clearance
        self.sweep_halfspan = sweep_halfspan

    def act(self, observations, sim_time):
        # Prevent the inherited straight-in rule for disconnected supplied
        # guides.  Holders and the initial crew retain the validated behavior.
        filtered=[]
        for state in observations:
            aid=state["agent_id"]
            group=self.group_for.get(aid)
            independent=bool(self.stations) and (group is None or id(group) not in self.stations)
            if independent:
                state=dict(state)
                state["observations"]=[o for o in state["observations"] if o["type"]!="Predator"]
            filtered.append(state)
        actions=super().act(filtered,sim_time)
        raw={s["agent_id"]:s for s in observations}

        for i,action in enumerate(actions):
            aid=action["agent_id"]
            if aid not in self.group_for:continue
            group=self.group_for[aid]
            if id(group) in self.stations:continue
            pose=self.poses[aid]
            edges=[e for e in group.edges if 69.9<=math.dist(*e)<=100.1]
            if not edges:continue

            # Undo inherited scanning before applying this route.
            modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[raw[aid]["biome"]]
            theta=_base.wrap(pose.theta-action["turn_angle"])
            pose.p=_base.sub(pose.p,_base.rot((action["move_distance"]*modifier,0.),theta+action["move_direction"]))
            pose.theta=theta

            edge=min(edges,key=lambda e:math.dist(pose.p,_base.mul(_base.add(*e),.5)))
            tangent=_base.unit(_base.sub(edge[1],edge[0]));mid=_base.mul(_base.add(*edge),.5)
            normal=(-tangent[1],tangent[0])
            if _base.dot(_base.sub(pose.p,mid),normal)<0:normal=_base.mul(normal,-1)
            # Give tangent a deterministic orientation in the guide's internal
            # frame; no world-axis value is supplied by the fixture.
            if tangent[1]<0 or (abs(tangent[1])<1e-6 and tangent[0]<0):tangent=_base.mul(tangent,-1)
            front=_base.add(mid,_base.mul(normal,5.1))
            line=_base.add(front,_base.mul(normal,self.line_clearance))
            worker=self.workers.setdefault(aid,{})
            phase=worker.setdefault("sweep_phase","approach_end")
            start=_base.add(line,_base.mul(tangent,-self.sweep_halfspan))
            finish=_base.add(line,_base.mul(tangent,self.sweep_halfspan))
            if phase=="approach_end" and math.dist(pose.p,start)<7:
                phase=worker["sweep_phase"]="cross_front"
                self._event("tangent_sweep_started",guide=aid)
            if phase=="cross_front" and math.dist(pose.p,finish)<7:
                phase=worker["sweep_phase"]="handoff"
                self._event("tangent_handoff_started",guide=aid)
            target=start if phase=="approach_end" else finish if phase=="cross_front" else _base.add(front,_base.mul(tangent,10.))
            threats=[o for o in raw[aid]["observations"] if o["type"]=="Predator"]
            turn=0.
            if threats:
                nearest=min(threats,key=lambda o:o["distance"])
                turn=_base.wrap(nearest["angle"]+math.pi)
            delta=_base.sub(target,pose.p)
            distance=min(raw[aid]["sprint_speed"],_base.norm(delta)/modifier)
            direction=_base.wrap(math.atan2(delta[1],delta[0])-pose.theta) if _base.norm(delta)>.01 else 0.
            actions[i]=dict(agent_id=aid,move_distance=distance,move_direction=direction,
                            turn_angle=turn,spawn_agent=False)
            self.decisions[aid]={"rule":"tangent_"+phase}
            pose.p=_base.add(pose.p,_base.rot((distance*modifier,0.),pose.theta+direction))
            pose.theta=_base.wrap(pose.theta+turn)
        return actions


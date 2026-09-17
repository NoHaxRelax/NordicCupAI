"""Observation-only rest-synchronised intake for an occupied wall funnel.

This controller extends the preserved staged-guide experiment.  Holders infer a
rest window only from repeated native predator observations.  A separately
supplied guide orbits outside the occupied trap's hearing zone until every
previously counted predator is stationary, then sprints to the observed front
face.  No predator IDs, energy, rest flags, target choices, fixture coordinates,
or evaluator state enter :meth:`act`.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1] / "wall_funneling" / "observed_policy_experimental.py"
_spec = importlib.util.spec_from_file_location("wall_funneling_staged_v10", BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base)


class ObservedFunnel(_base.ObservedFunnel):
    """Gate independent intake guides on holder-observed stationarity."""

    def __init__(self, capacity=33, gate=True, replenish=False, gather=False,
                 orbit_radius=34.0, still_delay=0.25, fresh_rest_limit=1.4):
        super().__init__(capacity=capacity, gate=gate, replenish=replenish, gather=gather)
        self.orbit_radius = orbit_radius
        self.still_delay = still_delay
        self.fresh_rest_limit = fresh_rest_limit
        self.rest_windows = {}

    def _safe_station(self, sim_time):
        return next((s for key, s in self.stations.items()
                     if self.rest_windows.get(key, {}).get("start") is not None
                     and self.still_delay <= sim_time-self.rest_windows[key]["start"] <= self.fresh_rest_limit), None)

    def _update_rest_windows(self, prior, sim_time):
        for key, station in self.stations.items():
            window = self.rest_windows.setdefault(key, {"start":None, "moving":None})
            old = prior.get(key, [])
            points = station.get("previous", [])
            complete = bool(points) and len(points) == len(old) == station.get("seen", 0)
            moving = False
            stationary = complete
            unmatched = list(old)
            for p in points:
                if not unmatched:
                    stationary = False
                    break
                q = min(unmatched, key=lambda q:math.dist(p, q))
                d = math.dist(p, q)
                moving |= d > 1e-5
                stationary &= d <= 1e-5
                unmatched.remove(q)
            if complete and moving:
                window["moving"] = sim_time
                window["start"] = None
            elif stationary and window["start"] is None and window["moving"] is not None \
                    and sim_time-window["moving"] <= .21:
                # A fresh window requires a directly observed move-to-still
                # transition.  Reappearance after occlusion is insufficient.
                window["start"] = sim_time
            elif not complete:
                window["start"] = None

    @staticmethod
    def _long_edge(group):
        edges = [e for e in group.edges if 69.9 <= math.dist(*e) <= 100.1]
        return edges[0] if edges else None

    def act(self, observations, sim_time):
        # Independent guides must not let the inherited controller commit before
        # an observed rest window.  Holder observations remain unchanged.
        filtered = []
        safe_before = self._safe_station(sim_time)
        prior = {key:list(station.get("previous", [])) for key,station in self.stations.items()}
        for state in observations:
            aid = state["agent_id"]
            group = self.group_for.get(aid)
            independent = group is None or id(group) not in self.stations
            committed = self.workers.get(aid, {}).get("front_commit", False)
            if self.stations and independent and not committed and safe_before is None:
                state = dict(state)
                state["observations"] = [o for o in state["observations"] if o["type"] != "Predator"]
            elif committed:
                # A committed handoff must finish inside the freshly observed
                # stationary window; the native sprint cap remains unchanged.
                state = dict(state)
                state["speed"] = state["sprint_speed"]
            filtered.append(state)

        actions = super().act(filtered, sim_time)
        self._update_rest_windows(prior, sim_time)
        original = {s["agent_id"]: s for s in observations}
        safe = self._safe_station(sim_time)

        for i, action in enumerate(actions):
            aid = action["agent_id"]
            if aid not in self.group_for:
                continue
            group = self.group_for[aid]
            if id(group) in self.stations:
                continue
            edge = self._long_edge(group)
            if edge is None:
                continue
            pose = self.poses[aid]
            worker = self.workers.setdefault(aid, {})
            if worker.get("front_commit"):
                continue

            # Undo the inherited scan action before replacing it.
            modifier = dict(forest=1., grassland=1., swamp=.5, desert=.8, river=.3)[original[aid]["biome"]]
            old_theta = _base.wrap(pose.theta - action["turn_angle"])
            travelled = action["move_distance"] * modifier
            pose.p = _base.sub(pose.p, _base.rot((travelled, 0.), old_theta + action["move_direction"]))
            pose.theta = old_theta

            tangent = _base.unit(_base.sub(edge[1], edge[0]))
            mid = _base.mul(_base.add(*edge), .5)
            normal = (-tangent[1], tangent[0])
            if _base.dot(_base.sub(pose.p, mid), normal) < 0:
                normal = _base.mul(normal, -1)
            front = _base.add(mid, _base.mul(normal, 5.1))
            threats = [o for o in original[aid]["observations"] if o["type"] == "Predator"]

            if safe is not None and threats:
                worker["front_commit"] = True
                target = front
                rule = "commit_during_holder_observed_rest"
                window = self.rest_windows.get(next((k for k,v in self.stations.items() if v is safe), None), {})
                self._event("intake_commit", guide=aid,
                            observed_still_for=round(sim_time-window.get("start", sim_time), 2))
            else:
                center = _base.add(front, _base.mul(normal, 118.0))
                angle = worker.setdefault("orbit_angle", math.atan2(pose.p[1]-center[1], pose.p[0]-center[0]))
                if math.dist(pose.p, center) < 2:
                    # Enter the orbit tangentially.  Moving radially outward
                    # would run directly into the follower behind the guide.
                    angle = math.atan2(normal[1], normal[0]) + math.pi/2
                    worker["orbit_angle"] = angle
                target = _base.add(center, (self.orbit_radius*math.cos(angle), self.orbit_radius*math.sin(angle)))
                if math.dist(pose.p, target) < 10:
                    angle += .32
                    worker["orbit_angle"] = angle
                    target = _base.add(center, (self.orbit_radius*math.cos(angle), self.orbit_radius*math.sin(angle)))
                rule = "orbit_until_holder_observed_rest"

            delta = _base.sub(target, pose.p)
            speed = original[aid]["sprint_speed"]
            distance = min(speed, _base.norm(delta)/modifier)
            direction = _base.wrap(math.atan2(delta[1], delta[0])-pose.theta) if _base.norm(delta) > .01 else 0.
            turn = 0.
            if threats:
                nearest = min(threats, key=lambda o:o["distance"])
                turn = _base.wrap(nearest["angle"] + math.pi)
            replacement = dict(agent_id=aid, move_distance=distance,
                               move_direction=direction, turn_angle=turn,
                               spawn_agent=False)
            actions[i] = replacement
            self.decisions[aid] = {"rule": rule}
            pose.p = _base.add(pose.p, _base.rot((distance*modifier, 0.), pose.theta+direction))
            pose.theta = _base.wrap(pose.theta+turn)
        return actions

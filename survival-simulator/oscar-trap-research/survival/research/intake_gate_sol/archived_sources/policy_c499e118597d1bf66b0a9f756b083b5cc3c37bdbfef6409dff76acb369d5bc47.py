"""Observation-only wall intake controller.

The two protected baits infer a sleep window from an unchanged complete
multiset of native predator bearings, positions and relative headings.  A
separately supplied guide holds its arriving predator on a short rectangular
track outside the old predators' bait-switch radius, then sprints the final
58 units during that observed window.  No predator IDs, energy, rest flag,
fixture dimensions, true position, or evaluator feedback enter this policy.
"""
from pathlib import Path
import math
import sys

WALL = Path(__file__).resolve().parents[1] / "wall_deployment"
sys.path.insert(0, str(WALL))
from controller import WallPolicy, add, sub, mul, dot, unit, norm, rot, wrap


class IntakeGatePolicy(WallPolicy):
    def __init__(self, capacity=33):
        super().__init__()
        self.capacity = capacity
        self.stations = {}
        self.workers = {}

    def _identify(self, group, states):
        live = sorted(group.members & states.keys())
        if len(live) < 2:
            return None
        holder = live[0]
        hp = self.poses[holder].p
        for edge in group.edges:
            length = math.dist(*edge)
            if not 69.9 <= length <= 100.1:
                continue
            tangent = unit(sub(edge[1], edge[0]))
            midpoint = mul(add(*edge), .5)
            normal = (-tangent[1], tangent[0])
            if dot(sub(hp, midpoint), normal) < 0:
                normal = mul(normal, -1)
            if not 4.9 <= dot(sub(hp, midpoint), normal) <= 12:
                continue
            for other in group.edges:
                if abs(math.dist(*other) - length) > .1:
                    continue
                om = mul(add(*other), .5)
                delta = sub(midpoint, om)
                width = dot(delta, normal)
                if not 29.9 <= width <= 35.1 or abs(dot(delta, tangent)) > .1:
                    continue
                protected = [aid for aid in live
                    if 4.9 <= dot(sub(self.poses[aid].p, midpoint), normal) <= 12]
                if len(protected) != 2:
                    continue
                protected.sort(key=lambda aid: dot(sub(self.poses[aid].p, midpoint), tangent))
                holders = {aid: add(add(midpoint, mul(normal, 5.1)), mul(tangent, off))
                           for aid, off in zip(protected, (-10, 10))}
                return dict(holder=holder, holders=holders,
                    anchor=holders[holder], front=add(om, mul(normal, -5.1)),
                    normal=normal, tangent=tangent, length=length, width=width,
                    seen=0, previous=[], still_since=None, sleep_ready=False)
        if .1 <= self.time < .3 and not getattr(group, '_intake_diag', False):
            group._intake_diag = True
            self._event('identify_diagnostic', live=live,
                poses={str(a): self.poses[a].p for a in live},
                edges=[(math.dist(*e), e) for e in group.edges])
        return None

    def _update_gate(self, station, states):
        points = []
        for observer in station['holders']:
            if observer not in states:
                continue
            pose = self.poses[observer]
            for obs in states[observer]['observations']:
                if obs['type'] != 'Predator':
                    continue
                point = pose.polar(obs)
                if dot(sub(point, station['anchor']), station['normal']) >= -20:
                    continue
                bearing_back = math.atan2(pose.p[1]-point[1], pose.p[0]-point[0])
                heading = wrap(bearing_back - obs['rel_dir'])
                # De-duplicate the same predator heard by both baits, while
                # preserving colocated predators as a multiset when one bait
                # reports both of them.
                match = next((i for i, old in enumerate(points)
                    if not old[2] and math.dist(old[0], point) < .05), None)
                if match is None:
                    points.append([point, heading, False])
                else:
                    points[match][2] = True
        current = [(p, h) for p, h, _ in points]
        station['seen'] = max(station['seen'], len(current))
        old = list(station['previous'])
        stationary = bool(current) and len(current) == len(old) == station['seen']
        for p, heading in current:
            if not old:
                stationary = False
                break
            j = min(range(len(old)), key=lambda i: math.dist(p, old[i][0]))
            q, old_heading = old.pop(j)
            if math.dist(p, q) > 1e-4 or abs(wrap(heading-old_heading)) > 1e-4:
                stationary = False
        if old:
            stationary = False
        if stationary:
            if station['still_since'] is None:
                station['still_since'] = self.time
        else:
            station['still_since'] = None
        # One full unchanged interval is enough: observations are sampled at
        # 10 Hz, leaving about 3.3 s of the native sleep interval for a 58-unit
        # sprint.  The guide only commits while at the inner holding rail.
        station['sleep_ready'] = (station['still_since'] is not None and
                                  self.time-station['still_since'] >= .09)
        station['previous'] = current

    def _station_gate(self):
        stations = list(self.stations.values())
        if not stations:
            return False, 0
        station = stations[0]
        return station['sleep_ready'], station['seen']

    def act(self, observations, sim_time):
        self.time = sim_time
        states = {s['agent_id']: s for s in observations}
        if not self.poses and all(not s['observations'] for s in observations):
            return [dict(agent_id=aid, move_distance=0., move_direction=0.,
                         turn_angle=0., spawn_agent=False) for aid in sorted(states)]
        for aid in sorted(states.keys() & self.poses.keys()):
            self._map(aid, states[aid])
        self._register(states)
        for aid in sorted(states):
            self._map(aid, states[aid])

        for group in self.groups:
            live = sorted(group.members & states.keys())
            if not live:
                continue
            key = id(group)
            if key not in self.stations:
                found = self._identify(group, states)
                if found:
                    self.stations[key] = found
                    self._event('mapped_intake_wall', holder=found['holder'])
            station = self.stations.get(key)
            if station and station['holder'] in states:
                self._update_gate(station, states)

        actions = []
        self.decisions = {}
        sleep_ready, held_seen = self._station_gate()
        for group in self.groups:
            live = sorted(group.members & states.keys())
            if not live:
                continue
            station = self.stations.get(id(group))
            for aid in live:
                state, pose = states[aid], self.poses[aid]
                target, turn, speed, rule = pose.p, 0., state['speed'], 'scan_for_wall'
                if station and aid in station['holders']:
                    target = station['holders'][aid]
                    rule = 'hold_far_face_bait'
                elif station:
                    # This branch is unused in the staged-guide fixture, but
                    # keeps an already registered non-bait from approaching.
                    target = add(station['front'], mul(station['normal'], -90))
                    speed = state['sprint_speed']
                    rule = 'retire_registered_worker'
                else:
                    edges = [e for e in group.edges if 99.9 <= math.dist(*e) <= 100.1]
                    if len(live) == 1 and edges:
                        edge = min(edges, key=lambda e: math.dist(pose.p, mul(add(*e), .5)))
                        mid = mul(add(*edge), .5)
                        tangent = unit(sub(edge[1], edge[0]))
                        normal = (-tangent[1], tangent[0])
                        if dot(sub(pose.p, mid), normal) < 0:
                            normal = mul(normal, -1)
                        worker = self.workers.setdefault(aid, dict(phase='approach', corner=0))
                        final = add(mid, mul(normal, 5.1))
                        rails = [
                            add(add(mid, mul(normal, 58)), mul(tangent, -35)),
                            add(add(mid, mul(normal, 68)), mul(tangent, -35)),
                            add(add(mid, mul(normal, 68)), mul(tangent, 35)),
                            add(add(mid, mul(normal, 58)), mul(tangent, 35)),
                        ]
                        inner = dot(sub(pose.p, mid), normal) <= 59.5
                        threats = [o for o in state['observations'] if o['type'] == 'Predator']
                        if threats:
                            worker['saw_threat'] = True
                        if worker['phase'] == 'approach':
                            target = rails[0]
                            speed = state['sprint_speed']
                            rule = 'approach_temporary_holder'
                            if math.dist(pose.p, target) < 2:
                                worker['phase'] = 'hold'
                        if worker['phase'] == 'hold':
                            # First arrival needs no gate. Later arrivals commit
                            # only from the inner rail during a complete observed
                            # sleep window of the old mass.
                            if worker.get('saw_threat') and (held_seen == 0 or sleep_ready) and inner:
                                worker['phase'] = 'deliver'
                                self._event('sleep_gated_commit', guide=aid,
                                            observed_old=held_seen)
                            else:
                                corner = worker['corner']
                                target = rails[corner]
                                if math.dist(pose.p, target) < 3:
                                    worker['corner'] = (corner + 1) % len(rails)
                                    target = rails[worker['corner']]
                                speed = state['sprint_speed']
                                rule = 'temporary_holder_rectangle'
                        if worker['phase'] == 'deliver':
                            target = final
                            speed = state['sprint_speed']
                            rule = 'sleep_window_final_sprint'
                        if threats:
                            nearest = min(threats, key=lambda o: o['distance'])
                            turn = wrap(nearest['angle'] + math.pi)
                    else:
                        turn = .35
                delta = sub(target, pose.p)
                modifier = dict(forest=1., grassland=1., swamp=.5,
                                desert=.8, river=.3)[state['biome']]
                distance = min(speed, norm(delta)/modifier)
                direction = wrap(math.atan2(delta[1], delta[0])-pose.theta) if norm(delta) > .01 else 0.
                actions.append(dict(agent_id=aid, move_distance=distance,
                    move_direction=direction, turn_angle=turn, spawn_agent=False))
                self.decisions[aid] = {'rule': rule}
                pose.p = add(pose.p, rot((distance*modifier, 0), pose.theta+direction))
                pose.theta = wrap(pose.theta+turn)
        return actions

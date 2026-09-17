"""Observation-only local wall/colony controller. No engine objects are accepted.

Coordinates are internal map estimates, initialized at (0, 0, heading=0) for
each disconnected founder. Native Edge endpoints correct walking odometry;
Agent bearings and rel_dir register nearby founders and newborns. Fruit tracks
are spatial matches, not hidden IDs; first sighting gives a ripeness lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from shapely.geometry import LineString, Point

Point2 = tuple[float, float]


def add(a, b): return (a[0]+b[0], a[1]+b[1])
def sub(a, b): return (a[0]-b[0], a[1]-b[1])
def mul(a, s): return (a[0]*s, a[1]*s)
def dot(a, b): return a[0]*b[0]+a[1]*b[1]
def norm(a): return math.hypot(*a)
def unit(a): return mul(a, 1/max(norm(a), 1e-12))
def rot(a, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (a[0]*c-a[1]*s, a[0]*s+a[1]*c)
def wrap(a): return (a+math.pi) % (2*math.pi)-math.pi


def point_segment(p, a, b):
    v = sub(b, a)
    t = max(0., min(1., dot(sub(p, a), v)/max(dot(v,v), 1e-12)))
    return math.dist(p, add(a, mul(v,t)))


def segment_distance(a, b, c, d):
    # Fast planar segment distance; avoids constructing geometry per graph edge.
    u, v, w = sub(b,a), sub(d,c), sub(c,a)
    cross = lambda x,y: x[0]*y[1]-x[1]*y[0]
    den = cross(u,v)
    if abs(den)>1e-10:
        t, s = cross(w,v)/den, cross(w,u)/den
        if 0 <= t <= 1 and 0 <= s <= 1: return 0.
    return min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))


@dataclass
class Pose:
    p: Point2 = (0., 0.)
    theta: float = 0.

    def transform(self, p): return add(self.p, rot(p, self.theta))
    def polar(self, observation):
        return self.transform(mul((math.cos(observation['angle']),
                                   math.sin(observation['angle'])), observation['distance']))


@dataclass
class Food:
    p: Point2
    first: float
    last: float
    kind: str


@dataclass
class Trap:
    anchor: Point2
    normal: Point2
    edge: tuple[Point2, Point2]
    holder: int
    created: float
    incoming: int | None = None
    last_swap: float = -100.
    last_predator: float = 0.
    stage: str = 'hold'


@dataclass
class Group:
    members: set[int] = field(default_factory=set)
    edges: list = field(default_factory=list)
    food: list[Food] = field(default_factory=list)
    trap: Trap | None = None
    retired: set[int] = field(default_factory=set)
    predator_tracks: list = field(default_factory=list)
    waypoints: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    wander: dict = field(default_factory=dict)
    fruit_claims: set = field(default_factory=set)
    wall_cooldown: float = 0.


class WallPolicy:
    def __init__(self, wall=True, ripeness_wait=15., retire_age=65., active_cap=3):
        self.wall = wall
        self.ripeness_wait = ripeness_wait
        self.retire_age = retire_age
        self.active_cap = active_cap
        self.poses: dict[int, Pose] = {}
        self.groups: list[Group] = []
        self.group_for = {}
        self.time = 0.
        self.events = []
        self.decisions = {}
        self.metrics = dict(edge_corrections=0, max_edge_correction=0.,
                            rest_attempts=0, recognition=0, handoffs=0,
                            emergency_flees=0, predicted_births=0, abandoned=0)

    def _event(self, kind, **data):
        self.events.append(dict(time=round(self.time, 2), kind=kind, **data))

    def _register(self, states):
        # Nearby entities can establish an exact relative frame. Iteration lets
        # a chain of currently observed agents join without global positions.
        unregistered = set(states)-set(self.poses)
        while unregistered:
            progress = False
            for aid in sorted(set(states) & set(self.poses)):
                pose, group = self.poses[aid], self.group_for[aid]
                for o in states[aid]['observations']:
                    if o['type'] == 'Agent' and o['id'] in unregistered:
                        other = o['id']
                        p = pose.polar(o)
                        theta = wrap(math.atan2(pose.p[1]-p[1], pose.p[0]-p[0])-o['rel_dir'])
                        self.poses[other] = Pose(p, theta)
                        self.group_for[other] = group
                        group.members.add(other)
                        unregistered.remove(other)
                        progress = True
            if not progress:
                aid = min(unregistered)
                self.poses[aid] = Pose()
                group = Group(members={aid})
                self.groups.append(group)
                self.group_for[aid] = group
                unregistered.remove(aid)

    def _map(self, aid, state):
        pose, group = self.poses[aid], self.group_for[aid]
        raw = [o['coords'] for o in state['observations'] if o['type'] == 'Edge']
        # Match the directed endpoint pair or its reversal. Length AND direction
        # constrain matches; minimum translation favors the local odometry prior.
        candidates = []
        for local in raw:
            a, b = map(pose.transform, local)
            for x, y in group.edges:
                for c, d in ((x, y), (y, x)):
                    shift = sub(c, a)
                    if norm(sub(add(b, shift), d)) < .02 and norm(shift) < 20.01:
                        candidates.append(shift)
        if candidates:
            correction = min(candidates, key=norm)
            pose.p = add(pose.p, correction)
            if norm(correction) > .02:
                self.metrics['edge_corrections'] += 1
                self.metrics['max_edge_correction'] = max(self.metrics['max_edge_correction'], norm(correction))
        for local in raw:
            edge = tuple(map(pose.transform, local))
            if not any((math.dist(edge[0], e[0]) < .05 and math.dist(edge[1], e[1]) < .05)
                       or (math.dist(edge[0], e[1]) < .05 and math.dist(edge[1], e[0]) < .05)
                       for e in group.edges):
                group.edges.append(edge)
        seen_food = []
        for o in state['observations']:
            if o['type'] not in ('Fruit', 'Tree'): continue
            p = pose.polar(o)
            match = next((f for f in group.food if f.kind == o['type'] and math.dist(f.p, p) < 1.5), None)
            if match is None:
                match = Food(p, self.time, self.time, o['type'])
                group.food.append(match)
            match.last = self.time
            seen_food.append(match)
        # Hearing ignores occlusion, so a missing nearby food really disappeared.
        group.food = [f for f in group.food if (f in seen_food or
                      math.dist(f.p, pose.p) > state['hearing_radius']-2 or
                      self.time-f.last < .05) and self.time-f.last < (12 if f.kind == 'Fruit' else 60)]

    def _predators(self, group, states):
        points = []
        for aid in sorted(group.members & states.keys()):
            for o in states[aid]['observations']:
                if o['type'] != 'Predator': continue
                p = self.poses[aid].polar(o)
                if any(math.dist(p, q) < 3 for q in points): continue
                points.append(p)
        tracks = []
        for p in points:
            old = next((t for t in group.predator_tracks if math.dist(t['p'], p) < 2), None)
            tracks.append(dict(p=p, still=old['still'] if old else self.time))
        group.predator_tracks = tracks
        return points

    def _blocked(self, group, start, end, clearance=6.):
        if math.dist(start, end) < .01: return False
        # Boundary of an observed obstacle is sufficient to forbid crossing.
        for a,b in group.edges:
            if (max(start[0],end[0])+clearance < min(a[0],b[0]) or
                min(start[0],end[0])-clearance > max(a[0],b[0]) or
                max(start[1],end[1])+clearance < min(a[1],b[1]) or
                min(start[1],end[1])-clearance > max(a[1],b[1])): continue
            distance = segment_distance(start,end,a,b)
            if distance < clearance-1e-4:
                initial = point_segment(start,a,b)
                # Native births can start inside our planning clearance. Permit
                # leaving that buffer without crossing the observed wall.
                if initial>.05 and initial<clearance and distance>=initial-.01 and point_segment(end,a,b)>initial+.01:
                    continue
                return True
        return False

    def _waypoint(self, group, aid, target):
        p = self.poses[aid].p
        if not self._blocked(group, p, target): return target
        last = group.targets.get(aid)
        if last and math.dist(last[0],target)<2 and self.time-last[1]<1:
            return p
        # Local visibility graph around observed edge endpoints. Deliberately
        # bounded to nearby geometry; map corrections resolve unseen collisions.
        edges = sorted((e for e in group.edges if point_segment(p,*e)<160),
                       key=lambda e:point_segment(p,*e))[:12]
        nodes = [p, target]
        for a, b in edges:
            u = unit(sub(b, a)); n = (-u[1], u[0])
            for c, sign in ((a, -1), (b, 1)):
                for side in (-1, 1):
                    q = add(c, add(mul(u, sign*8), mul(n, side*8)))
                    if all(LineString(e).distance(Point(q)) >= 6.1 for e in edges): nodes.append(q)
        # A short path cache saves rebuilding the graph each tick.
        previous = group.waypoints.get(aid)
        if previous and math.dist(previous[-1], target) < 2:
            while len(previous)>1 and math.dist(p, previous[0]) < 2: previous.pop(0)
            if not self._blocked(group, p, previous[0]): return previous[0]
        import heapq
        heap = [(0., 0, [])]; visited = set(); best = {0:0.}
        while heap:
            cost, i, path = heapq.heappop(heap)
            if i in visited: continue
            visited.add(i)
            if i == 1:
                group.waypoints[aid] = [nodes[j] for j in path]
                return group.waypoints[aid][0]
            for j in range(1, len(nodes)):
                if j in visited or j == i: continue
                cost_to = cost+math.dist(nodes[i], nodes[j])
                if cost_to >= best.get(j,math.inf): continue
                if not self._blocked(group, nodes[i], nodes[j]):
                    best[j] = cost_to
                    heapq.heappush(heap, (cost_to, j, path+[j]))
        group.targets[aid] = (target,self.time)
        return p

    def _recognize(self, group, states, predators):
        if not self.wall or group.trap is not None or self.time<group.wall_cooldown: return
        for aid in sorted(group.members & states.keys()):
            p = self.poses[aid].p
            for predator in predators:
                if math.dist(p, predator) > 58: continue
                for edge in group.edges:
                    a, b = edge; length = math.dist(a, b)
                    if not 69.9 <= length < 500: continue
                    u = unit(sub(b, a)); n = (-u[1], u[0])
                    if dot(sub(p, a), n) < 0: n = mul(n, -1)
                    across = dot(sub(predator, a), n) < -15
                    offset = dot(sub(p, a), u)
                    if across and 15 < offset < length-15 and 4 < dot(sub(p, a), n) < 14:
                        # Keep current along-wall coordinate, rather than pulling
                        # predator toward an unobserved corner at acquisition.
                        anchor = add(add(a, mul(u, offset)), mul(n, 7))
                        group.trap = Trap(anchor, n, edge, aid, self.time, last_predator=self.time)
                        self.metrics['recognition'] += 1
                        self._event('recognize_acquired', holder=aid, anchor=anchor)
                        return
        # Rest is hidden. >=0.3 s of stationary observations is a heuristic,
        # not proof. Require a known adjacent short edge and a short safe route.
        for tr in group.predator_tracks:
            if self.time-tr['still'] < .3: continue
            pred = tr['p']
            for a, b in group.edges:
                if not 69.9 <= math.dist(a, b) < 250: continue
                u = unit(sub(b, a)); n = (-u[1], u[0])
                if dot(sub(pred, a), n) < 0: n = mul(n, -1)
                if not 8 < dot(sub(pred, a), n) < 25: continue
                for c, d in group.edges:
                    for corner, far in ((c, d), (d, c)):
                        if min(math.dist(corner, a), math.dist(corner, b)) > .1: continue
                        short = sub(far, corner)
                        if not 14.9 <= norm(short) <= 35.1 or abs(dot(unit(short), u)) > .01: continue
                        if dot(short, n) >= 0: continue
                        midpoint = mul(add(a, b), .5)
                        anchor = add(midpoint, mul(n, -(norm(short)+7)))
                        for aid in sorted(group.members & states.keys()):
                            if states[aid]['energy'] < 110: continue
                            p = self.poses[aid].p
                            if math.dist(p, anchor) > 145: continue
                            waypoint = self._waypoint(group, aid, anchor)
                            route = group.waypoints.get(aid, [waypoint, anchor])
                            distance = sum(math.dist(x, y) for x, y in zip([p]+route, route))
                            if distance > 150 or math.dist(waypoint, p) < .1: continue
                            opposite_edge = (add(a, short), add(b, short))
                            # Two adjacent perpendicular full edges define the
                            # native rectangle. Retain the inferred far face so
                            # its equal-length segment cannot be mistaken for
                            # the near face during localization around a corner.
                            if not any(math.dist(opposite_edge[0],e[0])<.05 and
                                       math.dist(opposite_edge[1],e[1])<.05 for e in group.edges):
                                group.edges.append(opposite_edge)
                            group.trap = Trap(anchor, mul(n, -1), opposite_edge, aid,
                                              self.time, stage='acquire', last_predator=self.time)
                            self.metrics['rest_attempts'] += 1
                            self._event('rest_attempt', holder=aid, route_distance=round(distance, 2))
                            return

    def _roles(self, group, states):
        live = group.members & states.keys()
        trap = group.trap
        if trap is None: return
        if trap.holder not in live: trap.holder = None
        if trap.incoming not in live: trap.incoming = None
        if trap.stage == 'acquire' and trap.holder is not None:
            if math.dist(self.poses[trap.holder].p, trap.anchor) < 2:
                trap.stage = 'hold'
                self._event('arrived', holder=trap.holder)
            elif self.time-trap.created > 4:
                self._event('acquisition_timeout', holder=trap.holder)
                group.trap = None
                return
        if trap.incoming is not None and math.dist(self.poses[trap.incoming].p, trap.anchor) < 1.5:
            old = trap.holder
            trap.holder, trap.incoming = trap.incoming, None
            if old in live and states[old]['age'] >= self.retire_age: group.retired.add(old)
            trap.last_swap = self.time
            self.metrics['handoffs'] += 1
            self._event('handoff', outgoing=old, incoming=trap.holder)
        candidates = [aid for aid in live-group.retired if aid != trap.holder]
        need = trap.holder is None or (self.time-trap.last_swap > 6 and
               (states[trap.holder]['energy'] < 85 or states[trap.holder]['age'] >= self.retire_age))
        if need and trap.incoming is None and candidates:
            best = max(candidates, key=lambda aid: states[aid]['energy']-
                       8*max(0, states[aid]['age']-55)-.1*math.dist(self.poses[aid].p, trap.anchor))
            if trap.holder is None or states[best]['energy'] > states[trap.holder]['energy']+25 or (
                states[best]['age'] < self.retire_age-10 and states[trap.holder]['age'] >= self.retire_age):
                trap.incoming = best

    def _safe_food(self, group, point):
        trap = group.trap
        if trap is None: return True
        return dot(sub(point, trap.anchor), trap.normal) >= 18 and math.dist(point, trap.anchor) < 350

    def _forage(self, group, aid, state):
        p = self.poses[aid].p
        if state['energy'] > state['max_energy']-65: return p, 'save_food'
        fruits = [f for f in group.food if f.kind == 'Fruit' and self._safe_food(group, f.p)
                  and (self.time-f.first >= self.ripeness_wait or state['energy'] < 85)
                  and id(f) not in group.fruit_claims]
        if fruits:
            fruit = min(fruits, key=lambda f: math.dist(p, f.p))
            group.fruit_claims.add(id(fruit))
            return fruit.p, 'forage'
        # Visit a remembered tree, then scan while waiting for native fruit.
        trees = [f for f in group.food if f.kind == 'Tree' and self._safe_food(group, f.p)]
        if trees:
            tree = min(trees, key=lambda f: math.dist(p, f.p))
            if math.dist(p, tree.p) > 45: return tree.p, 'find_fruit'
            # Keep away from unripe fruit, avoiding involuntary young collection.
            return p, 'scan_orchard'
        if group.trap:
            trap = group.trap
            tangent = (-trap.normal[1], trap.normal[0])
            phase = int(self.time/12+aid) % 4
            return add(trap.anchor, add(mul(trap.normal, 130+70*(phase%2)),
                       mul(tangent, 120 if phase<2 else -120))), 'search_protected'
        # Persistent headings prevent the previous turn-per-tick circular search.
        angle = (aid*2.399963 + int(self.time/18)*1.1) % (2*math.pi)
        return add(p, mul((math.cos(angle), math.sin(angle)), 30)), 'explore'

    def act(self, observations: list[dict], sim_time: float) -> list[dict]:
        """Accept only native ObservationResponse dictionaries and public time."""
        self.time = sim_time
        states = {s['agent_id']: s for s in observations}
        if sim_time == 0 and all(not s['observations'] for s in states.values()):
            # The native initial response has no sensor data. One legal idle
            # tick obtains it without inventing coordinates or moving blindly.
            return [dict(agent_id=aid, move_distance=0., move_direction=0.,
                         turn_angle=0., spawn_agent=False) for aid in sorted(states)]
        # Correct established poses before using them to register a newborn.
        for aid in sorted(states.keys() & self.poses.keys()): self._map(aid, states[aid])
        self._register(states)
        for aid in sorted(states): self._map(aid, states[aid])
        actions = []
        self.decisions = {}
        for group in self.groups:
            live = group.members & states.keys()
            if not live: continue
            predators = self._predators(group, states)
            if group.trap:
                trap = group.trap
                unsafe = any(dot(sub(p,trap.anchor),trap.normal)>-10 and
                             math.dist(p,trap.anchor)<120 for p in predators)
                if unsafe:
                    self._event('abandon_protected_side_intrusion',holder=trap.holder)
                    self.metrics['abandoned'] += 1
                    group.trap = None
                    group.wall_cooldown = self.time+30
            self._recognize(group, states, predators)
            self._roles(group, states)
            group.fruit_claims = set()
            trap = group.trap
            active = len(live-group.retired)
            # Renew the oldest forager before an old active cohort consumes all
            # replacement slots. Keep two active workers until a birth arrives.
            if active>=3:
                old_workers = [aid for aid in live-group.retired
                    if (not trap or aid not in (trap.holder,trap.incoming)) and states[aid]['age']>75]
                if old_workers:
                    oldest=max(old_workers,key=lambda aid:states[aid]['age'])
                    group.retired.add(oldest); active-=1
            births_planned = 0
            for aid in sorted(live):
                state, pose = states[aid], self.poses[aid]
                if trap and aid in (trap.holder, trap.incoming):
                    target, rule = trap.anchor, 'hold' if aid == trap.holder else 'incoming'
                elif aid in group.retired:
                    if trap:
                        target = add(trap.anchor, mul(trap.normal, 280))
                    else: target = pose.p
                    rule = 'retire'
                else:
                    target, rule = self._forage(group, aid, state)
                # Opposite-face predators are intentionally retained. A second
                # predator on the protected side gets ordinary collision-aware
                # escape, including the holder abandoning an unsafe wall.
                threats = [p for p in predators if math.dist(p, pose.p) < 90 and
                           not (trap and dot(sub(p, trap.anchor), trap.normal) < -18)]
                sprint = False
                look_at = None
                assess_rest = False
                if threats and not (trap and trap.stage == 'acquire'):
                    nearest = min(threats, key=lambda p: math.dist(p, pose.p))
                    target = add(pose.p, mul(unit(sub(pose.p, nearest)), 30))
                    sprint = math.dist(nearest, pose.p) < 65 and state['energy'] > state['max_energy']/5+8
                    look_at = nearest
                    assess_rest = (self.wall and trap is None and math.dist(nearest,pose.p)>45 and
                        any(69.9<=math.dist(*e)<250 for e in group.edges) and
                        any(14.9<=math.dist(*e)<=35.1 for e in group.edges) and
                        any(math.dist(t['p'],nearest)<2 and self.time-t['still']<.3 for t in group.predator_tracks))
                    rule = 'escape_second_predator' if trap else 'escape'
                    self.metrics['emergency_flees'] += 1
                waypoint = self._waypoint(group, aid, target)
                delta = sub(waypoint, pose.p)
                modifier = dict(forest=1., grassland=1., swamp=.5, desert=.8, river=.3)[state['biome']]
                speed = min(state['speed'], state['sprint_speed'])
                if sprint: speed = state['sprint_speed']
                if assess_rest: speed = min(speed,4.)
                distance = min(speed, norm(delta)/modifier)
                direction = wrap(math.atan2(delta[1], delta[0])-pose.theta) if norm(delta)>.01 else 0.
                turn = 0.
                if distance > .1:
                    turn = direction
                elif rule in ('scan_orchard', 'save_food', 'hold', 'retire') and int(round(sim_time*10)) % 10 == 0:
                    turn = .7
                if look_at is not None:
                    v = sub(look_at,pose.p)
                    turn = wrap(math.atan2(v[1],v[0])-pose.theta)
                spawn = (aid not in group.retired and (not trap or aid != trap.incoming)
                         and active+births_planned < self.active_cap and len(live)+births_planned < 6
                         and state['age'] >= 25 and state['energy'] > 175 and not threats)
                # Retire old workers only when there is a younger reserve.
                if not trap and state['age'] > self.retire_age and active > 2:
                    group.retired.add(aid); active -= 1
                if spawn:
                    births_planned += 1
                    self.metrics['predicted_births'] += 1
                action = dict(agent_id=aid, move_distance=distance, move_direction=direction,
                              turn_angle=turn, spawn_agent=spawn)
                actions.append(action)
                self.decisions[aid] = dict(rule=rule, detail='Observation map, remembered food, legal birth/rotation.')
                # Prediction only. Next visible static edge corrects collision
                # displacement. No engine post-action position enters policy.
                pose.p = add(pose.p, rot((distance*modifier, 0), pose.theta+direction))
                pose.theta = wrap(pose.theta+turn)
        return sorted(actions, key=lambda a: a['agent_id'])

"""Society baseline: verbatim copy of Oscar's research controller
(nordic-ai-cup-2026/survival/research/society/society.py, v11+ line, 17 Sept 2026).

Observation-only foraging, predator-aware retreat and generation management.
The trapper policy layers trap roles on top of this and writes the actions it
overrides back into ``Mind.last_action`` so the odometry stays correct.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from src.utils.DTOs import ActionRequest

TAU = 2*math.pi
MOVE_PENALTY = dict(forest=1.0, grassland=1.0, swamp=0.5, desert=0.8, river=0.3)


def wrap(a): return (a+math.pi) % TAU-math.pi
def add(a, b): return (a[0]+b[0], a[1]+b[1])
def sub(a, b): return (a[0]-b[0], a[1]-b[1])
def mul(a, s): return (a[0]*s, a[1]*s)
def norm(a): return math.hypot(a[0], a[1])
def rot(a, t):
    c, s = math.cos(t), math.sin(t)
    return (a[0]*c-a[1]*s, a[0]*s+a[1]*c)
def segments_cross(a, b, c, d):
    def orient(p, q, r): return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
def point_segment(p, a, b):
    v = sub(b, a); w = sub(p, a)
    t = max(0., min(1., (v[0]*w[0]+v[1]*w[1])/max(v[0]*v[0]+v[1]*v[1], 1e-9)))
    return math.dist(p, add(a, mul(v, t)))


@dataclass
class Pose:
    p: tuple = (0., 0.)
    theta: float = 0.
    def transform(self, local): return add(self.p, rot(local, self.theta))
    def polar(self, o):
        return self.transform((math.cos(o['angle'])*o['distance'], math.sin(o['angle'])*o['distance']))
    def local(self, p):  # world point -> (distance, relative angle)
        d = sub(p, self.p)
        return norm(d), wrap(math.atan2(d[1], d[0])-self.theta)


@dataclass
class Food:
    p: tuple
    first: float
    last: float
    fresh: bool = False
    shared: bool = False


@dataclass
class TreeMem:
    p: tuple
    first: float
    last: float
    fruit_seen: float = -1e9
    waited: float = 0.
    barren_until: float = -1.


@dataclass
class Track:
    p: tuple
    v: tuple
    last: float
    rel_dir: float = 0.
    seen_ticks: int = 1
    still_ticks: int = 0
    seen_now: bool = True


@dataclass
class Mind:
    born: float
    pose: Pose = field(default_factory=Pose)
    fruits: list = field(default_factory=list)
    trees: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    edges: list = field(default_factory=list)      # (a, b, last_seen)
    visited: dict = field(default_factory=dict)
    last_action: tuple | None = None
    prev_p: tuple = (0., 0.)
    prev_hear: float = 50.
    explore: tuple = (None, -1.)
    detour: tuple = (None, -1.)
    hist: list = field(default_factory=list)
    wall_ticks: int = 0
    retreat_ticks: int = 0
    last_threat: float = -1e9
    leave_until: float = -1.
    leave_from: tuple | None = None
    sprinting: bool = False
    retreat_heading: float = 0.
    target_key: tuple | None = None
    best_d: float = 1e9
    no_progress: int = 0
    blocked: list = field(default_factory=list)   # (point, until)
    river_heading: float | None = None
    river_ticks: int = 0
    river_left: float = -1e9
    land_p: tuple = (0., 0.)
    land_known: bool = False
    river_cells: dict = field(default_factory=dict)
    last_food: float = -1e9
    river_dir: float = 0.
    river_dir_time: float = -1e9
    emergency: bool = False
    range_hist: list = field(default_factory=list)
    blocked_heading: float | None = None
    blocked_time: float = -1e9
    walk_ticks: int = 0
    roam_hist: list = field(default_factory=list)
    stall_heading: float | None = None
    stall_time: float = -1e9
    young: int = 0


class SocietyPolicy:
    def __init__(self, seed=0, *, pop_cap=6, min_pop=3, cap_halflife=1500., mature_age=30.,
                 elder_age=62., ripen_wait=19., breed_reserve=200., elder_reserve=110.,
                 quiet_time=6., see_margin=0.25, charge_radius=90., stale_margin=15.,
                 track_memory=6., explore_persist=8., hungry=0.30, glance_every=10, near_threat=110., scan_turn=0.5, fit_slack=1.0, share=0, share_radius=150., glance_walk=25, **_):
        self.rng = random.Random(seed)
        self.P = dict(pop_cap=pop_cap, min_pop=min_pop, cap_halflife=cap_halflife, mature_age=mature_age,
                      elder_age=elder_age, ripen_wait=ripen_wait, breed_reserve=breed_reserve,
                      elder_reserve=elder_reserve, quiet_time=quiet_time, see_margin=see_margin,
                      charge_radius=charge_radius, stale_margin=stale_margin, track_memory=track_memory,
                      explore_persist=explore_persist, hungry=hungry, glance_every=glance_every, near_threat=near_threat, scan_turn=scan_turn, fit_slack=fit_slack, share=share, share_radius=share_radius, glance_walk=glance_walk)
        self.time = 0.
        self.minds: dict[int, Mind] = {}
        self.decisions: dict[int, tuple] = {}
        self.metrics = dict(births_requested=0, flee_ticks=0, sprint_ticks=0, wait_ticks=0, explore_ticks=0,
                            leave_ticks=0, ignored_predator_ticks=0, glances=0, stuck_events=0,
                            barren_marks=0, phantom_fruit=0, phantom_tree=0, deflections=0)

    # ------------------------------------------------------------ memory
    def _odometry(self, m: Mind, s):
        """Advance the pose by last tick's action (engine order: move, then turn)."""
        act = m.last_action
        if act is None: return
        dist, direction, turn, biome, energy_before, speed, sprint, max_e = act
        d = max(0., min(dist, sprint))
        if energy_before < max_e/5 and d > speed: d = speed
        d *= MOVE_PENALTY.get(biome, 1.0)
        pose = m.pose
        ang = pose.theta+direction
        if d > 0:
            near = [(a, b) for a, b, t in m.edges if self.time-t < 3. and point_segment(pose.p, a, b) < d+8]
            def blocked(q):
                for a, b in near:
                    if point_segment(q, a, b) < 5.5 or segments_cross(pose.p, q, a, b): return True
                return False
            q = add(pose.p, (d*math.cos(ang), d*math.sin(ang)))
            if near and blocked(q):
                step = math.pi/18; moved = False
                for i in range(36):
                    a = ang+step*((i+1)//2)*(-1)**i
                    q2 = add(pose.p, (d*math.cos(a), d*math.sin(a)))
                    if not blocked(q2):
                        q = q2; moved = True; break
                if not moved: q = pose.p
                self.metrics['deflections'] += 1
            pose.p = q
        pose.theta = wrap(pose.theta+turn)

    def _observe(self, m: Mind, s):
        pose = m.pose; hearing = s['hearing_radius']; cone = s['vision_angle']; vr = s['vision_range']
        obs = s['observations']
        # edges (own frame, short memory)
        for o in obs:
            if o['type'] != 'Edge': continue
            a, b = map(pose.transform, map(tuple, o['coords']))
            for k, (ea, eb, t) in enumerate(m.edges):
                if (math.dist(a, ea) < 6 and math.dist(b, eb) < 6) or (math.dist(a, eb) < 6 and math.dist(b, ea) < 6):
                    m.edges[k] = (a, b, self.time); break
            else:
                m.edges.append((a, b, self.time))
        m.edges = [e for e in m.edges if self.time-e[2] < 40.][-150:]
        # fruit
        seen = []
        for o in obs:
            if o['type'] != 'Fruit': continue
            p = pose.polar(o)
            f = next((f for f in m.fruits if math.dist(f.p, p) < 4), None)
            if f is None:
                fresh = math.dist(m.prev_p, p) < m.prev_hear-3   # was inside hearing last tick: it just spawned
                f = Food(p, self.time, self.time, fresh); m.fruits.append(f)
                for t in m.trees:
                    if math.dist(t.p, p) < 70: t.waited = 0.; t.barren_until = -1.
            f.last = self.time; f.p = p; seen.append(f); m.last_food = self.time
            for t in m.trees:
                if math.dist(t.p, p) < 70: t.fruit_seen = self.time
        kept = []
        for f in m.fruits:
            if f in seen: f.shared = False; kept.append(f); continue
            if math.dist(f.p, pose.p) < hearing-3: self.metrics['phantom_fruit'] += 1; continue
            if f.shared:
                d, ang = pose.local(f.p)
                if d < min(vr, 200)-30 and abs(ang) < cone/2-0.12: self.metrics['phantom_shared'] = self.metrics.get('phantom_shared', 0)+1; continue
            if self.time-f.last > 45.: continue
            kept.append(f)
        m.fruits = kept
        # trees
        seen_t = []
        for o in obs:
            if o['type'] != 'Tree': continue
            p = pose.polar(o)
            t = next((t for t in m.trees if math.dist(t.p, p) < 12), None)
            if t is None:
                t = TreeMem(p, self.time, self.time); m.trees.append(t)
            t.last = self.time; t.p = p; seen_t.append(t)
        kept = []
        for t in m.trees:
            if t in seen_t: kept.append(t); continue
            d, ang = pose.local(t.p)
            if d < min(vr, 200)-30 and abs(ang) < cone/2-0.12 and d > hearing:
                self.metrics['phantom_tree'] += 1; continue   # should be visible, is not: gone or misplaced
            if self.time-t.last > 120.: continue
            kept.append(t)
        m.trees = kept
        # predators
        for tr in m.tracks: tr.seen_now = False
        for o in obs:
            if o['type'] != 'Predator': continue
            p = pose.polar(o)
            tr = min((t for t in m.tracks if math.dist(t.p, p) < 70), key=lambda t: math.dist(t.p, p), default=None)
            if tr is None:
                m.tracks.append(Track(p, (0., 0.), self.time, o['rel_dir']))
            else:
                dt = self.time-tr.last
                moved = math.dist(p, tr.p)
                if 0 < dt <= .15:
                    v = mul(sub(p, tr.p), 1/dt)
                    tr.v = (0.5*tr.v[0]+0.5*v[0], 0.5*tr.v[1]+0.5*v[1]) if tr.seen_ticks > 1 else v
                    tr.still_ticks = tr.still_ticks+1 if moved < 1.0 else 0
                else:
                    tr.v = (0., 0.); tr.still_ticks = 0
                tr.p = p; tr.last = self.time; tr.rel_dir = o['rel_dir']; tr.seen_ticks += 1; tr.seen_now = True
            if o['distance'] < 260: m.last_threat = self.time
        m.tracks = [t for t in m.tracks if self.time-t.last < self.P['track_memory']]
        # visited cells (own frame)
        m.visited[(int(pose.p[0]//100), int(pose.p[1]//100))] = self.time
        m.prev_p = pose.p; m.prev_hear = hearing

    # ------------------------------------------------------------ helpers
    def _wall_repulsion(self, m: Mind, vec, radius=40., max_age=1.5):
        pose = m.pose
        for a, b, t in m.edges:
            if self.time-t > max_age: continue
            d = point_segment(pose.p, a, b)
            if d < radius:
                ab = sub(b, a); px, py = pose.p
                tt = max(0., min(1., ((px-a[0])*ab[0]+(py-a[1])*ab[1])/max(ab[0]**2+ab[1]**2, 1e-9)))
                c = add(a, mul(ab, tt)); away = sub(pose.p, c); n = norm(away)
                if n < 1e-6: continue
                vec = add(vec, mul(away, 2.2*(radius-d)/radius/n))
        return vec

    def _goto(self, m: Mind, s, target, stop=6., speed=None):
        pose = m.pose
        d, ang = pose.local(target)
        if d <= stop: return 0., 0., 0.
        direction = ang
        walk = speed if speed is not None else min(s['speed'], s['sprint_speed'])
        return min(walk, max(0., d-stop*0.5)), direction, max(-.35, min(.35, direction))

    def _progress(self, m: Mind, target, d, tree=None):
        """Track approach progress toward a target; after 15 ticks without gaining 2 units, block it and detour."""
        key = (round(target[0]/10), round(target[1]/10))
        if m.target_key != key:
            m.target_key = key; m.best_d = d; m.no_progress = 0; return
        if d < m.best_d-2.: m.best_d = d; m.no_progress = 0
        else: m.no_progress += 1
        if m.no_progress >= 15:
            self.metrics['stuck_events'] += 1
            m.blocked.append((target, self.time+40.))
            if tree is not None: tree.barren_until = self.time+60.
            m.detour = (m.pose.theta+self.rng.choice([-1, 1])*math.pi/2+self.rng.uniform(-.4, .4), self.time+2.)
            m.target_key = None; m.no_progress = 0

    def _share(self, states):
        """When A sees B, B's frame can be expressed in A's frame from that one
        observation (position from distance/angle, heading from rel_dir). Copy B's recent
        fruit, productive trees and predator tracks into A. No global frame is needed;
        copied fruit is verified by A's own senses like anything else it remembers."""
        if not self.P['share']: return
        R = self.P['share_radius']
        for aid, s in states.items():
            A = self.minds[aid]; pa = A.pose
            if A.fruits: continue   # only agents with nothing of their own take second-hand knowledge
            for o in s['observations']:
                if o['type'] != 'Agent' or o['id'] not in self.minds or o['distance'] < 1e-6: continue
                B = self.minds[o['id']]; pb = B.pose
                pB_A = pa.polar(o)
                thB_A = wrap(math.atan2(pa.p[1]-pB_A[1], pa.p[0]-pB_A[0])-o['rel_dir'])
                dth = wrap(thB_A-pb.theta)
                shift = sub(pB_A, rot(pb.p, dth))
                T = lambda q: add(rot(q, dth), shift)
                n = 0
                for f in B.fruits:
                    if self.time-f.last > 15.: continue
                    q = T(f.p)
                    if math.dist(q, pa.p) > R: continue
                    if any(math.dist(q, g.p) < 6 for g in A.fruits): continue
                    A.fruits.append(Food(q, f.first, f.last, f.fresh, True)); n += 1
                    if n >= 3: break
                for t in B.trees:
                    if self.time-t.fruit_seen > 30. or t.barren_until > self.time: continue
                    q = T(t.p)
                    if math.dist(q, pa.p) > R or any(math.dist(q, g.p) < 15 for g in A.trees): continue
                    A.trees.append(TreeMem(q, t.first, t.last, t.fruit_seen))
                for tr in B.tracks:
                    if self.time-tr.last > 2.: continue
                    q = T(tr.p)
                    if any(math.dist(q, g.p) < 60 for g in A.tracks): continue
                    A.tracks.append(Track(q, rot(tr.v, dth), tr.last, tr.rel_dir, tr.seen_ticks, 0, False))
                self.metrics['shared_items'] = self.metrics.get('shared_items', 0)+n

    def _watch_turn(self, m: Mind):
        """Bearing to keep a recently seen predator inside the cone, or None."""
        pose = m.pose; best = None
        for tr in m.tracks:
            age = self.time-tr.last
            if age > 6.: continue
            p = add(tr.p, mul(tr.v, min(age, 1.0))) if (not tr.seen_now and tr.seen_ticks > 1) else tr.p
            d, ang = pose.local(p)
            if d < 220 and (best is None or d < best[0]): best = (d, ang)
        if best is None: return None
        return max(-.8, min(.8, best[1])) if abs(best[1]) < 2.6 else best[1]

    def _idle_turn(self, m: Mind):
        """Turn while standing: face the nearest recently seen predator (keeps it in the
        cone and makes it circle instead of charge), otherwise scan quickly."""
        pose = m.pose; best = None
        for tr in m.tracks:
            age = self.time-tr.last
            if age > 4.: continue
            p = add(tr.p, mul(tr.v, min(age, 1.0))) if (not tr.seen_now and tr.seen_ticks > 1) else tr.p
            d, ang = pose.local(p)
            if d < 260 and (best is None or d < best[0]): best = (d, ang)
        if best is not None:
            return max(-.8, min(.8, best[1])) if abs(best[1]) < 2.6 else best[1]
        return self.P['scan_turn']

    @staticmethod
    def _fitness(s):
        """Trait quality from the observation fields (traits mutate +-50 % at 10 % per
        birth, so a lineage drifts deaf, blind or slow within ~15 generations)."""
        c = lambda v, lo, hi: max(lo, min(hi, v))
        return (c((s['sprint_speed']-12.)/10., -2., 1.5)+c((s['speed']-8.)/6., -1.5, 1.)
                + c((s['hearing_radius']-40.)/20., -2., 1.)+c((s['vision_range']-160.)/60., -1.5, 1.)
                + c((s['vision_angle']-0.85)/0.4, -1.5, 1.)+0.5*c((s['max_energy']-400.)/200., -1., 1.))

    def _cap(self):
        c = self.P['pop_cap']*0.5**(self.time/self.P['cap_halflife'])
        return int(max(self.P['min_pop'], min(self.P['pop_cap'], round(c))))

    def _retreat_heading(self, m: Mind, threats):
        """Best of 16 headings: away from every threat (closer ones weigh more), off
        walls, and committed to the previous retreat heading so two predators cannot
        make us jiggle in place."""
        pose = m.pose; best = None
        prev = m.retreat_heading if m.retreat_ticks > 1 else None
        rep = []
        for d, ang, p, tr in threats:
            w = min(4., (140./max(d, 20.))**2)
            rep.append((w, wrap(pose.theta+ang+math.pi)))
        wsum = sum(w for w, _ in rep) or 1.
        base = rep[0][1] if rep else pose.theta
        for k in range(16):
            h = base+k*TAU/16
            score = sum(w*math.cos(h-a) for w, a in rep)/wsum
            if prev is not None: score += 0.45*math.cos(h-prev)
            end_ = add(pose.p, (60*math.cos(h), 60*math.sin(h)))
            mid_ = add(pose.p, (25*math.cos(h), 25*math.sin(h)))
            if m.blocked_heading is not None and self.time-m.blocked_time < 3.:
                score -= 2.0*max(0., math.cos(h-m.blocked_heading))
            for a, b, t in m.edges:
                if segments_cross(pose.p, end_, a, b) or point_segment(end_, a, b) < 8:
                    score -= 1.5 if (segments_cross(pose.p, mid_, a, b) or point_segment(mid_, a, b) < 8) else 0.7
                    break
                dwall = point_segment(pose.p, a, b)
                if dwall < 40:
                    # pinned against a wall: prefer headings that gain clearance
                    ab = sub(b, a); px, py = pose.p
                    tt = max(0., min(1., ((px-a[0])*ab[0]+(py-a[1])*ab[1])/max(ab[0]**2+ab[1]**2, 1e-9)))
                    c = add(a, mul(ab, tt)); aw = math.atan2(py-c[1], px-c[0])
                    score += 0.4*(1-dwall/40)*math.cos(h-aw)
            if best is None or score > best[0]: best = (score, h)
        return best[1]

    # ------------------------------------------------------------ main
    def __call__(self, states_list, sim_time):
        self.time = sim_time
        states = {s['agent_id']: s for s in states_list}
        for aid in list(self.minds):
            if aid not in states: del self.minds[aid]
        for aid, s in states.items():
            m = self.minds.get(aid)
            if m is None:
                m = self.minds[aid] = Mind(born=sim_time, last_food=sim_time)
            else:
                self._odometry(m, s)
            self._observe(m, s)
        self._share(states)
        young_pop = sum(1 for x in states.values() if x['age'] <= self.P['elder_age'])
        cap = self._cap()
        fit = {aid: self._fitness(x) for aid, x in states.items()}
        best_fit = max(fit.values()) if fit else 0.
        sound = {aid: (x['hearing_radius'] >= 40. and x['sprint_speed'] >= 15. and x['vision_range'] >= 150.
                       and x['vision_angle'] >= 0.8 and x['speed'] >= 8.) for aid, x in states.items()}
        any_sound = any(sound.values())
        for m in self.minds.values(): m.emergency = young_pop == 0; m.young = young_pop
        births = 0
        actions = []
        for s in sorted(states.values(), key=lambda s: (-s['age'], s['agent_id'])):
            aid = s['agent_id']; m = self.minds[aid]; pose = m.pose
            energy, max_e = s['energy'], s['max_energy']
            walk = min(s['speed'], s['sprint_speed']); sprint = s['sprint_speed']
            can_sprint = energy > max_e/5+4
            elder = s['age'] > self.P['elder_age']
            spawn = False; dist = direction = turn = 0.; mode = 'idle'
            margin = self.P['stale_margin']
            # ---- threat assessment ----
            # every remembered predator is a threat when close (it can turn on us in a
            # tick); further out only if it can perceive us (rel_dir = our bearing in its
            # frame; its cone is 60 degrees to 250, hearing 60)
            threats = []     # (distance, bearing, world point, track)
            resting = None
            for tr in m.tracks:
                age = self.time-tr.last
                p = add(tr.p, mul(tr.v, min(age, 1.0))) if (not tr.seen_now and tr.seen_ticks > 1) else tr.p
                d, ang = pose.local(p)
                if tr.seen_now and tr.still_ticks >= 3:
                    if resting is None or d < resting[0]: resting = (d, ang, p, tr)
                    continue
                sees_me = d < 62+margin or (abs(tr.rel_dir) < math.pi/6+self.P['see_margin'] and d < 250+margin)
                if not tr.seen_now and age > 1.5: sees_me = sees_me and d < 120
                if d < self.P['near_threat'] or sees_me:
                    threats.append((d, ang, p, tr))
                elif tr.seen_now:
                    self.metrics['ignored_predator_ticks'] += 1
            threats.sort(key=lambda x: x[0])
            engaged = threats[0] if threats else None
            if engaged is not None:
                d, ang, p, tr = engaged
                mode = 'flee'; self.metrics['flee_ticks'] += 1; m.retreat_ticks += 1
                heading = self._retreat_heading(m, threats)
                m.retreat_heading = heading
                direction = wrap(heading-pose.theta)
                dm = d-margin
                m.range_hist.append(d)
                if len(m.range_hist) > 4: m.range_hist.pop(0)
                if m.sprinting and len(m.range_hist) == 4 and m.range_hist[-1] <= m.range_hist[0]+2 and tr.seen_now:
                    # four sprint ticks straight away from it and no gap opened: we are boxed in
                    m.blocked_heading = heading; m.blocked_time = self.time; m.range_hist.clear()
                    self.metrics['flee_blocked'] = self.metrics.get('flee_blocked', 0)+1
                    heading = self._retreat_heading(m, threats); m.retreat_heading = heading
                    direction = wrap(heading-pose.theta)
                if dm < self.P['charge_radius'] and can_sprint:
                    dist = sprint; m.sprinting = True; self.metrics['sprint_ticks'] += 1
                elif m.sprinting and dm < self.P['charge_radius']+25 and can_sprint:
                    dist = sprint; self.metrics['sprint_ticks'] += 1
                else:
                    dist = walk; m.sprinting = False
                if dm < self.P['charge_radius']:
                    # it charges regardless of facing: look where we run (walls ahead)
                    turn = max(-1.2, min(1.2, direction)) if abs(direction) < 2.6 else direction
                elif self.P['glance_every'] and m.retreat_ticks % self.P['glance_every'] == 0:
                    turn = direction if abs(direction) < 3.1 else 3.1; self.metrics['glances'] += 1
                else:
                    # face the nearest predator (it circles instead of charging); keep it in our cone
                    turn = ang if abs(ang) > 2.6 else max(-.8, min(.8, ang))
            else:
                m.sprinting = False; m.retreat_ticks = 0; m.range_hist.clear()
                if resting is not None and resting[0] < 150:
                    d, ang, p, tr = resting
                    mode = 'leave'; self.metrics['leave_ticks'] += 1
                    away_ang = wrap(pose.theta+ang+math.pi)
                    heading = self._retreat_heading(m, [resting])
                    m.retreat_heading = heading
                    direction = wrap(heading-pose.theta); dist = walk
                    turn = max(-.5, min(.5, direction))
                else:
                    dist, direction, turn, mode = self._forage(m, s, elder)
                    if dist > 0:
                        watch = self._watch_turn(m)
                        m.walk_ticks += 1
                        if watch is not None:
                            turn = watch; self.metrics['watch_ticks'] = self.metrics.get('watch_ticks', 0)+1
                        elif self.P['glance_walk'] and m.walk_ticks % self.P['glance_walk'] == 0:
                            turn = math.pi; self.metrics['walk_glances'] = self.metrics.get('walk_glances', 0)+1
                        elif self.P['glance_walk'] and m.walk_ticks % self.P['glance_walk'] == 1 and m.walk_ticks > 1:
                            turn = wrap(direction)   # come back to the travel direction
                # ---- reproduction ----
                quiet = self.time-m.last_threat > self.P['quiet_time']
                food_in_sight = sum(1 for f in m.fruits if math.dist(f.p, pose.p) < 80) >= 2 or \
                    any(self.time-t.fruit_seen < 20 and math.dist(t.p, pose.p) < 90 for t in m.trees)
                emergency = len(states) <= 2
                # selective breeding: only lineages near the best trait quality reproduce
                # (a colony of two or fewer young breeds regardless)
                good_genes = (sound[aid] if any_sound else fit[aid] >= best_fit-self.P['fit_slack']) or young_pop < 2
                if births < 2 and quiet and food_in_sight and good_genes:
                    if emergency and energy > 140:
                        spawn = True
                    elif young_pop+births < cap:
                        if elder and energy > self.P['elder_reserve']:
                            spawn = True
                        elif s['age'] > self.P['mature_age'] and energy > self.P['breed_reserve']:
                            spawn = True
                if spawn:
                    births += 1; self.metrics['births_requested'] += 1
            # ---- stuck detection against a visible wall ----
            ahead = None
            if dist > 0:
                hx, hy = math.cos(pose.theta+direction), math.sin(pose.theta+direction)
                for o in s['observations']:
                    if o['type'] != 'Edge': continue
                    (x1, y1), (x2, y2) = o['coords']
                    dl = point_segment((0., 0.), (x1, y1), (x2, y2))
                    if dl < 12:
                        # local frame: our facing is +x; direction is relative to facing
                        cx, cy = math.cos(direction), math.sin(direction)
                        vx, vy = x2-x1, y2-y1
                        tt = max(0., min(1., -(x1*vx+y1*vy)/max(vx*vx+vy*vy, 1e-9)))
                        qx, qy = x1+tt*vx, y1+tt*vy
                        if cx*qx+cy*qy > 0: ahead = (qx, qy); break
            m.wall_ticks = m.wall_ticks+1 if ahead is not None else 0
            if m.wall_ticks >= 15 and mode not in ('flee',):
                self.metrics['stuck_events'] += 1; m.wall_ticks = 0
                m.detour = (pose.theta+self.rng.uniform(math.pi/2, 3*math.pi/2), self.time+3.)
                for t in m.trees:
                    if math.dist(t.p, pose.p) < 120: t.barren_until = self.time+60.
                m.explore = (None, -1.)
            if m.detour[0] is not None and self.time <= m.detour[1] and mode not in ('flee',):
                h = m.detour[0]
                direction = wrap(h-pose.theta); dist = walk; turn = max(-.3, min(.3, direction))
                mode = 'detour'
            self.decisions[aid] = (mode, round(dist, 1), round(engaged[0], 1) if engaged else None,
                                   len(m.fruits), len(m.trees), len(m.tracks))
            m.last_action = (dist, direction, turn, s['biome'], energy, s['speed'], s['sprint_speed'], max_e)
            actions.append((aid, ActionRequest(agent_id=aid, move_distance=float(dist), move_direction=float(direction),
                                               turn_angle=float(turn), spawn_agent=bool(spawn))))
        return actions

    # ------------------------------------------------------------ foraging
    def _forage(self, m: Mind, s, elder):
        pose = m.pose; energy, max_e = s['energy'], s['max_energy']
        walk = min(s['speed'], s['sprint_speed'])
        hungry = energy < max_e*self.P['hungry']
        want = energy < max_e-65.
        if elder:
            # elders leave the food to the young; their death is free. A small colony
            # (two or fewer young) or fruit within a few steps makes them eat again.
            at_hand = any(math.dist(f.p, pose.p) < 40 for f in m.fruits)
            want = m.young <= 2 or (at_hand and energy < max_e-65.)
            hungry = m.young <= 2
        danger = [tr.p for tr in m.tracks if self.time-tr.last < 5.]
        m.blocked = [b for b in m.blocked if b[1] > self.time]
        def safe(p): return all(math.dist(p, q) > 90 for q in danger) and all(math.dist(p, b[0]) > 12 for b in m.blocked)
        others = [(pose.polar(o), o['id']) for o in s['observations'] if o['type'] == 'Agent']
        aid = s['agent_id']
        in_river = s['biome'] == 'river'
        if in_river:
            if m.river_ticks == 0:
                m.river_dir = self._last_heading(m); m.river_dir_time = self.time
            m.river_ticks += 1
            m.river_cells[(int(pose.p[0]//100), int(pose.p[1]//100))] = self.time
        else:
            if m.river_ticks > 0: m.river_left = self.time
            m.river_ticks = 0; m.land_p = pose.p; m.land_known = True
        # 1. fruit
        if want:
            cands = []
            for f in m.fruits:
                if not safe(f.p): continue
                d = math.dist(f.p, pose.p)
                if in_river and d > 60: continue
                ripe = (not f.fresh) or (self.time-f.first >= self.P['ripen_wait'])
                if ripe or hungry: cands.append((d, f))
            if cands:
                d, f = min(cands, key=lambda x: x[0])
                dist, direction, turn = self._goto(m, s, f.p, stop=0.)
                self._progress(m, f.p, d)
                return min(walk, d+2), direction, turn, 'fruit'
            # a fresh, unripe fruit nearby: wait for it just outside auto-collection range
            fresh = [(math.dist(f.p, pose.p), f) for f in m.fruits if f.fresh and safe(f.p)]
            if fresh and not hungry and not in_river:
                d, f = min(fresh, key=lambda x: x[0])
                if d < 120:
                    if d > 22:
                        dist, direction, turn = self._goto(m, s, f.p, stop=18.)
                        self._progress(m, f.p, d)
                        return dist, direction, turn, 'waitfruit'
                    m.target_key = None
                    self.metrics['wait_ticks'] += 1
                    return 0., 0., self._idle_turn(m), 'waitfruit'
        # river: water is slow (0.3) at full walking cost. Go back to the last land point
        # while it is close, otherwise commit to a straight crossing.
        desperate = self.time-m.last_food > 25. and self.time-m.born > 5.
        if in_river:
            # water: 30 % speed at full cost. Never dither: hold the heading we entered
            # with, and turn a quarter every 7 s in case we are running along the river.
            self.metrics['river_ticks'] = self.metrics.get('river_ticks', 0)+1
            if m.river_ticks % 70 == 0: m.river_dir += math.pi/2
            direction = wrap(m.river_dir-pose.theta)
            return walk, direction, max(-.4, min(.4, direction)), 'river'
        # low on energy with nothing to eat: sit by a tree (fruit comes to trees) rather than roam
        if energy < 110. and self.time < 60. and not desperate and not elder:
            near = [t for t in m.trees if t.barren_until < self.time and safe(t.p) and math.dist(t.p, pose.p) < 160]
            if near:
                t = min(near, key=lambda t: math.dist(t.p, pose.p))
                d, ang = pose.local(t.p)
                if d > 42:
                    dist, direction, turn = self._goto(m, s, t.p, stop=34.)
                    self._progress(m, t.p, d, tree=t)
                    return dist, direction, turn, 'tree-conserve'
                m.target_key = None
                t.waited += .1
                if t.waited > 25.:
                    t.barren_until = self.time+60.; t.waited = 0.; self.metrics['barren_marks'] += 1
                self.metrics['wait_ticks'] += 1
                return 0., 0., self._idle_turn(m), 'conserve'
        if elder and not want:
            self.metrics['wait_ticks'] += 1
            m.target_key = None
            return 0., 0., self._idle_turn(m), 'elderwait'
        # 2. a productive tree (fruit seen around it recently): sit there, fresh spawns are
        #    heard and can be timed for ripeness
        trees = [t for t in m.trees if t.barren_until < self.time and safe(t.p) and self.time-t.fruit_seen < 30.]
        if trees:
            def key(t):
                crowded = any(math.dist(op, t.p) < 45 and oid < aid for op, oid in others)
                return math.dist(t.p, pose.p)+(250 if crowded else 0)
            t = min(trees, key=key)
            d, ang = pose.local(t.p)
            if d < 320:
                if d > 42:
                    dist, direction, turn = self._goto(m, s, t.p, stop=34.)
                    self._progress(m, t.p, d, tree=t)
                    return dist, direction, turn, 'tree'
                m.target_key = None
                if any(math.dist(op, t.p) < 45 and oid < aid for op, oid in others) and not hungry:
                    t.barren_until = self.time+40.; t.waited = 0.; self.metrics['yield_marks'] = self.metrics.get('yield_marks', 0)+1
                    return 0., 0., 0., 'treeyield'
                t.waited += .1
                if t.waited > (8. if hungry else 15.):
                    t.barren_until = self.time+60.; t.waited = 0.; self.metrics['barren_marks'] += 1
                self.metrics['wait_ticks'] += 1
                return 0., 0., self._idle_turn(m), 'treewait'
        # 3. roam: fruit lies around trees, so head for unvisited ground and known trees
        self.metrics['explore_ticks'] += 1
        m.target_key = None
        h, until = m.explore
        blocked_ahead = h is not None and any(
            segments_cross(pose.p, add(pose.p, (70*math.cos(h), 70*math.sin(h))), a, b)
            for a, b, t in m.edges if self.time-t < 1.0)
        m.roam_hist.append(pose.p)
        if len(m.roam_hist) > 15: m.roam_hist.pop(0)
        stalled = len(m.roam_hist) == 15 and math.dist(m.roam_hist[0], pose.p) < 15 and not in_river
        if stalled:
            self.metrics['roam_stalls'] = self.metrics.get('roam_stalls', 0)+1; m.roam_hist.clear()
            m.stall_heading = h; m.stall_time = self.time
        if h is None or self.time > until or blocked_ahead or stalled:
            best = None
            known = [t for t in m.trees if t.barren_until < self.time]
            # centroid of recently visited ground: roam away from it so the walk expands
            cells = [(c, t) for c, t in m.visited.items() if self.time-t < 240.]
            away = None
            if len(cells) >= 3:
                cx = sum(c[0]*100+50 for c, t in cells)/len(cells); cy = sum(c[1]*100+50 for c, t in cells)/len(cells)
                if math.dist((cx, cy), pose.p) > 30: away = math.atan2(pose.p[1]-cy, pose.p[0]-cx)
            for k in range(16):
                cand = pose.theta+k*TAU/16+self.rng.uniform(-.15, .15)
                score = 0.
                for r in (100, 200, 300):
                    cell = (int((pose.p[0]+r*math.cos(cand))//100), int((pose.p[1]+r*math.sin(cand))//100))
                    score += 1. if cell not in m.visited else max(0., (self.time-m.visited[cell])/300-1)
                end_ = add(pose.p, (200*math.cos(cand), 200*math.sin(cand)))
                if any(segments_cross(pose.p, end_, a, b) for a, b, t in m.edges if self.time-t < 8.): score -= 2.5
                if not desperate:
                    for r in (60, 120, 200, 280):
                        cell = (int((pose.p[0]+r*math.cos(cand))//100), int((pose.p[1]+r*math.sin(cand))//100))
                        if cell in m.river_cells and self.time-m.river_cells[cell] < 240.: score -= 0.8
                if away is not None: score += 0.7*math.cos(cand-away)
                if m.stall_heading is not None and self.time-m.stall_time < 20.: score -= 1.5*max(0., math.cos(cand-m.stall_heading))
                for t in known:
                    dt_, at_ = pose.local(t.p)
                    if 60 < dt_ < 350 and abs(wrap(pose.theta+at_-cand)) < 0.35: score += 0.8
                score += self.rng.uniform(0, .3)
                if best is None or score > best[0]: best = (score, cand)
            h = best[1]; m.explore = (h, self.time+self.P['explore_persist'])
        vec = self._wall_repulsion(m, (math.cos(h), math.sin(h)), radius=45.)
        heading = math.atan2(vec[1], vec[0])
        direction = wrap(heading-pose.theta)
        return walk, direction, max(-.3, min(.3, direction)), 'explore'

    def _last_heading(self, m: Mind):
        act = m.last_action
        if act is None or act[0] <= 0: return m.pose.theta
        return m.pose.theta+act[1]

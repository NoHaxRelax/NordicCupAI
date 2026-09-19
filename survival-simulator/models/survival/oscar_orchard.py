"""Orchard policy: no-predator survival to 3000 s with a maximal ripe-fruit harvest.

Observation-only controller. Inputs are the per-agent observation dictionaries
and simulation time; no engine state, map coordinates, fruit ages or tree ages.

Design
  * Poses: every agent keeps a dead-reckoned pose. Heading never drifts (turns
    are applied exactly); position drifts only on unpredicted collisions and is
    corrected against remembered trees and, once seen, the boundary walls.
  * Groups: newborns are placed in the parent's frame from the parent's own
    observation of the child (distance, angle, rel_dir), so a lineage shares one
    frame and one map (trees, fruit, coverage cells with biome). Groups merge
    when members see each other (agent observations carry ids). A boundary edge
    (length > 1000) anchors a group to absolute coordinates.
  * Posts: each agent is assigned to a live tree (at most `tree_slots` per
    tree), sits on it, sweeps slowly so the whole fruit ring is covered, and
    eats fruit when its known age reaches ripeness (20 s). Fruit age is known
    when the spot was observed shortly before the fruit appeared.
  * Old age: the hidden senescence threshold is detected from an energy drop
    exceeding the known action costs; old agents convert their energy into
    children (100 -> 75) and stop travelling.
  * Population: young population is capped by an estimate of the live tree
    count (which halves every ~600 s); births go to the highest-fitness parents
    (vision range, hearing, max energy) so late-game watchers see farther.
  * Exploration: agents without a post walk to the stalest coverage cell
    (weighted by biome tree-spawn rate) and sweep on arrival.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from src.utils.DTOs import ActionRequest

TAU = 2*math.pi
INF = float('inf')
MOVE_PENALTY = dict(forest=1.0, grassland=1.0, swamp=0.5, desert=0.8, river=0.3)
TREE_RATE = dict(forest=1.0, grassland=0.5, swamp=0.9, desert=0.1, river=0.0)
FRUIT_RATE = dict(forest=0.1, grassland=0.1, swamp=0.08, desert=0.05, river=0.0)
W, H = 1600., 1200.   # used only after a group is anchored to a boundary wall
CELL = 100.


def wrap(a): return (a+math.pi) % TAU-math.pi
def add(a, b): return (a[0]+b[0], a[1]+b[1])
def sub(a, b): return (a[0]-b[0], a[1]-b[1])
def mul(a, s): return (a[0]*s, a[1]*s)
def norm(a): return math.hypot(a[0], a[1])
def rot(a, t):
    c, s = math.cos(t), math.sin(t)
    return (a[0]*c-a[1]*s, a[0]*s+a[1]*c)
def unit(t): return (math.cos(t), math.sin(t))
def segments_cross(a, b, c, d):
    def orient(p, q, r): return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
def point_segment(p, a, b):
    v = sub(b, a); w = sub(p, a)
    t = max(0., min(1., (v[0]*w[0]+v[1]*w[1])/max(v[0]*v[0]+v[1]*v[1], 1e-9)))
    return math.dist(p, add(a, mul(v, t)))
def cell_of(p): return (math.floor(p[0]/CELL), math.floor(p[1]/CELL))


@dataclass
class Pose:
    p: tuple = (0., 0.)
    theta: float = 0.
    def transform(self, local): return add(self.p, rot(local, self.theta))
    def polar(self, o): return self.transform((math.cos(o['angle'])*o['distance'], math.sin(o['angle'])*o['distance']))
    def local(self, p):
        d = sub(p, self.p)
        return norm(d), wrap(math.atan2(d[1], d[0])-self.theta)


@dataclass
class Tree:
    id: int
    p: tuple
    first: float
    last: float
    fresh: bool
    fruit_seen: float = -INF
    dead: bool = False
    assigned: set = field(default_factory=set)
    fruit_here: int = 0
    fruit_free: int = 0
    cluster: tuple = ()
    cluster_t: float = -1.


@dataclass
class Fruit:
    id: int
    p: tuple
    born_lo: float
    born_hi: float
    last: float
    claimed: int | None = None


@dataclass
class Mind:
    aid: int
    group: int
    pose: Pose
    born: float
    last_action: tuple | None = None
    spawned_ok: bool = False
    prev_pose: Pose | None = None
    prev_hear: float = 50.
    prev_cone: float = math.pi/3
    prev_vis: float = 200.
    edges: list = field(default_factory=list)      # (a, b, last_seen) in group frame
    old: bool = False
    old_since: float = INF
    energy_prev: float | None = None
    post: int | None = None
    fruit: int | None = None
    explore: tuple | None = None                   # (target point, until)
    sweep_left: int = 0
    target_key: tuple | None = None
    best_d: float = INF
    no_progress: int = 0
    detour: tuple = (None, -1.)
    blocked: list = field(default_factory=list)    # (point, until)
    sweep_sign: float = 1.
    last_eat: float = -INF
    prev_marks: list = field(default_factory=list)   # static landmark points seen last tick (group frame)
    hear_hist: list = field(default_factory=list)    # (time, position, hearing radius), recent ticks
    heir_done: bool = False
    repost_at: float = 0.
    post_since: float = -INF
    last_site: float = 0.
    watch: tuple | None = None                    # (target point, chosen at)


class Group:
    def __init__(self, gid):
        self.id = gid
        self.agents: set[int] = set()
        self.trees: dict[int, Tree] = {}
        self.fruits: dict[int, Fruit] = {}
        self.tgrid: dict[tuple, set] = {}           # cell -> tree ids
        self.fgrid: dict[tuple, set] = {}           # cell -> fruit ids
        self.cells: dict[tuple, list] = {}          # cell -> [last_observed, biome or None]
        self.anchored = False
        self.next_tree = 0
        self.next_fruit = 0
        self.seen_trees: set[int] = set()
        self.seen_fruits: set[int] = set()

    # spatial hash helpers (cell = CELL units)
    def add_tree(self, t):
        self.trees[t.id] = t; self.tgrid.setdefault(cell_of(t.p), []).append(t)
    def del_tree(self, tid):
        t = self.trees.pop(tid); cell = self.tgrid.get(cell_of(t.p))
        if cell and t in cell: cell.remove(t)
    def move_tree(self, t, p):
        cell = self.tgrid.get(cell_of(t.p))
        if cell and t in cell: cell.remove(t)
        t.p = p; self.tgrid.setdefault(cell_of(p), []).append(t)
    def add_fruit(self, f):
        self.fruits[f.id] = f; self.fgrid.setdefault(cell_of(f.p), []).append(f)
    def del_fruit(self, fid):
        f = self.fruits.pop(fid); cell = self.fgrid.get(cell_of(f.p))
        if cell and f in cell: cell.remove(f)
    def rebuild_grids(self):
        self.tgrid = {}; self.fgrid = {}
        for t in self.trees.values(): self.tgrid.setdefault(cell_of(t.p), []).append(t)
        for f in self.fruits.values(): self.fgrid.setdefault(cell_of(f.p), []).append(f)
    def _near(self, grid, p, r):
        px, py = p; cx = math.floor(px/CELL); cy = math.floor(py/CELL); n = int(r//CELL)+1; out = []
        dist = math.dist
        for dx in range(-n, n+1):
            for dy in range(-n, n+1):
                cell = grid.get((cx+dx, cy+dy))
                if not cell: continue
                for it in cell:
                    if dist(it.p, p) <= r: out.append(it)
        return out
    def near_trees(self, p, r): return self._near(self.tgrid, p, r)
    def near_fruits(self, p, r): return self._near(self.fgrid, p, r)


class OrchardPolicy:
    def __init__(self, seed=0, *, cap_mult=0.3, cap_min=4, cap_max=20, n0=80., tree_half=600.,
                 tree_slots=1, breed_reserve=200., emergency_reserve=105., ripen_wait=20., wait_tol=25., dump_slack=0.3,
                 sweep_rate=0.03, explore_radius=450., old_dump=True, fit_vision=1.0, fit_hear=0.3,
                 fit_energy=0.2, births_per_tick=3, fruit_reach=200., tree_reach=420., hungry=45.,
                 idle_sweep=True, site_min=5., breed_reserve_late=200., reserve_t0=600., reserve_t1=1800., dist_pen=0.1, dump_food=3, vo_win_fruit=4.5, vo_win_far=4.5, vo_cap=12., heirs=1, extra_old=True, heir_age=55., heir_reserve=250., explore_min=60., cull=False, travel_turn=0.25, heir_select=True, heir_slack=0.05, repost_every=10., switch_gain=100., fruit_min_wait=0., no_eat_age=INF, late_still_t=INF, post_radius=30., min_stay=15., hungry_margin=5., fit_speed=0.3, explore_energy=200., watch_patience=30., watch_reach=500., watch_refresh=60., select_min_young=0, heir_at_food=False, dump_food_site=2, dump_mult=1.0,
                 cluster_radius=0., spread_weight=0., feed_mode='hungry', low_pop_reserve=200., lone_reach_mult=1.0,
                 old_reach=60., old_eat_last=True, rot_margin=47., dump_after_t=INF, cap_tree_slack=1, cap_hard_min=2, heir_needs_site=True, nursery_bonus=0., **_):
        self.rng = random.Random(seed)
        self.P = dict(cap_mult=cap_mult, cap_min=cap_min, cap_max=cap_max, n0=n0, tree_half=tree_half,
                      tree_slots=tree_slots, breed_reserve=breed_reserve, emergency_reserve=emergency_reserve,
                      ripen_wait=ripen_wait, wait_tol=wait_tol, sweep_rate=sweep_rate, explore_radius=explore_radius,
                      old_dump=old_dump, fit_vision=fit_vision, fit_hear=fit_hear, fit_energy=fit_energy,
                      births_per_tick=births_per_tick, fruit_reach=fruit_reach, tree_reach=tree_reach, hungry=hungry,
                      idle_sweep=idle_sweep, dump_slack=dump_slack, site_min=site_min, breed_reserve_late=breed_reserve_late,
                      reserve_t0=reserve_t0, reserve_t1=reserve_t1, dist_pen=dist_pen, dump_food=dump_food, vo_win_fruit=vo_win_fruit, vo_win_far=vo_win_far, vo_cap=vo_cap, heirs=heirs, extra_old=extra_old, heir_age=heir_age, heir_reserve=heir_reserve, explore_min=explore_min, cull=cull, travel_turn=travel_turn, heir_select=heir_select, heir_slack=heir_slack, repost_every=repost_every, switch_gain=switch_gain, fruit_min_wait=fruit_min_wait, no_eat_age=no_eat_age, late_still_t=late_still_t, post_radius=post_radius, min_stay=min_stay, hungry_margin=hungry_margin, fit_speed=fit_speed, explore_energy=explore_energy, watch_patience=watch_patience, watch_reach=watch_reach, watch_refresh=watch_refresh, select_min_young=select_min_young, heir_at_food=heir_at_food, dump_food_site=dump_food_site, dump_mult=dump_mult,
                      cluster_radius=cluster_radius, spread_weight=spread_weight, feed_mode=feed_mode, low_pop_reserve=low_pop_reserve,
                      lone_reach_mult=lone_reach_mult, old_reach=old_reach, old_eat_last=old_eat_last, rot_margin=rot_margin, dump_after_t=dump_after_t, cap_tree_slack=cap_tree_slack, cap_hard_min=cap_hard_min, heir_needs_site=heir_needs_site, nursery_bonus=nursery_bonus)
        self.time = 0.
        self.minds: dict[int, Mind] = {}
        self.groups: dict[int, Group] = {}
        self.next_group = 0
        self.last_spawners: list[int] = []
        self.culled: set[int] = set()
        self.decisions: dict[int, tuple] = {}
        self.metrics = dict(births=0, old_births=0, emergency_births=0, pose_corrections=0, anchors=0, merges=0,
                            deflections=0, stuck_events=0, explore_ticks=0, idle_ticks=0, travel_ticks=0,
                            fruit_ticks=0, waits_started=0, old_detected=0, trees_seen=0, fruits_seen=0,
                            fresh_fruits=0, new_groups=0, vo_corrections=0, heir_births=0, starve_ticks=0, post_changes=0, post_dist=0., post_dead=0)

    # ------------------------------------------------------------ groups
    def _new_group(self):
        g = Group(self.next_group); self.groups[g.id] = g; self.next_group += 1
        self.metrics['new_groups'] += 1
        return g

    def _transform_group(self, g: Group, dth, shift):
        T = lambda q: add(rot(q, dth), shift)
        for aid in g.agents:
            m = self.minds[aid]
            m.pose = Pose(T(m.pose.p), wrap(m.pose.theta+dth))
            if m.prev_pose is not None: m.prev_pose = Pose(T(m.prev_pose.p), wrap(m.prev_pose.theta+dth))
            m.edges = [(T(a), T(b), t) for a, b, t in m.edges]
            m.blocked = [(T(p), u) for p, u in m.blocked]
            m.hear_hist = [(ht, T(hp), hr) for ht, hp, hr in m.hear_hist]
            if m.explore is not None: m.explore = (T(m.explore[0]), m.explore[1])
            if m.watch is not None: m.watch = (T(m.watch[0]), m.watch[1])
            if m.detour[0] is not None: m.detour = (wrap(m.detour[0]+dth), m.detour[1])
            m.target_key = None
        for t in g.trees.values(): t.p = T(t.p)
        for f in g.fruits.values(): f.p = T(f.p)
        g.rebuild_grids()
        cells = {}
        for c, v in g.cells.items():
            q = T(((c[0]+.5)*CELL, (c[1]+.5)*CELL)); nc = cell_of(q)
            if nc in cells: cells[nc][0] = max(cells[nc][0], v[0]); cells[nc][1] = cells[nc][1] or v[1]
            else: cells[nc] = list(v)
        g.cells = cells

    def _merge(self, ga: Group, gb: Group, dth, shift):
        """Express gb in ga's frame (rotation dth, translation shift) and absorb it."""
        self._transform_group(gb, dth, shift)
        for aid in gb.agents:
            self.minds[aid].group = ga.id; ga.agents.add(aid)
        for t in gb.trees.values():
            same = next(iter(ga.near_trees(t.p, 12)), None)
            if same is None:
                t.id = ga.next_tree; ga.next_tree += 1; ga.add_tree(t)
                for aid in t.assigned: self.minds[aid].post = t.id
            else:
                same.first = min(same.first, t.first); same.last = max(same.last, t.last)
                same.fruit_seen = max(same.fruit_seen, t.fruit_seen); same.fresh = same.fresh or t.fresh
                for aid in t.assigned: self.minds[aid].post = same.id; same.assigned.add(aid)
        for f in gb.fruits.values():
            same = next(iter(ga.near_fruits(f.p, 4)), None)
            if same is None:
                f.id = ga.next_fruit; ga.next_fruit += 1; ga.add_fruit(f)
                if f.claimed is not None: self.minds[f.claimed].fruit = f.id
            else:
                same.born_lo = max(same.born_lo, f.born_lo); same.born_hi = min(same.born_hi, f.born_hi)
                if same.claimed is None and f.claimed is not None: same.claimed = f.claimed
                if f.claimed is not None: self.minds[f.claimed].fruit = same.id if same.claimed == f.claimed else None
        for c, v in gb.cells.items():
            if c in ga.cells: ga.cells[c][0] = max(ga.cells[c][0], v[0]); ga.cells[c][1] = ga.cells[c][1] or v[1]
            else: ga.cells[c] = list(v)
        ga.anchored = ga.anchored or gb.anchored
        del self.groups[gb.id]
        self.metrics['merges'] += 1

    def _merge_groups(self, states):
        for aid, s in states.items():
            m = self.minds[aid]
            for o in s['observations']:
                if o['type'] != 'Agent' or o.get('id') not in self.minds: continue
                mb = self.minds[o['id']]
                if mb.group == m.group: continue
                ga, gb = self.groups[m.group], self.groups[mb.group]
                pb_est = self._pose_from_observer(m.pose, o); pB, thB = pb_est.p, pb_est.theta
                if gb.anchored and not ga.anchored:
                    # express ga in gb's frame instead: A's pose in gb frame from B's view of A is not
                    # available; invert the transform gb->ga
                    dth = wrap(thB-mb.pose.theta); shift = sub(pB, rot(mb.pose.p, dth))
                    inv_dth = -dth; inv_shift = mul(rot(shift, inv_dth), -1.)
                    self._merge(gb, ga, inv_dth, inv_shift)
                elif ga.anchored and gb.anchored:
                    self._merge(ga, gb, 0., (0., 0.))
                else:
                    dth = wrap(thB-mb.pose.theta); shift = sub(pB, rot(mb.pose.p, dth))
                    self._merge(ga, gb, dth, shift)
                return self._merge_groups(states)   # restart: groups changed

    # ------------------------------------------------------------ registration
    def _register(self, new_ids, states):
        spawners = [a for a in self.last_spawners if a in self.minds]
        for k, cid in enumerate(new_ids):
            parent = spawners[k] if k < len(spawners) else None
            pose = None
            if parent is not None:
                pm = self.minds[parent]
                o = next((o for o in states[parent]['observations'] if o['type'] == 'Agent' and o.get('id') == cid), None)
                if o is not None:
                    pose = self._pose_from_observer(pm.pose, o)
                else:
                    o = next((o for o in states[cid]['observations'] if o['type'] == 'Agent' and o.get('id') == parent), None)
                    if o is not None:
                        if o['distance'] < 1e-6:
                            pose = Pose(pm.pose.p, wrap(-o['angle']))
                        else:
                            th = wrap(pm.pose.theta-o['angle']-math.pi+o['rel_dir'])
                            pose = Pose(sub(pm.pose.p, mul(unit(th+o['angle']), o['distance'])), th)
            if pose is None:
                for aid, s in states.items():
                    if aid == cid or aid not in self.minds: continue
                    o = next((o for o in s['observations'] if o['type'] == 'Agent' and o.get('id') == cid), None)
                    if o is not None:
                        pm = self.minds[aid]; parent = aid
                        pose = self._pose_from_observer(pm.pose, o); break
            if pose is None:
                g = self._new_group(); pose = Pose((0., 0.), 0.)
            else:
                g = self.groups[self.minds[parent].group]
            m = Mind(aid=cid, group=g.id, pose=pose, born=self.time)
            m.energy_prev = states[cid]['energy']
            self.minds[cid] = m; g.agents.add(cid)

    @staticmethod
    def _pose_from_observer(op: Pose, o):
        """Pose of an observed agent from the observer's pose and its observation
        (distance, angle, rel_dir). At distance 0 the engine's bearing is atan2(0,0)=0,
        so rel_dir alone carries the heading."""
        if o['distance'] < 1e-6:
            return Pose(op.p, wrap(-o['rel_dir']))
        return Pose(op.polar(o), wrap(op.theta+o['angle']+math.pi-o['rel_dir']))

    # ------------------------------------------------------------ odometry
    def _odometry(self, m: Mind, s):
        act = m.last_action
        if act is None: return
        dist, direction, turn, biome, energy_before, speed, sprint, max_e = act
        d = max(0., min(dist, sprint))
        if energy_before < max_e/5 and d > speed: d = speed
        d *= MOVE_PENALTY.get(biome, 1.0)
        pose = m.pose
        ang = pose.theta+direction
        if d > 0:
            near = [(a, b) for a, b, t in m.edges if self.time-t < 4. and point_segment(pose.p, a, b) < d+8]
            def blocked(q):
                for a, b in near:
                    if point_segment(q, a, b) < 5.5 or segments_cross(pose.p, q, a, b): return True
                return False
            q = add(pose.p, mul(unit(ang), d))
            if near and blocked(q):
                step = math.pi/18; moved = False
                for i in range(36):
                    a = ang+step*((i+1)//2)*(-1)**i
                    q2 = add(pose.p, mul(unit(a), d))
                    if not blocked(q2): q = q2; moved = True; break
                if not moved: q = pose.p
                self.metrics['deflections'] += 1
            pose.p = q
        pose.theta = wrap(pose.theta+turn)
        g = self.groups[m.group]
        if g.anchored:
            pose.p = (min(max(pose.p[0], 5.), W-5.), min(max(pose.p[1], 5.), H-5.))

    # ------------------------------------------------------------ perception
    def _in_view(self, pose: Pose, hear, cone, vr, p, margin=0.):
        d, ang = pose.local(p)
        if d <= hear-margin-1: return True
        return d <= vr-margin-5 and abs(ang) <= cone/2-0.06

    def _anchor(self, m: Mind, o):
        """A boundary edge (length > 1000) gives the absolute pose exactly."""
        (x1, y1), (x2, y2) = o['coords']
        L = math.hypot(x2-x1, y2-y1)
        phi = math.atan2(y2-y1, x2-x1)
        if L > 1500:   # top/bottom wall, absolute direction +x
            theta = wrap(-phi)
            offset = rot((x1, y1), theta)
            cands = [(0., H-30.), (0., H)] if offset[1] > 0 else [(0., 30.), (0., 0.)]
        else:          # left/right wall, absolute direction +y
            theta = wrap(math.pi/2-phi)
            offset = rot((x1, y1), theta)
            cands = [(W-30., 0.), (W, 0.)] if offset[0] > 0 else [(30., 0.), (0., 0.)]
        best = None
        for sx, sy in cands:
            px, py = sub((sx, sy), rot((x1, y1), theta))
            if 35 <= px <= W-35 and 35 <= py <= H-35:
                best = (px, py); break
        if best is None: return
        g = self.groups[m.group]
        new = Pose(best, theta)
        if not g.anchored:
            dth = wrap(new.theta-m.pose.theta); shift = sub(new.p, rot(m.pose.p, dth))
            self._transform_group(g, dth, shift); g.anchored = True
            self.metrics['anchors'] += 1
        else:
            err = sub(new.p, m.pose.p)
            if norm(err) > 0.5:
                m.pose.p = new.p; self.metrics['pose_corrections'] += 1

    def _observe(self, m: Mind, s):
        g = self.groups[m.group]; pose = m.pose
        hear, cone, vr = s['hearing_radius'], s['vision_angle'], s['vision_range']
        obs = s['observations']
        moved = m.last_action is not None and m.last_action[0] > 0
        # visual odometry: static landmarks seen now and last tick must coincide; the median
        # shift is this tick's unpredicted displacement (engine deflections, boundary clamps)
        wf, wt = self.P['vo_win_fruit'], self.P['vo_win_far']
        marks = []   # (point, match window): fruit cluster tightly, trees and edge corners do not
        for o in obs:
            k = o['type']
            if k == 'Fruit': marks.append((pose.polar(o), wf))
            elif k == 'Tree': marks.append((pose.polar(o), wt))
            elif k == 'Edge':
                marks.append((pose.transform(tuple(o['coords'][0])), wt)); marks.append((pose.transform(tuple(o['coords'][1])), wt))
        if moved and marks and m.prev_marks:
            pairs = []
            B = 25.
            buckets = {}
            for idx, (r, _) in enumerate(m.prev_marks):
                buckets.setdefault((int(r[0]//B), int(r[1]//B)), []).append(idx)
            prev = m.prev_marks
            for q, win in marks:
                cx, cy = int(q[0]//B), int(q[1]//B)
                cand = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cand.extend(buckets.get((cx+dx, cy+dy), ()))
                cand.sort()
                best = None
                for idx in cand:
                    r = prev[idx][0]
                    dd = math.dist(q, r)
                    if dd < win and (best is None or dd < best[0]): best = (dd, r)
                if best is not None: pairs.append((best[1][0]-q[0], best[1][1]-q[1]))
            if len(pairs) >= 2:
                xs = sorted(a for a, b in pairs); ys = sorted(b for a, b in pairs)
                ex, ey = xs[len(xs)//2], ys[len(ys)//2]
                agree = sum(1 for a, b in pairs if abs(a-ex) < 0.8 and abs(b-ey) < 0.8)
                if 0.05 < math.hypot(ex, ey) < self.P['vo_cap'] and agree >= max(2, (len(pairs)+1)//2):
                    pose.p = add(pose.p, (ex, ey)); self.metrics['vo_corrections'] += 1
                    marks = [(add(q, (ex, ey)), w) for q, w in marks]
        m.prev_marks = marks
        # edges (own short memory) and anchoring
        for o in obs:
            if o['type'] != 'Edge': continue
            (x1, y1), (x2, y2) = o['coords']
            if math.hypot(x2-x1, y2-y1) > 1000: self._anchor(m, o)
        for o in obs:
            if o['type'] != 'Edge': continue
            a, b = map(pose.transform, map(tuple, o['coords']))
            ax, ay = a; bx, by = b
            for k, (ea, eb, t) in enumerate(m.edges):
                if abs(ea[0]-ax) < 6 and abs(ea[1]-ay) < 6 and abs(eb[0]-bx) < 6 and abs(eb[1]-by) < 6 \
                        and math.dist(a, ea) < 6 and math.dist(b, eb) < 6:
                    m.edges[k] = (a, b, self.time); break
            else:
                m.edges.append((a, b, self.time))
        m.edges = [e for e in m.edges if self.time-e[2] < 40.][-150:]
        # trees: association, pose correction against a well-known tree, new trees
        for o in obs:
            if o['type'] != 'Tree': continue
            p = pose.polar(o)
            t = min((t for t in g.near_trees(p, 12) if not t.dead), key=lambda t: math.dist(t.p, p), default=None)
            if t is None:
                fresh = m.prev_pose is not None and self._in_view(m.prev_pose, m.prev_hear, m.prev_cone, m.prev_vis, p, margin=3.)
                t = Tree(g.next_tree, p, self.time, self.time, fresh); g.add_tree(t); g.next_tree += 1
                self.metrics['trees_seen'] += 1
            else:
                err = sub(t.p, p)
                if moved and 0.3 < norm(err) < 8 and self.time-t.last < 2.:
                    pose.p = add(pose.p, err); self.metrics['pose_corrections'] += 1; moved = False
                elif norm(err) >= 1.0 and self.time-t.last >= 2.:
                    g.move_tree(t, p)
            t.last = self.time; g.seen_trees.add(t.id)
        # fruit
        for o in obs:
            if o['type'] != 'Fruit': continue
            p = pose.polar(o)
            f = min(g.near_fruits(p, 5), key=lambda f: math.dist(f.p, p), default=None)
            if f is None:
                lo = -INF
                if m.prev_pose is not None and self._in_view(m.prev_pose, m.prev_hear, m.prev_cone, m.prev_vis, p, margin=3.):
                    lo = self.time-0.1
                else:
                    # latest moment any group member's hearing disc (exact, unoccluded) covered the spot
                    for a in g.agents:
                        for ht, hp, hr in reversed(self.minds[a].hear_hist):
                            if math.dist(hp, p) < hr-2.:
                                lo = max(lo, ht); break
                lo = max(lo, 0.)   # nothing exists before the game starts
                f = Fruit(g.next_fruit, p, lo, self.time, self.time); g.add_fruit(f); g.next_fruit += 1
                self.metrics['fruits_seen'] += 1
                if self.time-lo <= self.P['wait_tol']: self.metrics['fresh_fruits'] += 1
                for t in g.near_trees(p, 70):
                    if not t.dead: t.fruit_seen = self.time
            f.last = self.time; g.seen_fruits.add(f.id)
        # coverage cells
        c0 = cell_of(pose.p)
        v = g.cells.get(c0)
        if v is None: g.cells[c0] = [self.time, s['biome']]
        else: v[0] = self.time; v[1] = s['biome']
        for r in (60., 120., 175.):
            if r > vr: break
            for a in (-cone/3, 0., cone/3):
                c = cell_of(add(pose.p, mul(unit(pose.theta+a), r)))
                v = g.cells.get(c)
                if v is None: g.cells[c] = [self.time, None]
                else: v[0] = self.time
        # old age from the energy ledger
        if m.energy_prev is not None and m.last_action is not None and not m.old:
            dist, direction, turn, biome, e0, speed, sprint, max_e = m.last_action
            d = max(0., min(dist, sprint))
            if e0 < max_e/5 and d > speed: d = speed
            cost = d*0.05 if d <= speed else speed*0.05+(d-speed)*0.5
            cost += min(math.pi, abs(turn))/TAU+0.1+(100. if m.spawned_ok else 0.)
            dev = (m.energy_prev-s['energy'])-cost
            if 0.45 < dev < 2.5 and s['energy'] < s['max_energy']-0.5 or s['age'] > 120.5:
                m.old = True; m.old_since = self.time; self.metrics['old_detected'] += 1
        m.energy_prev = s['energy']
        m.hear_hist.append((self.time, pose.p, hear))
        if len(m.hear_hist) > 400: del m.hear_hist[:100]
        if m.prev_marks:
            # express landmarks with the final pose of this tick (anchor/tree corrections may have moved it)
            m.prev_marks = []
            for o in obs:
                k = o['type']
                if k == 'Fruit': m.prev_marks.append((pose.polar(o), wf))
                elif k == 'Tree': m.prev_marks.append((pose.polar(o), wt))
                elif k == 'Edge':
                    m.prev_marks.append((pose.transform(tuple(o['coords'][0])), wt)); m.prev_marks.append((pose.transform(tuple(o['coords'][1])), wt))
        m.prev_pose = Pose(pose.p, pose.theta); m.prev_hear = hear; m.prev_cone = cone; m.prev_vis = vr

    def _maintain(self, g: Group, states):
        now = self.time
        # which remembered items would some agent perceive right now?
        vis_t, vis_f = set(), set()
        seen_t, seen_f = g.seen_trees, g.seen_fruits
        for a in g.agents:
            m = self.minds[a]; s = states[a]
            h, c, v = s['hearing_radius'], s['vision_angle'], s['vision_range']
            edges = None
            def occluded(p):
                nonlocal edges
                if math.dist(p, m.pose.p) <= h-3.: return False   # hearing works through walls
                if edges is None: edges = [(ea, eb) for ea, eb, et in m.edges if now-et < 40.]
                return any(segments_cross(m.pose.p, p, ea, eb) for ea, eb in edges)
            for t in g.near_trees(m.pose.p, v):
                if t.id in vis_t or t.id in seen_t or t.dead: continue    # only unseen live memories can be proven gone
                if self._in_view(m.pose, h, c, v, t.p, 20.) and not occluded(t.p): vis_t.add(t.id)
            for f in g.near_fruits(m.pose.p, v):
                if f.id in vis_f or f.id in seen_f: continue
                if self._in_view(m.pose, h, c, v, f.p, 8.) and not occluded(f.p): vis_f.add(f.id)
        for tid in list(g.trees):
            t = g.trees[tid]
            if t.dead:
                if now-t.last > 55.: g.del_tree(tid)
                continue
            if now > t.first+62.5 or (tid not in g.seen_trees and tid in vis_t):
                t.dead = True
                continue
            t.assigned = {a for a in t.assigned if a in self.minds and self.minds[a].post == tid}
        for fid in list(g.fruits):
            f = g.fruits[fid]
            gone = now > f.born_hi+50.05 or (fid not in g.seen_fruits and fid in vis_f)
            if gone:
                if f.claimed is not None and f.claimed in self.minds and self.minds[f.claimed].fruit == fid:
                    self.minds[f.claimed].fruit = None
                g.del_fruit(fid); continue
            if f.claimed is not None and (f.claimed not in self.minds or self.minds[f.claimed].fruit != fid): f.claimed = None
        for t in g.trees.values():
            near = g.near_fruits(t.p, 70.)
            t.fruit_here = len(near); t.fruit_free = sum(1 for f in near if f.claimed is None)
            t.cluster_t = -1.
        g.seen_trees.clear(); g.seen_fruits.clear()

    # ------------------------------------------------------------ economy
    def _n_est(self): return self.P['n0']*0.5**(self.time/self.P['tree_half'])
    def _cap(self):
        c = int(max(self.P['cap_min'], min(self.P['cap_max'], round(self.P['cap_mult']*self._n_est()))))
        if self.P['cap_tree_slack'] >= 0 and self.groups:
            # the population can only be fed by trees somebody knows about
            known = max(sum(1 for t in g.trees.values() if not t.dead) for g in self.groups.values())
            c = min(c, max(self.P['cap_hard_min'], known*self.P['tree_slots']+self.P['cap_tree_slack']))
        return c
    def _fitness(self, s):
        """Harvesting/watching quality from public traits: swept vision area (range and cone),
        hearing disc, energy capacity, walking speed."""
        return (self.P['fit_vision']*(s['vision_range']/200.)**2*min(1.5, s['vision_angle']/1.0472)
                + self.P['fit_hear']*(s['hearing_radius']/50.)**2
                + self.P['fit_energy']*min(2., s['max_energy']/500.)+self.P['fit_speed']*min(1.5, min(s['speed'], s['sprint_speed'])/10.))

    def _ready(self, f: Fruit, energy=INF, old=False):
        """Ripe by known age, or age unknown, or the agent cannot afford the remaining wait."""
        if self.time < f.born_hi+self.P['fruit_min_wait']: return False   # extreme rule: always wait after first sighting
        if f.born_lo == -INF: return True
        # eat when surely ripe (20 s after the latest possible birth) or just before it can rot (50 s after the earliest)
        t_eat = min(f.born_hi+self.P['ripen_wait'], f.born_lo+self.P['rot_margin'])
        left = t_eat-self.time
        if left <= 0.: return True
        if old: return False
        return energy < left+self.P['hungry_margin']

    def _reserve(self):
        P = self.P; t = self.time
        if t <= P['reserve_t0']: return P['breed_reserve']
        if t >= P['reserve_t1']: return P['breed_reserve_late']
        return P['breed_reserve']+(P['breed_reserve_late']-P['breed_reserve'])*(t-P['reserve_t0'])/(P['reserve_t1']-P['reserve_t0'])

    def _site_fruit(self, g: Group, t: Tree, aid):
        return t.fruit_free

    def _site_ok(self, g: Group, t: Tree, aid):
        return (not t.dead) or t.fruit_here > 0

    def _tree_value(self, g: Group, t: Tree, m: Mind, s):
        """Expected fruit energy at a site for this agent, minus travel, or -inf if unaffordable."""
        n = len(t.assigned-{m.aid})
        d = math.dist(t.p, m.pose.p)
        reach = self.P['tree_reach']*(self.P['lone_reach_mult'] if len(g.agents) <= 1 else 1.)
        if d > reach: return -INF
        walk = max(1., min(s['speed'], s['sprint_speed'])*MOVE_PENALTY.get(s['biome'], 1.))
        travel_t = d/walk/10.; travel_e = d*0.05+travel_t
        wait = 0.; future = 0.
        if not t.dead:
            if n >= self.P['tree_slots']: return -INF
            remaining = (t.first+58.-self.time) if t.fresh else (t.first+55.-self.time)
            remaining -= travel_t
            if remaining < 6.: return -INF
            wait = max(0., t.first+20.-self.time-travel_t) if t.fresh else 0.
            cell = g.cells.get(cell_of(t.p)); biome = cell[1] if cell else None
            rate = FRUIT_RATE.get(biome, 0.08)*60.
            known = self.time-t.first
            if not t.fresh and known > 15. and self.time-t.fruit_seen > known: rate *= 0.5   # watched, never fruited
            future = max(0., remaining-wait)*rate
        here = 55.*self._site_fruit(g, t, m.aid)
        if self.P['cluster_radius'] > 0.:
            # a site is worth every tree around it that the agent can also harvest from there
            if t.cluster_t != self.time:
                t.cluster = g.near_trees(t.p, self.P['cluster_radius']); t.cluster_t = self.time
            for u in t.cluster:
                if u.id == t.id or u.dead or len(u.assigned-{m.aid}) >= self.P['tree_slots']: continue
                ur = (u.first+58.-self.time) if u.fresh else (u.first+55.-self.time)
                if ur > 6.:
                    ucell = g.cells.get(cell_of(u.p)); ub = ucell[1] if ucell else None
                    future += 0.7*max(0., ur)*FRUIT_RATE.get(ub, 0.08)*60.
                here += 55.*u.fruit_free
        if future+here <= 0.: return -INF
        if s['energy']-travel_e-wait-12. < 0.: return -INF
        value = (future+here)/(n+1)-travel_e-0.5*wait-self.P['dist_pen']*d
        if self.P['nursery_bonus'] > 0. and not m.heir_done and s['age'] >= self.P['heir_age']-8.:
            # a parent about to leave an heir prefers a site where the child can eat at once
            value += self.P['nursery_bonus']*min(4, t.fruit_free)
        if self.P['spread_weight'] > 0.:
            # family coverage: prefer sites far from where the others already look
            others = [self.minds[a].pose.p for a in g.agents if a != m.aid]
            if others:
                gap = min(math.dist(t.p, q) for q in others)
                value += self.P['spread_weight']*60.*min(1., gap/max(1., s['vision_range']))
        return value

    def _assign_posts(self, g: Group, states, unavailable=()):
        sites = [t for t in g.trees.values()]
        agents = sorted(g.agents, key=lambda a: states[a]['energy'])
        for a in agents:
            if a in unavailable: continue
            m = self.minds[a]
            if m.old: continue
            keep = m.post is not None and m.post in g.trees and self._site_ok(g, g.trees[m.post], a)
            if keep and (self.time < m.repost_at or self.time-m.post_since < self.P['min_stay']): continue
            cur_v = -INF
            if keep:
                cur = g.trees[m.post]; cur_v = self._tree_value(g, cur, m, states[a])
                m.repost_at = self.time+self.P['repost_every']
            best = None
            for t in sites:
                if keep and t.id == m.post: continue
                if not self._site_ok(g, t, a): continue
                v = self._tree_value(g, t, m, states[a])
                if v > self.P['site_min'] and (best is None or v > best[0]): best = (v, t)
            if keep and (best is None or best[0] < cur_v+self.P['switch_gain']): continue
            if m.post is not None and m.post in g.trees: g.trees[m.post].assigned.discard(a)
            m.post = None
            if best is not None:
                t = best[1]; m.post = t.id; t.assigned.add(a); m.explore = None; m.target_key = None; m.post_since = self.time
                self.metrics['post_changes'] += 1; self.metrics['post_dist'] += math.dist(t.p, m.pose.p)

    def _assign_fruits(self, g: Group, states, unavailable=()):
        """Ready fruit goes to the hungriest agent that can reach it, then by distance."""
        for a in g.agents:
            m = self.minds[a]
            if m.fruit is not None and (m.fruit not in g.fruits or g.fruits[m.fruit].claimed != a): m.fruit = None
        pairs = []
        for a in g.agents:
            if a in unavailable: continue
            m = self.minds[a]
            if m.fruit is not None: continue
            s = states[a]
            if s['age'] > self.P['no_eat_age']: continue   # extreme rule: elders leave all fruit to the young
            full = s['energy'] > s['max_energy']-30.
            reach = self.P['old_reach'] if m.old else self.P['fruit_reach']*(self.P['lone_reach_mult'] if len(g.agents) <= 1 else 1.)
            for f in g.near_fruits(m.pose.p, reach):
                if f.claimed is not None: continue
                d = math.dist(f.p, m.pose.p)
                if not self._ready(f, s['energy'], m.old): continue
                owe_heir = (not m.heir_done) and s['age'] >= self.P['heir_age']-5. and s['energy'] < self.P['heir_reserve']+20.
                if m.old: bucket = 10 if self.P['old_eat_last'] else 5
                elif a in self.culled: bucket = 10
                elif owe_heir: bucket = 0
                elif full: bucket = 9
                elif self.P['feed_mode'] == 'breed':
                    # food is for reproduction: agents that a meal brings to breeding energy first, then the rest by hunger
                    bucket = 1 if s['energy'] < self._reserve()+20. else 2+int(s['energy']//120)
                else:
                    bucket = int(s['energy']//60)
                pairs.append((bucket, -self._fitness(s), d, a, f.id))
        pairs.sort()
        taken = set()
        for b, nf, d, a, fid in pairs:
            if a in taken or g.fruits[fid].claimed is not None: continue
            g.fruits[fid].claimed = a; self.minds[a].fruit = fid; taken.add(a)

    # ------------------------------------------------------------ navigation
    def _goto(self, m: Mind, s, target, stop=2.):
        pose = m.pose
        d, ang = pose.local(target)
        if d <= stop: return 0., 0., 0.
        walk = min(s['speed'], s['sprint_speed'])
        step = min(walk, max(0., d-stop*0.5))
        heading = pose.theta+ang
        look = min(45., d)
        recent = [(a, b) for a, b, t in m.edges if self.time-t < 25. and point_segment(pose.p, a, b) < look+10.]
        g = self.groups[m.group]
        unripe = [f.p for f in g.near_fruits(pose.p, look+20.) if not self._ready(f) and f.id != m.fruit]
        def clear(h, look):
            q = add(pose.p, mul(unit(h), look))
            for a, b in recent:
                if segments_cross(pose.p, q, a, b) or point_segment(q, a, b) < 7.: return False
            for fp in unripe:
                if point_segment(fp, pose.p, q) < 15.: return False
            return True
        if (recent or unripe) and not clear(heading, look):
            for k in range(1, 12):
                for sgn in (1, -1):
                    h = heading+sgn*k*math.pi/12
                    if clear(h, look): heading = h; break
                else: continue
                break
        direction = wrap(heading-pose.theta)
        tt = self.P['travel_turn']
        turn = max(-tt, min(tt, direction))
        return step, direction, turn

    def _progress(self, m: Mind, target, d):
        key = (round(target[0]/10), round(target[1]/10))
        if m.target_key != key:
            m.target_key = key; m.best_d = d; m.no_progress = 0; return False
        if d < m.best_d-1.5: m.best_d = d; m.no_progress = 0
        else: m.no_progress += 1
        if m.no_progress >= 15:
            self.metrics['stuck_events'] += 1
            m.blocked.append((target, self.time+40.))
            m.detour = (m.pose.theta+self.rng.choice([-1, 1])*math.pi/2+self.rng.uniform(-.4, .4), self.time+2.)
            m.target_key = None; m.no_progress = 0
            return True
        return False

    def _explore_target(self, m: Mind, g: Group, s):
        pose = m.pose; R = self.P['explore_radius']
        others = []
        for a in g.agents:
            if a == m.aid: continue
            om = self.minds[a]
            if om.post is not None and om.post in g.trees: others.append(g.trees[om.post].p)
            elif om.explore is not None: others.append(om.explore[0])
            else: others.append(om.pose.p)
        c0 = cell_of(pose.p); n = int(R//CELL)+1
        best = None
        for dx in range(-n, n+1):
            for dy in range(-n, n+1):
                c = (c0[0]+dx, c0[1]+dy)
                center = ((c[0]+.5)*CELL, (c[1]+.5)*CELL)
                if g.anchored and not (40 < center[0] < W-40 and 40 < center[1] < H-40): continue
                d = math.dist(center, pose.p)
                if d > R or d < 60: continue
                v = g.cells.get(c)
                stale = 1.2 if v is None else min(1., (self.time-v[0])/200.)
                biome = v[1] if v is not None else None
                w = TREE_RATE.get(biome, 0.6) if biome else 0.6
                score = stale*w-0.6*d/R
                if any(math.dist(center, q) < 130 for q in others): score -= 0.5
                if any(math.dist(center, b[0]) < 40 and b[1] > self.time for b in m.blocked): score -= 1.
                if any(segments_cross(pose.p, center, a, b) for a, b, t in m.edges if self.time-t < 30.): score -= 0.4
                score += self.rng.uniform(0, .08)
                if best is None or score > best[0]: best = (score, center)
        if best is None: return None
        target = add(best[1], (self.rng.uniform(-20, 20), self.rng.uniform(-20, 20)))
        walk = max(1., min(s['speed'], s['sprint_speed'])*MOVE_PENALTY.get(s['biome'], 1.))
        return (target, self.time+math.dist(target, pose.p)/walk/10.+10.)

    def _watch_post(self, m: Mind, g: Group, s):
        """Stationary lookout: an interior spot whose vision disc covers many tree-friendly cells
        that no other member already watches. Trees spawn uniformly per biome rate, so a wide
        sweep from such a spot finds new trees for 1 energy/s."""
        pose = m.pose; vr = min(s['vision_range'], 400.); R = self.P['watch_reach']
        others = []
        for a in g.agents:
            if a == m.aid: continue
            om = self.minds[a]
            if om.post is not None and om.post in g.trees: others.append(g.trees[om.post].p)
            elif om.watch is not None: others.append(om.watch[0])
            else: others.append(om.pose.p)
        c0 = cell_of(pose.p); n = int(R//CELL)+1; k = int(vr//CELL)+1
        best = None
        for dx in range(-n, n+1):
            for dy in range(-n, n+1):
                c = (c0[0]+dx, c0[1]+dy); center = ((c[0]+.5)*CELL, (c[1]+.5)*CELL)
                if g.anchored and not (60 < center[0] < W-60 and 60 < center[1] < H-60): continue
                d = math.dist(center, pose.p)
                if d > R: continue
                cover = 0.
                for ex in range(-k, k+1):
                    for ey in range(-k, k+1):
                        cc = (c[0]+ex, c[1]+ey); q = ((cc[0]+.5)*CELL, (cc[1]+.5)*CELL)
                        if g.anchored and not (30 < q[0] < W-30 and 30 < q[1] < H-30): continue
                        if math.dist(q, center) > vr: continue
                        v = g.cells.get(cc); w = TREE_RATE.get(v[1], 0.5) if (v and v[1]) else 0.5
                        if any(math.dist(q, o) < vr*0.8 for o in others): w *= 0.3
                        cover += w
                score = cover-d*0.02+self.rng.uniform(0, .2)
                if best is None or score > best[0]: best = (score, center)
        return None if best is None else best[1]

    # ------------------------------------------------------------ main
    def __call__(self, states_list, sim_time, *, unavailable_agents=()):
        self.time = sim_time
        states = {s['agent_id']: s for s in states_list}
        unavailable = set(unavailable_agents).intersection(states)
        for aid in list(self.minds):
            if aid in states: continue
            m = self.minds.pop(aid); g = self.groups[m.group]; g.agents.discard(aid)
            for t in g.trees.values(): t.assigned.discard(aid)
            for f in g.fruits.values():
                if f.claimed == aid: f.claimed = None
            if not g.agents: del self.groups[g.id]
        for aid, s in states.items():
            if aid in self.minds: self._odometry(self.minds[aid], s)
        new = sorted(a for a in states if a not in self.minds)
        if new: self._register(new, states)
        self._merge_groups(states)
        for aid, s in states.items(): self._observe(self.minds[aid], s)
        for g in list(self.groups.values()): self._maintain(g, states)
        # External bait/guide roles still share observations, but cannot harvest
        # their assigned posts or fruit. Return those claims to the workforce.
        for aid in unavailable:
            m = self.minds[aid]; g = self.groups[m.group]
            if m.post in g.trees: g.trees[m.post].assigned.discard(aid)
            if m.fruit in g.fruits and g.fruits[m.fruit].claimed == aid:
                g.fruits[m.fruit].claimed = None
            m.post = m.fruit = None
        # surplus young agents above the cap (lowest fitness first) only get leftover fruit and no heirs
        self.culled = set()
        if self.P['cull']:
            young_ids = [a for a, m in self.minds.items() if not m.old and a not in unavailable]
            surplus = len(young_ids)-self._cap()
            if surplus > 0:
                ranked = sorted(young_ids, key=lambda a: (self._fitness(states[a]), states[a]['energy']))
                self.culled = set(ranked[:surplus])
        for g in self.groups.values():
            self._assign_posts(g, states, unavailable); self._assign_fruits(g, states, unavailable)
        # ---- reproduction plan (global) ----
        young = [a for a, m in self.minds.items() if not m.old and a not in unavailable]
        cap = self._cap(); pop = len(states)-len(unavailable)
        births_allowed = max(0, cap-len(young))
        spawn_set = set()
        def cost_now(m, dist, turn, s):
            d = max(0., min(dist, s['sprint_speed']))
            if s['energy'] < s['max_energy']/5 and d > s['speed']: d = s['speed']
            c = d*0.05 if d <= s['speed'] else s['speed']*0.05+(d-s['speed'])*0.5
            return c+min(math.pi, abs(turn))/TAU
        # decisions first (movement), then births are decided with known costs
        plans = {}
        for aid, s in states.items():
            plans[aid] = (0., 0., 0., 'external') if aid in unavailable else self._act(self.minds[aid], s, states)
        # 1. heirs: every senescent agent leaves one replacement (its energy is lost otherwise)
        young_now = len(young)
        fit = {aid: self._fitness(s) for aid, s in states.items()}
        elders = sorted((aid for aid, m in self.minds.items() if aid not in unavailable and (m.old or states[aid]['age'] >= self.P['heir_age'])),
                        key=lambda a: -states[a]['energy'])
        yfit = sorted(fit[a] for a in young) or [0.]
        median_fit = yfit[len(yfit)//2]
        for aid in elders:
            m = self.minds[aid]; s = states[aid]
            if m.heir_done or s['biome'] == 'river' or (aid in self.culled and not m.old): continue
            if self.P['heir_select'] and len(young) >= self.P['select_min_young'] and fit[aid] < median_fit-self.P['heir_slack']:
                continue   # weak lineage while the colony is large: slot goes to a fitter parent
            dist, direction, turn, mode = plans[aid]
            g = self.groups[m.group]
            at_food = m.post is not None and m.post in g.trees and (g.trees[m.post].fruit_here > 0 or not g.trees[m.post].dead)
            left = s['energy']-cost_now(m, dist, turn, s)
            if m.old:
                ok = left > 101.
            else:
                ok = left > self.P['heir_reserve'] or (self.P['heir_at_food'] and at_food and left > 130.)
            if ok and self.P['heir_needs_site'] and not at_food and young_now >= cap and pop > 2:
                ok = False   # no tree to inherit and the colony is already at its food-limited size
            if ok:
                spawn_set.add(aid); m.heir_done = True; self.metrics['old_births' if m.old else 'heir_births'] += 1; young_now += 1
        # 1b. rich senescent agents standing at a fruiting site add extra children (their energy is lost otherwise)
        for aid in elders:
            m = self.minds[aid]; s = states[aid]
            if not m.old or aid in spawn_set: continue
            g = self.groups[m.group]
            site = g.trees.get(m.post) if m.post is not None else None
            late = self.time >= self.P['dump_after_t']   # late game: leftover energy buys extra watchers anywhere
            if not late and (site is None or site.fruit_here < self.P['dump_food_site']): continue
            dist, direction, turn, mode = plans[aid]
            if s['energy']-cost_now(m, dist, turn, s) > 101. and young_now < cap*self.P['dump_mult']:
                spawn_set.add(aid); self.metrics['old_births'] += 1; young_now += 1
        # 2. emergency: tiny population
        if pop <= 2:
            for aid, s in states.items():
                if aid in unavailable: continue
                m = self.minds[aid]
                dist, direction, turn, mode = plans[aid]
                if aid not in spawn_set and s['energy']-cost_now(m, dist, turn, s) > self.P['emergency_reserve']:
                    spawn_set.add(aid); self.metrics['emergency_births'] += 1; young_now += 1
        # 3. open lineage slots: fittest eligible parents first (old parents with spare energy count too)
        slots = min(cap-young_now, self.P['births_per_tick'])
        if slots > 0:
            cands = []
            for aid, s in states.items():
                if aid in unavailable: continue
                m = self.minds[aid]
                if aid in spawn_set or s['biome'] == 'river': continue
                dist, direction, turn, mode = plans[aid]
                left = s['energy']-cost_now(m, dist, turn, s)
                if m.old:
                    if not self.P['extra_old'] or left <= 101.: continue
                elif left <= (min(self._reserve(), self.P['low_pop_reserve']) if len(young) < self.P['cap_min'] else self._reserve()): continue
                g = self.groups[m.group]
                food = (m.post is not None and m.post in g.trees and not g.trees[m.post].dead) or bool(g.near_fruits(m.pose.p, 90.))
                if not food: continue
                cands.append((fit[aid], left, aid))
            cands.sort(reverse=True)
            for f_, e, aid in cands[:slots]:
                spawn_set.add(aid); self.metrics['births' if not self.minds[aid].old else 'old_births'] += 1
        actions = []; self.last_spawners = []
        for aid in sorted(states):
            s = states[aid]; m = self.minds[aid]
            dist, direction, turn, mode = plans[aid]
            spawn = aid in spawn_set
            ok = spawn and s['energy']-cost_now(m, dist, turn, s) > 100.
            m.spawned_ok = ok
            if ok: self.last_spawners.append(aid)
            m.last_action = (dist, direction, turn, s['biome'], s['energy'], s['speed'], s['sprint_speed'], s['max_energy'])
            self.decisions[aid] = (mode, round(dist, 1), m.group, m.post, m.fruit, 'old' if m.old else '')
            actions.append((aid, ActionRequest(agent_id=aid, move_distance=float(dist), move_direction=float(direction),
                                               turn_angle=float(turn), spawn_agent=bool(spawn))))
        return actions

    # ------------------------------------------------------------ per-agent behaviour
    def _act(self, m: Mind, s, states):
        g = self.groups[m.group]; pose = m.pose
        walk = min(s['speed'], s['sprint_speed'])
        m.blocked = [b for b in m.blocked if b[1] > self.time]
        # detour after a stuck event
        if m.detour[0] is not None and self.time <= m.detour[1]:
            direction = wrap(m.detour[0]-pose.theta)
            return walk, direction, max(-.3, min(.3, direction)), 'detour'
        # 1. claimed fruit
        if m.fruit is not None and m.fruit in g.fruits:
            f = g.fruits[m.fruit]
            d, ang = pose.local(f.p)
            self.metrics['fruit_ticks'] += 1
            if self._progress(m, f.p, d):
                f.claimed = None; m.fruit = None
            else:
                dist, direction, turn = self._goto(m, s, f.p, stop=0.)
                return min(walk, d+1.), direction, turn, 'fruit'
        # extreme rule: late in the game, never travel unless a ripe fruit is claimed
        if self.time >= self.P['late_still_t']:
            self.metrics['still_ticks'] = self.metrics.get('still_ticks', 0)+1
            return 0., 0., self.P['sweep_rate'], 'still'
        # 2. post
        if m.post is not None and m.post in g.trees and self._site_ok(g, g.trees[m.post], m.aid):
            t = g.trees[m.post]
            m.last_site = self.time
            d, ang = pose.local(t.p)
            if d > self.P['post_radius']:
                self.metrics['travel_ticks'] += 1
                if self._progress(m, t.p, d):
                    t.assigned.discard(m.aid); m.post = None
                else:
                    dist, direction, turn = self._goto(m, s, t.p, stop=self.P['post_radius']-8.)
                    return dist, direction, turn, 'topost'
            m.target_key = None
            # keep unripe fruit outside the automatic-collection radius (no walk back to the centre)
            close = [f for f in g.near_fruits(pose.p, 16.) if not self._ready(f)]
            if close:
                f = min(close, key=lambda f: math.dist(f.p, pose.p))
                away = math.atan2(pose.p[1]-f.p[1], pose.p[0]-f.p[0])
                direction = wrap(away-pose.theta)
                return min(walk, 18.-math.dist(f.p, pose.p)+1.), direction, 0., 'stepback'
            self.metrics['idle_ticks'] += 1
            turn = self.P['sweep_rate']*m.sweep_sign if self.P['idle_sweep'] else 0.
            return 0., 0., turn, 'post'
        if m.old:
            self.metrics['idle_ticks'] += 1
            return 0., 0., self.P['sweep_rate'], 'oldidle'
        # 3. watch first: with wide vision a stationary sweep covers a large area for 1 energy/s;
        #    explore on foot only when rich and nothing has turned up for a while
        if s['energy'] < self.P['explore_energy'] or self.time-m.last_site < self.P['watch_patience']:
            self.metrics['starve_ticks'] += 1
            m.explore = None
            if m.watch is None or self.time-m.watch[1] > self.P['watch_refresh']:
                tgt = self._watch_post(m, g, s)
                m.watch = (tgt, self.time) if tgt is not None else None
                m.target_key = None
            if m.watch is not None and s['energy'] >= self.P['explore_min']:
                d, ang = pose.local(m.watch[0])
                if d > 25.:
                    if self._progress(m, m.watch[0], d): m.watch = None
                    else:
                        dist, direction, turn = self._goto(m, s, m.watch[0], stop=20.)
                        return dist, direction, turn, 'towatch'
            return 0., 0., self.P['sweep_rate']*3., 'watch'
        self.metrics['explore_ticks'] += 1
        if m.sweep_left > 0:
            m.sweep_left -= 1
            return 0., 0., 0.32, 'sweep'
        if m.explore is None or self.time > m.explore[1] or math.dist(m.explore[0], pose.p) < 25.:
            arrived = m.explore is not None and math.dist(m.explore[0], pose.p) < 25.
            m.explore = self._explore_target(m, g, s)
            m.target_key = None
            if arrived:
                m.sweep_left = 20
                return 0., 0., 0.32, 'sweep'
        if m.explore is None:
            return 0., 0., 0.32, 'sweep'
        target = m.explore[0]
        d, ang = pose.local(target)
        if self._progress(m, target, d):
            m.explore = None
            return 0., 0., 0., 'stuck'
        dist, direction, turn = self._goto(m, s, target, stop=20.)
        return dist, direction, turn, 'explore'

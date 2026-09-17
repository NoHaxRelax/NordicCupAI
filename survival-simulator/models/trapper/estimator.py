"""WorldState from ordinary observations only (no engine access).

What the observations give us, and how each becomes global knowledge:

* Own action + own biome -> dead reckoning. The engine moves ``distance x
  biome_modifier`` of the *starting* pixel, then turns; both are known exactly,
  so odometry is exact unless a collision deflects us (we replicate the
  engine's deflection against known rectangles).
* Edge observations are complete axis-aligned segments (both endpoints, in our
  body frame). One edge fixes our heading modulo 90 degrees (resolved with the
  odometry prior); an edge whose global position is known fixes our position.
  Arena boundary edges are 1600/1200 long and known a priori, so any agent that
  sees a boundary is absolutely localized at once.
* Agent observations (distance, angle, rel_dir, id) give the exact relative pose
  of the other agent, so localization spreads through the colony and separate
  frames merge.
* Two adjacent perpendicular edges determine a full rectangle; rectangles feed
  site detection and path planning.
* Predator observations give position and, through rel_dir, heading. They are
  one predator move stale; we advance them one step with the exact model.

Frames: every agent starts in its own frame (pose 0,0,0). Frames merge on
mutual sight. A frame becomes absolute when any member sees a boundary edge;
until then trap logic can still run in the frame (sites found from rectangles
in that frame), but planning across frames is not attempted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Rect, add, sub, mul, rot, wrap, dist, heading_of, point_segment
from .world import AgentView, FoodView, PredatorView, WorldState, MOVE_PENALTY, PREDATOR_RADIUS
from . import predator_model as pm

ARENA = (1600.0, 1200.0)
BOUNDARY = 30.0


@dataclass
class Frame:
    fid: int
    absolute: bool = False
    members: set = field(default_factory=set)
    edges: list = field(default_factory=list)       # ((ax, ay), (bx, by)) in frame coords
    rects: list = field(default_factory=list)       # Rect in frame coords
    fruits: list = field(default_factory=list)      # [x, y, first_seen, last_seen]
    trees: list = field(default_factory=list)
    tracks: list = field(default_factory=list)      # predator tracks
    biome: dict = field(default_factory=dict)       # (cx, cy) -> modifier
    edge_index: dict = field(default_factory=dict)  # (horizontal, round(length)) -> [edge indices]
    new_edges: list = field(default_factory=list)   # edges added since the last rectangle pass


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def to_frame(self, local):
        return add((self.x, self.y), rot(local, self.theta))


@dataclass
class Track:
    tid: int
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    last_seen: float = 0.0
    still: int = 0
    seen: int = 1


def _boundary_rects(w, h):
    return [Rect(0, 0, w, BOUNDARY), Rect(0, h - BOUNDARY, w, BOUNDARY), Rect(0, 0, BOUNDARY, h), Rect(w - BOUNDARY, 0, BOUNDARY, h)]


class EstimatedWorld:
    def __init__(self, arena=ARENA):
        self.width, self.height = arena
        self.poses: dict[int, Pose] = {}
        self.frame_of: dict[int, Frame] = {}
        self.frames: dict[int, Frame] = {}
        self._next_frame = 0
        self._next_track = 0
        self.last_actions: dict = {}         # aid -> ActionRequest applied last tick
        self.last_states: dict = {}          # aid -> previous DTO (biome, energy, traits)
        self.time = 0.0
        self.state: WorldState | None = None
        self.metrics = dict(edge_fixes=0, absolute_fixes=0, merges=0, rects=0, deflections=0)
        self.absolute_frame: Frame | None = None

    # ------------------------------------------------------------------ frames
    def _new_frame(self, aid):
        f = Frame(fid=self._next_frame)
        self._next_frame += 1
        f.members.add(aid)
        self.frames[f.fid] = f
        self.frame_of[aid] = f
        return f

    def _make_absolute(self, frame: Frame, shift, dtheta):
        """Re-express a frame in absolute coordinates: p' = shift + rot(p, dtheta)."""
        T = lambda q: add(shift, rot(q, dtheta))
        for aid in frame.members:
            po = self.poses[aid]
            nx, ny = T((po.x, po.y))
            po.x, po.y, po.theta = nx, ny, wrap(po.theta + dtheta)
        frame.edges = [(T(a), T(b)) for a, b in frame.edges]
        frame.edge_index = {}
        for idx, (a, b) in enumerate(frame.edges):
            frame.edge_index.setdefault(self._edge_key(a, b), []).append(idx)
        frame.rects = [self._rect_from_corners(T((r.x, r.y)), T((r.x2, r.y2))) for r in frame.rects]
        for lst in (frame.fruits, frame.trees):
            for it in lst:
                it[0], it[1] = T((it[0], it[1]))
        for tr in frame.tracks:
            tr.x, tr.y = T((tr.x, tr.y))
            tr.heading = wrap(tr.heading + dtheta)
            tr.vx, tr.vy = rot((tr.vx, tr.vy), dtheta)
        frame.biome = {}
        frame.absolute = True

    @staticmethod
    def _rect_from_corners(a, b):
        x1, x2 = sorted((a[0], b[0]))
        y1, y2 = sorted((a[1], b[1]))
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def _merge(self, dst: Frame, src: Frame, shift, dtheta):
        """Move everything of src into dst: p_dst = shift + rot(p_src, dtheta)."""
        if dst is src:
            return
        T = lambda q: add(shift, rot(q, dtheta))
        for aid in src.members:
            po = self.poses[aid]
            nx, ny = T((po.x, po.y))
            po.x, po.y, po.theta = nx, ny, wrap(po.theta + dtheta)
            self.frame_of[aid] = dst
            dst.members.add(aid)
        for a, b in src.edges:
            self._add_edge(dst, T(a), T(b))
        for r in src.rects:
            self._add_rect(dst, self._rect_from_corners(T((r.x, r.y)), T((r.x2, r.y2))))
        for it in src.fruits:
            self._add_food(dst.fruits, T((it[0], it[1])), it[2], it[3], radius=4.0)
        for it in src.trees:
            self._add_food(dst.trees, T((it[0], it[1])), it[2], it[3], radius=12.0)
        for tr in src.tracks:
            tr.x, tr.y = T((tr.x, tr.y)); tr.heading = wrap(tr.heading + dtheta); tr.vx, tr.vy = rot((tr.vx, tr.vy), dtheta)
            if not any(dist((tr.x, tr.y), (u.x, u.y)) < 25 for u in dst.tracks):
                dst.tracks.append(tr)
        del self.frames[src.fid]
        self.metrics['merges'] += 1

    # ------------------------------------------------------------------ map
    @staticmethod
    def _edge_key(a, b):
        return (abs(a[1] - b[1]) < 0.5, round(dist(a, b)))

    @staticmethod
    def _add_edge(frame: Frame, a, b):
        key = EstimatedWorld._edge_key(a, b)
        for idx in frame.edge_index.get(key, ()):
            c, d = frame.edges[idx]
            if (dist(a, c) < 0.5 and dist(b, d) < 0.5) or (dist(a, d) < 0.5 and dist(b, c) < 0.5):
                return
        frame.edges.append((a, b))
        frame.edge_index.setdefault(key, []).append(len(frame.edges) - 1)
        frame.new_edges.append(len(frame.edges) - 1)

    def _add_rect(self, frame: Frame, r: Rect):
        for k, o in enumerate(frame.rects):
            if abs(o.x - r.x) < 1.0 and abs(o.y - r.y) < 1.0 and abs(o.w - r.w) < 1.0 and abs(o.h - r.h) < 1.0:
                return
        frame.rects.append(r)
        self.metrics['rects'] += 1

    @staticmethod
    def _add_food(lst, p, first, last, radius):
        for it in lst:
            if dist((it[0], it[1]), p) < radius:
                it[3] = max(it[3], last)
                return
        lst.append([p[0], p[1], first, last])

    def _infer_rects(self, frame: Frame):
        """Two perpendicular edges sharing a corner determine a rectangle. Only edges
        added since the last pass are paired against the rest."""
        if not frame.new_edges:
            return
        edges = frame.edges
        new = frame.new_edges
        frame.new_edges = []
        for i in new:
            a, b = edges[i]
            horiz_i = abs(a[1] - b[1]) < 0.5
            for key, idxs in frame.edge_index.items():
                if key[0] == horiz_i:
                    continue
                for j in idxs:
                    if j == i:
                        continue
                    c, d = edges[j]
                    h, v = ((a, b), (c, d)) if horiz_i else ((c, d), (a, b))
                    for corner in (h[0], h[1]):
                        for vcorner in (v[0], v[1]):
                            if abs(corner[0] - vcorner[0]) < 0.5 and abs(corner[1] - vcorner[1]) < 0.5:
                                far_h = h[1] if corner is h[0] else h[0]
                                far_v = v[1] if vcorner is v[0] else v[0]
                                r = self._rect_from_corners(far_h, far_v)
                                if r.w >= 20 and r.h >= 20:
                                    self._add_rect(frame, r)

    # ------------------------------------------------------------------ update
    def note_actions(self, actions):
        self.last_actions = {aid: act for aid, act in actions}

    def update(self, states, sim_time) -> WorldState:
        self.time = sim_time
        by_id = {s['agent_id']: s for s in states}
        # forget the dead
        for aid in list(self.poses):
            if aid not in by_id:
                f = self.frame_of.pop(aid)
                f.members.discard(aid)
                self.poses.pop(aid, None)
                self.last_states.pop(aid, None)
                if not f.members and not f.absolute:
                    self.frames.pop(f.fid, None)
        # 1. odometry
        for aid, s in by_id.items():
            if aid not in self.poses:
                self.poses[aid] = Pose()
                self._new_frame(aid)
                continue
            self._odometry(aid, s)
        # 2. edges: heading + position fixes, map growth
        for aid, s in by_id.items():
            self._observe_edges(aid, s)
        # 3. agent sightings: merge frames / register newborns
        for aid, s in by_id.items():
            self._observe_agents(aid, s, by_id)
        # 4. entities
        for aid, s in by_id.items():
            self._observe_entities(aid, s)
        for f in self.frames.values():
            self._infer_rects(f)
        self.last_states = {aid: s for aid, s in by_id.items()}
        return self._build(by_id)

    def _odometry(self, aid, s):
        act = self.last_actions.get(aid)
        prev = self.last_states.get(aid)
        po = self.poses[aid]
        if act is None or prev is None:
            return
        cap = prev['sprint_speed'] if prev['energy'] >= prev['max_energy'] / 5 else min(prev['speed'], prev['sprint_speed'])
        d = max(0.0, min(float(act.move_distance), cap))
        d *= MOVE_PENALTY.get(prev['biome'], 1.0)
        ang = po.theta + float(act.move_direction)
        if d > 0:
            frame = self.frame_of[aid]
            nx, ny = po.x + d * math.cos(ang), po.y + d * math.sin(ang)
            rects = frame.rects + (_boundary_rects(self.width, self.height) if frame.absolute else [])
            if pm.in_obstacle((nx, ny), 5.0, rects):
                step = math.pi / 18
                placed = False
                for i in range(36):
                    test = ang + step * ((i + 1) // 2) * (-1) ** i
                    tx, ty = po.x + d * math.cos(test), po.y + d * math.sin(test)
                    if not pm.in_obstacle((tx, ty), 5.0, rects):
                        nx, ny = tx, ty
                        placed = True
                        break
                if not placed:
                    nx, ny = po.x, po.y
                self.metrics['deflections'] += 1
            po.x, po.y = nx, ny
            if frame.absolute:
                po.x = max(5.0, min(self.width - 5.0, po.x))
                po.y = max(5.0, min(self.height - 5.0, po.y))
        po.theta = wrap(po.theta + float(act.turn_angle))

    def _observe_edges(self, aid, s):
        """Engine edges are ordered +x or +y (top/bottom edges run left to right, left/right
        edges top to bottom), so an observed edge fixes our heading up to a two-way choice
        resolved by the prior, and an arena boundary edge (1600 or 1200 long) fixes heading
        and position exactly."""
        po = self.poses[aid]
        frame = self.frame_of[aid]
        raw = []
        seen = set()
        for o in s['observations']:
            if o['type'] != 'Edge':
                continue
            (ax, ay), (bx, by) = o['coords']
            key = (round(ax, 3), round(ay, 3), round(bx, 3), round(by, 3))
            if key in seen:
                continue
            seen.add(key)
            raw.append(((ax, ay), (bx, by)))
        if not raw:
            return
        W, H = self.width, self.height
        fixed = False
        # 1. an arena boundary edge: exact heading and absolute position
        for a, b in raw:
            length = dist(a, b)
            is_w = abs(length - W) < 0.5
            is_h = abs(length - H) < 0.5
            if not (is_w or is_h):
                continue
            local_dir = heading_of(sub(b, a))
            true_dir = 0.0 if is_w else math.pi / 2
            theta = wrap(true_dir - local_dir)
            ra = rot(a, theta)                      # edge start relative to us, world orientation
            if is_w:
                y_edge = BOUNDARY if ra[1] < 0 else H - BOUNDARY
                new_p = (0.0 - ra[0], y_edge - ra[1])
            else:
                x_edge = BOUNDARY if ra[0] < 0 else W - BOUNDARY
                new_p = (x_edge - ra[0], 0.0 - ra[1])
            dtheta = wrap(theta - po.theta)
            shift = sub(new_p, rot((po.x, po.y), dtheta))
            if not frame.absolute:
                self._make_absolute(frame, shift, dtheta)
                self.metrics['absolute_fixes'] += 1
                if self.absolute_frame is not None and self.absolute_frame is not frame and self.absolute_frame.fid in self.frames:
                    self._merge(self.absolute_frame, frame, (0.0, 0.0), 0.0)
                    frame = self.frame_of[aid]
                else:
                    self.absolute_frame = frame
            else:
                if dist(new_p, (po.x, po.y)) > 0.01 or abs(dtheta) > 1e-6:
                    self.metrics['edge_fixes'] += 1
                po.x, po.y, po.theta = new_p[0], new_p[1], theta
            fixed = True
            break
        if not fixed:
            # 2. heading from any edge: theta is -local_dir (edge runs +x) or pi/2-local_dir (+y)
            corrections = []
            for a, b in raw:
                local_dir = heading_of(sub(b, a))
                cands = (wrap(-local_dir), wrap(math.pi / 2 - local_dir))
                best = min(cands, key=lambda t: abs(wrap(t - po.theta)))
                corrections.append(wrap(best - po.theta))
            corr = sorted(corrections)[len(corrections) // 2]
            if abs(corr) > 1e-9:
                po.theta = wrap(po.theta + corr)
            # 3. position from a known interior edge (match by orientation, length, proximity)
            edges_w = [(po.to_frame(a), po.to_frame(b)) for a, b in raw]
            best = None
            for a, b in edges_w:
                length = dist(a, b)
                horiz = abs(a[1] - b[1]) < 0.5
                cands = []
                for L in (round(length) - 1, round(length), round(length) + 1):
                    cands += frame.edge_index.get((horiz, L), [])
                for idx in cands:
                    c, d = frame.edges[idx]
                    if abs(dist(c, d) - length) > 0.05:
                        continue
                    shift = sub(c, a)
                    if dist(add(b, shift), d) < 0.05 and dist(shift, (0, 0)) < 40.0:
                        if best is None or dist(shift, (0, 0)) < dist(best, (0, 0)):
                            best = shift
            if best is not None and dist(best, (0, 0)) > 1e-6:
                po.x += best[0]; po.y += best[1]
                self.metrics['edge_fixes'] += 1
        # 4. remember the edges (skip boundaries in absolute frames: known a priori)
        for a, b in raw:
            aw, bw = po.to_frame(a), po.to_frame(b)
            length = dist(aw, bw)
            if frame.absolute and (abs(length - W) < 0.5 or abs(length - H) < 0.5):
                continue
            self._add_edge(frame, aw, bw)

    def _observe_agents(self, aid, s, by_id):
        po = self.poses[aid]
        fa = self.frame_of[aid]
        for o in s['observations']:
            if o['type'] != 'Agent' or o.get('id') not in self.poses or o['distance'] < 1e-6:
                continue
            other = o['id']
            fb = self.frame_of[other]
            pb = self.poses[other]
            # other's pose in our frame
            p_other = po.to_frame((o['distance'] * math.cos(o['angle']), o['distance'] * math.sin(o['angle'])))
            th_other = wrap(heading_of(sub((po.x, po.y), p_other)) - o['rel_dir'])
            if fb is fa:
                # same frame: correct the other's estimate toward ours if it is a newborn/unfixed (small drift)
                err = dist((pb.x, pb.y), p_other)
                if err > 0.5 and err < 40:
                    pb.x, pb.y, pb.theta = p_other[0], p_other[1], th_other
                continue
            # different frames: bring fb into fa (or fa into fb when fb is absolute)
            dtheta = wrap(th_other - pb.theta)
            shift = sub(p_other, rot((pb.x, pb.y), dtheta))
            if fb.absolute and not fa.absolute:
                inv_theta = -dtheta
                inv_shift = mul(rot(shift, inv_theta), -1.0)
                self._merge(fb, fa, inv_shift, inv_theta)
            else:
                self._merge(fa, fb, shift, dtheta)
                if fb.absolute:
                    fa.absolute = True
                    self.absolute_frame = fa
        return

    def _observe_entities(self, aid, s):
        po = self.poses[aid]
        frame = self.frame_of[aid]
        for tr in frame.tracks:
            tr._seen_now = False
        for o in s['observations']:
            t = o['type']
            if t in ('Fruit', 'Tree'):
                p = po.to_frame((o['distance'] * math.cos(o['angle']), o['distance'] * math.sin(o['angle'])))
                self._add_food(frame.fruits if t == 'Fruit' else frame.trees, p, self.time, self.time, 4.0 if t == 'Fruit' else 12.0)
            elif t == 'Predator':
                p = po.to_frame((o['distance'] * math.cos(o['angle']), o['distance'] * math.sin(o['angle'])))
                heading = wrap(heading_of(sub((po.x, po.y), p)) - o['rel_dir'])
                tr = min((u for u in frame.tracks if dist((u.x, u.y), p) < 45), key=lambda u: dist((u.x, u.y), p), default=None)
                if tr is None:
                    tr = Track(self._next_track, p[0], p[1], heading, last_seen=self.time)
                    self._next_track += 1
                    frame.tracks.append(tr)
                else:
                    dt = self.time - tr.last_seen
                    if 0 < dt <= 0.15:
                        vx, vy = (p[0] - tr.x) / (dt * 10), (p[1] - tr.y) / (dt * 10)
                        tr.still = tr.still + 1 if math.hypot(vx, vy) < 0.5 else 0
                        tr.vx, tr.vy = vx, vy
                    tr.x, tr.y, tr.heading, tr.last_seen = p[0], p[1], heading, self.time
                    tr.seen += 1
                tr._seen_now = True
        frame.tracks = [tr for tr in frame.tracks if self.time - tr.last_seen < 8.0]
        # fruit that should be audible but is not: gone
        hearing = s['hearing_radius']
        heard = [po.to_frame((o['distance'] * math.cos(o['angle']), o['distance'] * math.sin(o['angle']))) for o in s['observations'] if o['type'] == 'Fruit']
        frame.fruits = [f for f in frame.fruits if not (dist((f[0], f[1]), (po.x, po.y)) < hearing - 3 and not any(dist((f[0], f[1]), q) < 4 for q in heard)) and self.time - f[3] < 50]
        frame.trees = [f for f in frame.trees if self.time - f[3] < 150]
        # own biome paints a coarse grid for path costs
        frame.biome[(int(po.x // 50), int(po.y // 50))] = MOVE_PENALTY.get(s['biome'], 1.0)

    # ------------------------------------------------------------------ output
    def _build(self, by_id) -> WorldState:
        frame = self.absolute_frame
        if frame is None or not frame.members:
            # largest frame as the working frame
            frame = max(self.frames.values(), key=lambda f: len(f.members), default=None)
        agents = {}
        predators = []
        rects = []
        fruits = trees = []
        if frame is not None:
            for aid in frame.members:
                s = by_id.get(aid)
                po = self.poses.get(aid)
                if s is None or po is None:
                    continue
                agents[aid] = AgentView(id=aid, x=po.x, y=po.y, heading=po.theta, energy=s['energy'], max_energy=s['max_energy'],
                                        age=s['age'], speed=s['speed'], sprint_speed=s['sprint_speed'], hearing=s['hearing_radius'],
                                        vision_range=s['vision_range'], vision_angle=s['vision_angle'], biome=s['biome'], state=s)
            rects = list(frame.rects) + (_boundary_rects(self.width, self.height) if frame.absolute else [])
            agent_tuples = [(a.id, a.x, a.y, a.heading) for a in agents.values()]
            for tr in frame.tracks:
                x, y, heading = tr.x, tr.y, tr.heading
                # advance one predator step: the observation predates this tick's predator move
                if getattr(tr, '_seen_now', False) and frame.absolute:
                    st = pm.PredState(x, y, heading)
                    try:
                        st2, info = pm.step(st, agent_tuples, rects, self.width, self.height, lambda p: 1.0)
                        x, y, heading = st2.x, st2.y, st2.heading
                    except Exception:
                        pass
                predators.append(PredatorView(pid=tr.tid, x=x, y=y, heading=heading, vx=tr.vx, vy=tr.vy, last_seen=tr.last_seen,
                                              still_ticks=tr.still, resting=None, energy=None, fresh=getattr(tr, '_seen_now', False)))
            fruits = [FoodView(f[0], f[1]) for f in frame.fruits]
            trees = [FoodView(f[0], f[1]) for f in frame.trees]
        world = _EstimatedState(time=self.time, width=self.width, height=self.height, agents=agents, predators=predators,
                                rects=rects, fruits=fruits, trees=trees, complete_map=False)
        world._frame = frame
        self.state = world
        return world


class _EstimatedState(WorldState):
    _frame = None

    @property
    def slow_key(self):
        f = self._frame
        return None if f is None else ('est', id(f), len(f.biome))

    def biome_at(self, p):
        f = self._frame
        if f is None:
            return 1.0
        return f.biome.get((int(p[0] // 50), int(p[1] // 50)), 1.0)

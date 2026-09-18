"""Turning-circle dodge as the close-range lead ("leash") movement.

The stock leash keeps 40-55 in front of a charging predator by matching its speed, which
means sprinting (about 200 energy per delivery, guide gate 330). This variant keeps the
predator inside hearing but lets it close, and at the last moment steps into its turning
circle (its direct chase turns at most 0.3 rad per tick, so at 15 per tick it circles a
radius of 50 and cannot reach an agent beside it). Against a sprinting predator that costs
one 20-unit sprint tick per dodge; against a walking one (below 40 energy) nothing but
walking. Established on an arranged wall fixture in Oscar's research checkout
(survival/research/guide_dodge/) and ported here unchanged in its mechanics.

Only the movement choice changes. Path planning, the hook, rest handling, the run-in down
the mouth axis, the flyby and the enter endgames are the stock leash's (lure.py). The
predator model is a vectorised clone of the public rule (predator.py at acfc31a4) with the
engine's collision deflection, evaluated on the oracle world. It reads predator energy only
through ``pred_next_speed`` like the leash does.

    from models.trapper.dodge import DodgeLure, install
    install()          # manager.Lure = DodgeLure for this process

Energy gate: the dodge can start from any energy that still affords one sprint tick
(``max_energy/5 + 12``); ``DodgeManager`` lowers the leash gate accordingly.
"""
from __future__ import annotations

import inspect
import math
import textwrap

import numpy as np

from . import lure as _lure
from .geometry import dist, heading_of, sub, wrap
from .lure import Lure, LEASH_GAP, is_resting, pred_next_speed, predator_target
from .motion import next_waypoint
from .world import PRED_HEARING as PRED_HEAR, PRED_VISION, PRED_CONE, PRED_WALK, PRED_SPRINT, PREDATOR_RADIUS, AGENT_RADIUS

HALF_CONE = PRED_CONE / 2
W = dict(cost=3., goal=1., unseen=40., band=1.5, band_gap=52., far=10., far_gap=60., margin=16.,
         dodged_angle=1.0, undodged_penalty=12., other_hard=30., other_soft=60., slow=8., local=380., sprint_away=20., clear_cap=50., clear_reward=.5, goal_ramp=(18., 33.))


def wrap_np(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def in_rects(pts, rects, radius):
    """Engine _in_obstacle: strict inequalities on rectangles expanded by radius."""
    inside = np.zeros(len(pts), dtype=bool)
    for (x, y, w, h) in rects:
        inside |= (pts[:, 0] > x - radius) & (pts[:, 0] < x + w + radius) & (pts[:, 1] > y - radius) & (pts[:, 1] < y + h + radius)
    return inside


def deflect(start, moves, rects, radius, world):
    """Engine collision handling: rotate the move in alternating 10-degree steps until clear."""
    end = start + moves
    blocked = in_rects(end, rects, radius)
    if blocked.any():
        length = np.linalg.norm(moves, axis=1)
        heading = np.arctan2(moves[:, 1], moves[:, 0])
        done = ~blocked
        for i in range(36):
            test = heading + math.pi / 18 * ((i + 1) // 2) * (-1) ** i
            cand = start + np.stack([length * np.cos(test), length * np.sin(test)], axis=1)
            ok = ~done & ~in_rects(cand, rects, radius)
            end[ok] = cand[ok]
            done |= ok
            if done.all():
                break
        end[~done] = start[~done]
    end[:, 0] = np.clip(end[:, 0], radius, world[0] - radius)
    end[:, 1] = np.clip(end[:, 1], radius, world[1] - radius)
    return end


def predator_step(p, h, guide, guide_dir, speed, rects, world, others=None):
    """Public predator rule, vectorised over N (predator state, guide position) pairs.

    ``others``: fixed points of other agents (M,2), heard inside 60 or seen inside 250 in the
    cone (line of sight ignored). The predator takes the nearest perceived one; if that is not
    the guide it moves toward it instead. Returns new position (N,2), new heading (N,),
    target (0 guide, 1 another agent, -1 none).
    """
    d = guide - p
    dist_g = np.linalg.norm(d, axis=1)
    ang = wrap_np(np.arctan2(d[:, 1], d[:, 0]) - h)
    seen = (dist_g <= PRED_HEAR) | ((dist_g <= PRED_VISION) & (np.abs(ang) <= HALF_CONE))
    n = len(p)
    if others is not None and len(others):
        do = others[None, :, :] - p[:, None, :]                       # (N,M,2)
        dd = np.linalg.norm(do, axis=2)
        ao = wrap_np(np.arctan2(do[:, :, 1], do[:, :, 0]) - h[:, None])
        so = (dd <= PRED_HEAR) | ((dd <= PRED_VISION) & (np.abs(ao) <= HALF_CONE))
        dd = np.where(so, dd, np.inf)
        k = np.argmin(dd, axis=1)
        do_best = dd[np.arange(n), k]
        ox = others[k, 0]; oy = others[k, 1]
    else:
        do_best = np.full(n, np.inf); ox = oy = np.zeros(n)
    other_first = np.isfinite(do_best) & ~(seen & (dist_g <= do_best))
    target = np.where(seen & ~other_first, 0, np.where(other_first, 1, -1))
    tx = np.where(target == 1, ox, guide[:, 0]); ty = np.where(target == 1, oy, guide[:, 1])
    tdist = np.where(target == 1, do_best, dist_g)
    tang = np.where(target == 1, wrap_np(np.arctan2(ty - p[:, 1], tx - p[:, 0]) - h), ang)
    rel = wrap_np(np.arctan2(p[:, 1] - guide[:, 1], p[:, 0] - guide[:, 0]) - guide_dir)
    rel = np.where(target == 1, math.pi, rel)                          # unknown facing of others: assume charge
    charge = (np.abs(rel) > math.pi / 2) | (tdist < PRED_HEAR * 1.5)
    turn = np.clip(tang * .5, -.3, .3); small = np.abs(tang) <= .05
    move_dir = np.where(charge, np.where(small, tang, turn), tang - np.sign(rel) * math.pi / 4)
    step = np.where(charge, np.minimum(speed, tdist), speed)
    step = np.where(target < 0, PRED_WALK, step); move_dir = np.where(target < 0, 0., move_dir)
    moves = np.stack([step * np.cos(h + move_dir), step * np.sin(h + move_dir)], axis=1)
    new = deflect(p.copy(), moves, rects, PREDATOR_RADIUS, world)
    nh = np.where(charge, np.where(small, h, h + turn), np.arctan2(ty - new[:, 1], tx - new[:, 0]))
    nh = np.where(target < 0, h, nh)
    return new, nh, target


class DodgeLure(Lure):
    """Stock leash with the approach movement replaced by the turning-circle dodge."""

    metrics = dict(dodge_ticks=0, dodge_sprint_ticks=0, dodge_energy=0.0)

    def act(self, d, a, p):
        self._d = d
        return super().act(d, a, p)

    # ---- the two leash entry points that choose an approach movement
    def _leash_choose(self, a, p, desired, gap, v_pred, gap_target, cap, others=(), run=False, ceil=None):
        d = self._d
        # the run down the mouth axis stays the stock speed-matched lead only when the guide must
        # enter the passage itself (the predator has to stay behind it); at a staffed mouth the
        # predator overshooting the dodging guide toward the bait is exactly the delivery
        if d is None or gap_target not in (LEASH_GAP, 110.0) or (run and d.become_bait):
            return super()._leash_choose(a, p, desired, gap, v_pred, gap_target, cap, others, run=run, ceil=ceil)
        if gap_target == 110.0:
            # the leash wanted to open the gap for its walk lead; the dodge just keeps towing
            wp = next_waypoint(a.p, d.path, reach=8.0) if d.path else None
            if wp is not None:
                desired = heading_of(sub(wp, a.p))
        return self._dodge_choose(a, p, desired, gap, others)

    def _walk_lead(self, d, a, p, gap, wp, goal, waking):
        """The leash falls back to its pivot-band walk lead when its sprint budget is gone; the
        dodge keeps towing as long as one sprint tick is affordable."""
        w = self.world
        resting = bool(p.resting) if p.resting is not None else (3 <= p.still_ticks < 30)
        if a.energy < a.max_energy / 5 + 12.0 or (resting and not waking):
            return super()._walk_lead(d, a, p, gap, wp, goal, waking)
        others = [q for q in w.recent_predators() if q.pid != p.pid and dist(q.p, a.p) < 140
                  and (q.pid in self.held or not (bool(q.resting) if q.resting is not None else is_resting(q)))]
        desired = heading_of(sub(wp, a.p))
        h, real, g2 = self._dodge_choose(a, p, desired, gap, others)
        if wp is goal or dist(wp, goal) < 1e-6:
            real = min(real, dist(a.p, goal))
        d.decision = f'dodge-lead (low budget): gap {gap:.0f}->{g2:.0f}, v {real:.0f}, wp {tuple(round(v) for v in wp)}, e {a.energy:.0f}'
        return self._move_heading(a, h, real) if real > 0.05 else self._facing(d, a, p, _lure.hold(a))

    def _flyby(self, d, a, p, gap):
        """Stock flyby sprints 14 ticks along the face; once the predator sits at the mouth on the
        bait, walking clear is enough. Sprint only while it could still lunge at us."""
        act = super()._flyby(d, a, p, gap)
        if act is not None and gap > 45.0 and act.get('move_distance', 0.) > a.walk:
            act['move_distance'] = float(a.walk)
            d.decision = d.decision.replace('sprinting aside', 'walking aside')
        return act

    # ---- the dodge chooser
    def _dodge_choose(self, a, p, desired, gap, others=()):
        w = self.world
        world = (w.width, w.height)
        pos = np.array(a.p, float); pp = np.array(p.p, float); ph = float(p.heading)
        rects = [(r.x, r.y, r.w, r.h) for r in w.rects if r.distance(a.p) < W['local']] if hasattr(w.rects[0], 'distance') \
            else [(r.x, r.y, r.w, r.h) for r in w.rects]
        mod = a.move_modifier
        walk = a.walk * mod
        can_sprint = a.can_sprint and a.energy >= a.max_energy / 5 + 12.0
        sprint = (a.sprint_speed if can_sprint else a.walk) * mod
        # predator speed on the oracle: 15 above 40 energy, 11 below (cannot sprint again before resting)
        v_now = pred_next_speed(p) * w.biome_at(p.p)
        walk_mode = 0.2 < v_now < 12.5
        speed = PRED_WALK if walk_mode else PRED_SPRINT
        speed *= w.biome_at(p.p)
        # candidates (ground displacement)
        angles = np.arange(24) * math.pi / 12
        steps = sorted({0., 5. * mod, walk, min(15.2 * mod, sprint), sprint})
        vec = np.array([s * np.array([math.cos(t), math.sin(t)]) for s in steps for t in angles])
        vec = np.unique(np.round(vec, 6), axis=0)
        q = deflect(np.repeat(pos[None], len(vec), 0), vec, rects, AGENT_RADIUS, world); vec = q - pos
        length = np.linalg.norm(vec, axis=1) / max(mod, 1e-6)          # requested distance
        cost = np.minimum(length, a.walk) * .05 + np.maximum(length - a.walk, 0) * .5
        facing = np.arctan2(vec[:, 1], vec[:, 0])                        # we move with our back to it
        facing = np.where(length > 1e-6, facing, a.heading)
        # other agents the predator might prefer (baits, bystanders): fixed points
        pts = np.array([[o.x, o.y] for o in w.agents.values() if o.id != a.id and dist(o.p, p.p) < 260], float) if w.agents else None
        pts = pts if pts is not None and len(pts) else None
        clear, first, p1 = self._clearance(pp, ph, q, facing, speed, speed, rects, world, walk, sprint if can_sprint else walk, pts)
        score = cost * W['cost'] + np.maximum(W['margin'] - clear, 0) ** 2 * 100 + (first < 0) * W['unseen']
        # prefer real room over skimming the kill radius: a dodge that makes it overshoot and circle
        # earns clearance of 30-50, hugging the margin earns none
        score -= np.minimum(clear, W['clear_cap']) * W['clear_reward']
        d1 = np.linalg.norm(q - p1, axis=1)
        # keep it inside hearing, but never chase an approaching predator to do so
        closing = d1 < gap - 2.0
        score += np.maximum(d1 - W['band_gap'], 0) * W['band'] * (~closing) + np.maximum(d1 - W['far_gap'], 0) * W['far'] * (~closing)
        # sprint only to dodge, never to outrun: a sprint directed away from it is the leash's
        # speed matching, which a finite horizon would otherwise postpone the dodge for forever
        away = heading_of(sub(a.p, p.p))
        score += (length > a.walk + 1e-6) * np.maximum(np.cos(wrap_np(facing - away)), 0) * W['sprint_away']
        # progress toward the waypoint heading
        dev = np.abs(wrap_np(facing - desired))
        progress = np.where(length > 1e-6, np.linalg.norm(vec, axis=1) * np.cos(dev), 0.)
        # inside the dodge range progress does not matter: the move that makes it overshoot wins
        lo, hi = W['goal_ramp']
        score -= progress * W['goal'] * min(1., max(0., (gap - lo) / (hi - lo)))
        # slower ground ahead costs (river, swamp)
        if w.slow_key is not None:
            here = w.biome_at(a.p)
            slow = np.array([w.biome_at((x, y)) < here - .05 for x, y in q])
            score += slow * W['slow']
        # other predators: predicted step toward us if they chase us, else straight on
        for o in others:
            chasing = predator_target(w, o) == a.id
            v_o = 0. if (bool(o.resting) if o.resting is not None else is_resting(o)) else pred_next_speed(o) * w.biome_at(o.p)
            if chasing and v_o > 0:
                on, _, _ = predator_step(np.repeat(np.array(o.p, float)[None], len(q), 0), np.full(len(q), float(o.heading)),
                                         q, facing, v_o, rects, world)
            else:
                on = np.repeat(np.array([o.x + o.vx, o.y + o.vy])[None], len(q), 0)
            do = np.linalg.norm(q - on, axis=1)
            score += np.maximum(W['other_hard'] - do, 0) ** 2 * 50 + np.maximum(W['other_soft'] - do, 0) * 2
        k = int(np.argmin(score))
        DodgeLure.last = dict(score=score, clear=clear, cost=cost, progress=progress, d1=d1, vec=vec, first=first, gap=gap, speed=speed, k=k, walk_mode=walk_mode)
        real = float(np.linalg.norm(vec[k]))
        DodgeLure.metrics['dodge_ticks'] += 1
        if length[k] > a.walk + 1e-6:
            DodgeLure.metrics['dodge_sprint_ticks'] += 1
        DodgeLure.metrics['dodge_energy'] += float(cost[k])
        h = float(facing[k]) if real > 1e-6 else heading_of(sub(a.p, p.p))
        return h, real, float(d1[k])

    def _clearance(self, p, h, q, facing, speed, later, rects, world, walk, sprint2, others):
        """Three-tick existential safety (see the research README): after the candidate move,
        does a second step (walk or sprint) and a third (walk) exist that keep every gap above
        the margin? The third step is discounted while the predator still points at us."""
        n = len(q); m = W['margin']; dirs = np.arange(16) * math.pi / 8
        wv = np.stack([walk * np.cos(dirs), walk * np.sin(dirs)], axis=1)
        m2 = np.concatenate([wv, np.stack([sprint2 * np.cos(dirs), sprint2 * np.sin(dirs)], axis=1)]); k2 = len(m2); k3 = 16
        p1, h1, t1 = predator_step(np.repeat(p[None], n, 0), np.full(n, h), q, facing, speed, rects, world, others)
        g1 = np.linalg.norm(q - p1, axis=1)
        q2 = deflect(np.repeat(q, k2, 0), np.tile(m2, (n, 1)), rects, AGENT_RADIUS, world)
        pr = np.repeat(p1, k2, 0); hr = np.repeat(h1, k2)
        f2 = np.arctan2(q2[:, 1] - np.repeat(q, k2, 0)[:, 1], q2[:, 0] - np.repeat(q, k2, 0)[:, 0])
        p2, h2, t2 = predator_step(pr, hr, q2, f2, later, rects, world, others)
        g2 = np.linalg.norm(q2 - p2, axis=1)
        q3 = deflect(np.repeat(q2, k3, 0), np.tile(wv, (n * k2, 1)), rects, AGENT_RADIUS, world)
        pr = np.repeat(p2, k3, 0); hr = np.repeat(h2, k3)
        f3 = np.arctan2(q3[:, 1] - np.repeat(q2, k3, 0)[:, 1], q3[:, 0] - np.repeat(q2, k3, 0)[:, 0])
        p3, h3, t3 = predator_step(pr, hr, q3, f3, later, rects, world, others)
        g3 = np.linalg.norm(q3 - p3, axis=1)
        aimed = np.abs(wrap_np(np.arctan2(q3[:, 1] - p3[:, 1], q3[:, 0] - p3[:, 0]) - h3)) < W['dodged_angle']
        g3 = np.where(aimed, g3 - W['undodged_penalty'], g3).reshape(n, k2, k3).max(axis=2)
        value = np.minimum(g2.reshape(n, k2), g3).max(axis=1)
        return np.minimum(g1, value), t1, p1


# Three lines of the stock leash body do not suit a walking dodge: its sprint-budget fallback
# (118 energy), its rest distance (42: a walker cannot dodge a wake-up lunge from there near a
# mouth) and the truncation of the last move to the distance left to the goal (which cut dodge
# steps to a few units at the flyby point). They are patched from the stock source at import
# time; every replacement must apply exactly once, so a change in lure.py fails loudly here.
_LEASH_PATCHES = [
    ("if (not a.can_sprint or a.energy < 118.0) and not (d.become_bait and dist(a.p, site.holder) < 60):",
     "if (not a.can_sprint or a.energy < a.max_energy / 5 + 12.0) and not (d.become_bait and dist(a.p, site.holder) < 60):"),
    ("if gap > 48:\n", "if gap > 56:\n"),
    ("return step_toward(a, p.p, min(a.walk * a.move_modifier, gap - 42.0))", "return step_toward(a, p.p, min(a.walk * a.move_modifier, gap - 50.0))"),
    ("if gap < 36:\n", "if gap < 44:\n"),
    ("return self._move_heading(a, h, min(a.walk * a.move_modifier, 42.0 - gap))", "return self._move_heading(a, h, min(a.walk * a.move_modifier, 50.0 - gap))"),
    ("if wp is goal or dist(wp, goal) < 1e-6:\n", "if (wp is goal or dist(wp, goal) < 1e-6) and d.become_bait:\n"),
]


def _patched_leash():
    src = textwrap.dedent(inspect.getsource(Lure._leash))
    for old, new in _LEASH_PATCHES:
        if src.count(old) != 1:
            raise RuntimeError(f'dodge.py: stock Lure._leash changed; patch not found once: {old[:60]!r}')
        src = src.replace(old, new)
    ns = dict(vars(_lure))
    exec(src, ns)
    return ns['_leash']


DodgeLure._leash = _patched_leash()


def install(manager_module=None):
    """Use DodgeLure for every delivery in this process (the manager instantiates ``Lure`` by name)."""
    from . import manager as _manager
    (manager_module or _manager).Lure = DodgeLure
    return DodgeLure

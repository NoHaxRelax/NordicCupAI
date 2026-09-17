"""Delivery of one predator to one trap site by one guide.

Protocol (wall site; the gap site differs only at the end):

ATTRACT  - intercept the predator's forward ray at ~140 ahead of it until we are
           the nearest agent it observes. Never inside 110 of an awake predator.
OPEN     - if it noticed us too close, run away (sprint if we can, steering
           around obstacles) until the gap is back above 110.
LEAD     - retreat toward the corridor entry while facing it with a small
           alternating gaze offset. Facing it keeps the engine in its pivot
           branch: it approaches at ~10.6/tick (sprinting) or ~7.8/tick
           (walking) in a zig-zag, slower than our walk, so a non-sprinting guide
           is safe while the gap stays above 90 (checked with the exact model in
           scripts/trapper/check_lead_dynamics.py). Because it follows us, we can
           only steer within a cone around "directly away": the desired heading is
           clamped into that cone, which makes the pair arc gently toward the
           goal. The approach ends with a straight run along the site axis.
CORRIDOR - at the corridor entry we turn our back: it charges straight down the
           axis behind us while we walk to the front point.
FRONT    - stand at the front point and get eaten; its next nearest audible agent
           is the holder across the wall (a gap: the bait it sees down the passage).
ENTER    - (become_bait, first predator at a gap) walk into the passage instead.

Everything reads the WorldState only; exact with the oracle, estimated later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import add, sub, mul, dot, dist, unit, wrap, heading_of, polar, path_clear, los_clear
from .motion import action, step_toward, hold, speed_for, next_waypoint
from .paths import plan
from .predator_model import PredState, observed_agents
from .sites import Site
from .world import WorldState, AgentView, PredatorView, PRED_CHARGE_RANGE

LEAD_DISTANCE = 125.0       # regulate the gap around this while leading (must stay > 90)
LEAD_MIN = 100.0            # below this a sprinting guide tops up the gap
ATTRACT_MIN = 110.0         # never approach an awake predator closer than this
WAIT_MIN = 125.0            # wait to be noticed only beyond this: its first pivot step eats ~10
GUIDE_RESERVE = 35.0        # abort attracting below this energy
GAZE_OFFSET = 0.12          # alternating facing offset that keeps it zig-zagging straight
RUN_IN = 250.0              # straight run along the axis before the corridor entry
CORRIDOR_MIN_GAP = 100.0    # turn our back only with at least this gap
RETREAT_DEV_SPRINT = math.radians(25)   # steering cone while it sprints (closes 0.6/tick head-on)
RETREAT_DEV_WALK = math.radians(40)     # steering cone while it walks (falls back 2.2/tick)


@dataclass
class Delivery:
    site: Site
    guide: int
    pid: int
    become_bait: bool = False
    phase: str = 'ATTRACT'
    created: float = 0.0
    waypoints: list = field(default_factory=list)
    gaze: int = 1
    events: list = field(default_factory=list)
    lost_ticks: int = 0
    stall_ticks: int = 0
    flee_heading: float | None = None
    defer_ticks: int = 0
    done: str | None = None       # 'delivered' | 'guide_captured' | 'failed:<reason>'
    decision: str = ''

    def event(self, t, kind, **data):
        self.events.append(dict(t=round(t, 1), kind=kind, **data))


def pred_state(p: PredatorView) -> PredState:
    return PredState(p.x, p.y, p.heading, energy=p.energy if p.energy is not None else 100.0,
                     resting=bool(p.resting) if p.resting is not None else False)


def predator_target(world: WorldState, p: PredatorView):
    """Nearest agent the predator observes right now (its next target), or None. Cached per world."""
    cache = world.__dict__.setdefault('_target_cache', {})
    if p.pid in cache:
        return cache[p.pid]
    agents = [(a.id, a.x, a.y, a.heading) for a in world.agents.values()]
    ids = observed_agents(pred_state(p), agents, world.rects)
    cache[p.pid] = ids[0] if ids else None
    return cache[p.pid]


def run_in_point(site: Site):
    return add(site.corridor_start, mul(site.normal, RUN_IN))


def is_resting(p: PredatorView):
    return bool(p.resting) if p.resting is not None else p.still_ticks >= 3


def steer(a: AgentView, p: PredatorView, desired, rects, max_dev, probe=70.0, keep_los=True):
    """Heading closest to ``desired`` within ``max_dev`` of directly-away-from-p that
    does not run into an obstacle within ``probe`` units and, when ``keep_los``,
    keeps the predator's line of sight to us after both move. Falls back to the
    clear heading with the largest away component."""
    away = heading_of(sub(a.p, p.p))
    best = None
    fallback = None
    p_next = add(p.p, mul(unit(sub(a.p, p.p)), 10.0))
    for k in range(-12, 13):
        h = away + max_dev * k / 12
        end = add(a.p, polar(h, probe))
        if not path_clear(a.p, end, 6.0, rects):
            continue
        cost = abs(wrap(h - desired))
        if keep_los and not los_clear(p_next, add(a.p, polar(h, 12.0)), rects):
            cost += 1.5
        if best is None or cost < best[0]:
            best = (cost, h)
        comp = math.cos(h - away)
        if fallback is None or comp > fallback[0]:
            fallback = (comp, h)
    if best is not None:
        return best[1]
    if fallback is not None:
        return fallback[1]
    return desired


DEFER = None   # returned by Lure.act when the society's own action should be used this tick


class Lure:
    """Per-tick action for the guide of one delivery."""

    def __init__(self, world: WorldState, held=()):
        self.world = world
        self.held = set(held)

    def act(self, d: Delivery, a: AgentView, p: PredatorView | None):
        w = self.world
        # another loose predator close by: let the society's flee logic act for us
        for q in w.predators:
            if p is not None and q.pid == p.pid or q.pid in self.held or is_resting(q):
                continue
            if dist(q.p, a.p) < 95 and d.phase not in ('FRONT', 'ENTER'):
                d.defer_ticks += 1
                if d.defer_ticks > 80:
                    d.done = 'failed:other_predator'
                d.decision = f'deferring: predator {q.pid} at {dist(q.p, a.p):.0f}'
                return DEFER
        if p is None:
            d.lost_ticks += 1
            if d.lost_ticks > 30:
                d.done = 'failed:predator_lost'
            d.decision = 'predator unknown; holding'
            return hold(a)
        d.lost_ticks = 0
        target = predator_target(w, p)
        following = target == a.id
        gap = dist(a.p, p.p)
        if d.phase in ('ATTRACT', 'OPEN', 'LEAD') and is_resting(p):
            d.decision = 'predator resting; waiting in its cone'
            return self._wait_in_cone(d, a, p)
        if d.phase == 'ATTRACT':
            if following:
                d.event(w.time, 'following', gap=round(gap, 1))
                if gap >= PRED_CHARGE_RANGE + 12:
                    d.phase = 'LEAD'
                    d.waypoints = self._lead_path(a, d.site)
                else:
                    d.phase = 'OPEN'
            else:
                return self._attract(d, a, p, target, gap)
        if d.phase == 'OPEN':
            if gap >= ATTRACT_MIN:
                d.phase = 'LEAD'
                d.defer_ticks = 0
                d.waypoints = self._lead_path(a, d.site)
                d.event(w.time, 'opened', gap=round(gap, 1))
            else:
                return self._open(d, a, p, gap)
        if d.phase == 'LEAD':
            if not following:
                d.stall_ticks += 1
                if d.stall_ticks > 20:
                    d.phase = 'ATTRACT'
                    d.stall_ticks = 0
                    d.event(w.time, 'lost_attention', target=target)
                    return self._attract(d, a, p, target, gap)
            else:
                d.stall_ticks = 0
            if gap < PRED_CHARGE_RANGE:
                d.phase = 'OPEN'
                d.event(w.time, 'too_close', gap=round(gap, 1))
                return self._open(d, a, p, gap)
            return self._lead(d, a, p, gap)
        if d.phase == 'CORRIDOR':
            return self._corridor(d, a, p, gap, following)
        if d.phase == 'FRONT':
            return self._front(d, a, p, gap)
        if d.phase == 'ENTER':
            return self._enter(d, a, p)
        return hold(a)

    # ------------------------------------------------------------------ helpers
    def _gaze(self, d: Delivery, a: AgentView, p: PredatorView):
        """Turn angle that faces the predator with an alternating small offset.
        Exact facing (rel_dir == 0) makes the engine charge straight; a constant
        offset makes it swing wide. Alternating keeps it in the slow pivot branch."""
        d.gaze = -d.gaze
        return float(wrap(heading_of(sub(p.p, a.p)) + GAZE_OFFSET * d.gaze - a.heading))

    def _facing(self, d, a, p, act):
        act['turn_angle'] = self._gaze(d, a, p)
        return act

    def _move_heading(self, a: AgentView, heading, real_distance):
        return step_toward(a, add(a.p, polar(heading, max(real_distance, 1.0) + 1.0)), real_distance)

    # ------------------------------------------------------------------ phases
    def _intercept_point(self, a: AgentView, p: PredatorView, ahead=140.0):
        """Point ``ahead`` in front of where the predator will be when we can get there."""
        v = 11.0 if p.speed < 12.0 else 15.0
        if p.speed < 0.5:
            v = 0.0
        h = (math.cos(p.heading), math.sin(p.heading))
        speed = a.walk * a.move_modifier
        for k in range(0, 36, 3):
            q = add(p.p, mul(h, ahead + v * k))
            if dist(a.p, q) <= speed * k + 1e-6:
                return q, k
        return None, None

    def _approach_front(self, d: Delivery, a: AgentView, p: PredatorView, gap, berth, label):
        w = self.world
        if a.energy < GUIDE_RESERVE and d.phase == 'ATTRACT':
            d.done = 'failed:guide_energy'
            d.decision = 'attract: out of energy'
            return hold(a)
        bearing = abs(wrap(heading_of(sub(a.p, p.p)) - p.heading))
        if gap < berth and not is_resting(p):
            d.decision = f'{label}: too close ({gap:.0f}); backing off'
            h = steer(a, p, heading_of(sub(a.p, p.p)), w.rects, math.radians(60))
            return self._facing(d, a, p, self._move_heading(a, h, speed_for(a, a.energy > 150)))
        if bearing < 0.35 and WAIT_MIN <= gap <= 240:
            d.decision = f'{label}: in its cone at {gap:.0f}, waiting'
            return self._facing(d, a, p, hold(a))
        goal, k = self._intercept_point(a, p, 140.0)
        if goal is None or (d.phase == 'ATTRACT' and k > 15):
            d.stall_ticks += 1
            if d.phase == 'ATTRACT' and d.stall_ticks > 10:
                d.done = 'failed:cannot_intercept'
                d.decision = 'attract: cannot intercept'
            else:
                d.decision = f'{label}: cannot intercept yet; waiting'
            return self._facing(d, a, p, hold(a))
        d.stall_ticks = 0
        if not d.waypoints or dist(d.waypoints[-1], goal) > 25:
            path = plan(w.rects, w.width, w.height, a.p, goal, radius=6.0, avoid=[(p.p, berth)])
            d.waypoints = (path[1:] if path and len(path) > 1 else [goal])
        wp = next_waypoint(a.p, d.waypoints) or goal
        d.decision = f'{label}: intercept in {k} ticks, gap {gap:.0f}, bearing {math.degrees(bearing):.0f}'
        return self._facing(d, a, p, step_toward(a, wp, speed_for(a, a.energy > 180 and bearing > 0.35)))

    def _attract(self, d: Delivery, a: AgentView, p: PredatorView, target, gap):
        w = self.world
        if target is not None and target in w.agents and not is_resting(p):
            # it is chasing someone else: be nearer to it than they are, but never inside 90
            other = w.agents[target]
            want = max(PRED_CHARGE_RANGE + 5, dist(other.p, p.p) - 15.0)
            if gap > want + 5:
                goal = add(p.p, mul(unit(sub(a.p, p.p)), want))
                d.decision = f'attract: undercutting agent {target} at {want:.0f}'
                return self._facing(d, a, p, step_toward(a, goal, speed_for(a, a.can_sprint)))
            d.decision = f'attract: waiting to undercut agent {target}'
            return self._facing(d, a, p, hold(a))
        return self._approach_front(d, a, p, gap, ATTRACT_MIN, 'attract')

    def _open(self, d: Delivery, a: AgentView, p: PredatorView, gap):
        # the society's joint-threat retreat is better tuned than anything here
        d.decision = f'open: society flees ({gap:.0f})'
        d.defer_ticks += 1
        if d.defer_ticks > 150:
            d.done = 'failed:cannot_open'
        return DEFER

    def _lead_path(self, a: AgentView, site: Site):
        """Waypoints to the corridor entry: join the approach axis at about our own
        distance from the trap (never further out than the run-in point, never
        inside the corridor), then walk the axis to the corridor start."""
        w = self.world
        out = dot(sub(a.p, site.front_mid), site.normal)
        from .sites import CORRIDOR
        entry_out = max(CORRIDOR + 30.0, min(CORRIDOR + RUN_IN, out - 40.0))
        entry = add(site.front_mid, mul(site.normal, entry_out))
        path = plan(w.rects, w.width, w.height, a.p, entry, radius=22.0)
        if path is None:
            path = plan(w.rects, w.width, w.height, a.p, entry, radius=12.0) or [a.p, entry]
        return path[1:] + [site.corridor_start]

    def _lead(self, d: Delivery, a: AgentView, p: PredatorView, gap):
        w = self.world
        site = d.site
        d.flee_heading = None
        wp = next_waypoint(a.p, d.waypoints)
        if dist(a.p, site.corridor_start) < 8.0:
            if gap >= CORRIDOR_MIN_GAP:
                d.phase = 'CORRIDOR'
                d.event(w.time, 'corridor', gap=round(gap, 1))
                return self._corridor(d, a, p, gap, True)
            # keep it in pivot mode down the axis until the gap is comfortable
            wp = site.front
        sprinting_pred = p.speed > 13.0
        desired = heading_of(sub(wp, a.p)) if wp else heading_of(sub(a.p, p.p))
        # In its pivot branch it closes radially at speed*cos(45): 10.6 sprinting, 7.8 walking.
        # Our radial retreat is walk*cos(dev); keep it above its closing speed plus a margin,
        # a larger one while the gap is short so we recover before steering again.
        walk = a.walk * a.move_modifier
        radial = (15.0 if sprinting_pred else 11.0) * math.cos(math.pi / 4)
        margin = 0.3 if gap >= LEAD_DISTANCE - 10 else 1.5
        # gap surplus above the target may be spent on steering, over ~8 ticks
        allowed_loss = max(0.0, (gap - LEAD_DISTANCE) / 8.0)
        need = radial + margin - allowed_loss
        if need <= 0:
            dev = math.pi
        else:
            ratio = need / max(walk, 1e-6)
            dev = 0.0 if ratio >= 1.0 else math.acos(ratio)
        away = heading_of(sub(a.p, p.p))
        needed = abs(wrap(desired - away))
        sprint_budget = a.energy - a.max_energy / 5 - 40.0
        if needed > dev + 0.05 and sprint_budget > 80.0 and a.can_sprint and gap >= LEAD_MIN - 5:
            # sprint-steer: at 20/tick we can retreat up to ~57 degrees off "away" and still out-pace
            # its radial closing, which rotates the pair about 0.13 rad/tick
            sp = a.sprint_speed * a.move_modifier
            ratio_s = (radial + margin) / max(sp, 1e-6)
            dev_s = 0.0 if ratio_s >= 1.0 else math.acos(ratio_s)
            h = steer(a, p, desired, w.rects, max(dev_s, dev))
            d.decision = f'lead: sprint-steer, gap {gap:.0f}, needed {math.degrees(needed):.0f}, dev {math.degrees(wrap(h - away)):.0f}'
            return self._facing(d, a, p, self._move_heading(a, h, sp))
        h = steer(a, p, desired, w.rects, dev)
        if gap < LEAD_MIN and a.can_sprint and sprinting_pred:
            d.decision = f'lead: topping up the gap ({gap:.0f})'
            act = self._move_heading(a, h, speed_for(a, True))
        else:
            # never slow down while it sprints; when it walks, keep the gap near LEAD_DISTANCE
            speed = walk
            if not sprinting_pred and gap > LEAD_DISTANCE + 10:
                speed = max(0.0, walk - (gap - LEAD_DISTANCE - 10) * 0.3)
            if gap > LEAD_DISTANCE + 30 and p.speed < 0.5:
                speed = 0.0
            d.decision = f'lead: gap {gap:.0f}, speed {speed:.0f}, desired {math.degrees(desired):.0f}, away {math.degrees(away):.0f}, h {math.degrees(h):.0f}, cone {math.degrees(dev):.0f}, wp {tuple(round(v) for v in wp) if wp else None}, {len(d.waypoints)} wps'
            act = self._move_heading(a, h, speed) if speed > 0 else hold(a)
        return self._facing(d, a, p, act)

    def _corridor(self, d: Delivery, a: AgentView, p: PredatorView, gap, following):
        w = self.world
        site = d.site
        if not following and not is_resting(p):
            d.stall_ticks += 1
            if d.stall_ticks > 15:
                d.phase = 'LEAD'
                d.waypoints = self._lead_path(a, site)
                d.event(w.time, 'corridor_lost')
                d.stall_ticks = 0
                return self._lead(d, a, p, gap)
        else:
            d.stall_ticks = 0
        goal = site.front if not d.become_bait else site.front_mid
        if dist(a.p, goal) < 2.0:
            d.phase = 'ENTER' if d.become_bait else 'FRONT'
            d.event(w.time, d.phase.lower(), gap=round(gap, 1))
            return self._enter(d, a, p) if d.become_bait else self._front(d, a, p, gap)
        # walk down the axis with our back to it; slow down if it lags far behind
        speed = a.walk * a.move_modifier
        if is_resting(p):
            speed = 0.0
        elif gap > 140:
            speed *= 0.5
        if gap < 25 and a.can_sprint and dist(a.p, goal) > 30:
            speed = speed_for(a, True)
        d.decision = f'corridor: gap {gap:.0f}'
        away = add(a.p, mul(mul(site.normal, -1.0), 50.0))
        return step_toward(a, goal, speed, face=away) if speed > 0 else action(a.id, 0.0, 0.0, wrap(heading_of(sub(away, a.p)) - a.heading))

    def _front(self, d: Delivery, a: AgentView, p: PredatorView, gap):
        site = d.site
        if dist(a.p, site.front) > 1.0:
            d.decision = 'front: settling'
            return step_toward(a, site.front, min(a.walk * a.move_modifier, dist(a.p, site.front)), face=site.front_mid)
        d.decision = f'front: waiting for capture, gap {gap:.0f}'
        return action(a.id, 0.0, 0.0, wrap(heading_of(sub(site.front_mid, a.p)) - a.heading))

    def _enter(self, d: Delivery, a: AgentView, p: PredatorView):
        site = d.site
        if dist(a.p, site.holder) > 0.6:
            d.decision = 'enter: walking into the passage'
            return step_toward(a, site.holder, min(a.walk * a.move_modifier, dist(a.p, site.holder)), face=site.holder)
        d.decision = 'bait: holding'
        d.done = 'delivered'
        return hold(a)

    def _wait_in_cone(self, d: Delivery, a: AgentView, p: PredatorView):
        w = self.world
        gap = dist(a.p, p.p)
        if gap < 75:
            d.decision = f'resting predator too close ({gap:.0f}); backing off'
            h = steer(a, p, heading_of(sub(a.p, p.p)), w.rects, math.radians(60))
            return self._facing(d, a, p, self._move_heading(a, h, a.walk * a.move_modifier))
        return self._approach_front(d, a, p, gap, 75.0, 'resting')


_holder_paths: dict = {}


class Holder:
    """Stand on a bait slot; approach it from the safe side. Paths are cached per
    agent and slot and re-planned every 3 s or when no progress is made."""

    def __init__(self, world: WorldState):
        self.world = world

    def act(self, a: AgentView, slot, site: Site, avoid=()):
        w = self.world
        if dist(a.p, slot) < 0.8:
            _holder_paths.pop(a.id, None)
            return hold(a), 'holding'
        key = (round(slot[0]), round(slot[1]))
        entry = _holder_paths.get(a.id)
        if entry is None or entry['key'] != key or w.time - entry['t'] > 3.0 or dist(a.p, entry['last']) < 0.5:
            path = plan(w.rects, w.width, w.height, a.p, slot, radius=6.0, avoid=avoid)
            if not path:
                return hold(a), 'holder: no path'
            entry = _holder_paths[a.id] = dict(key=key, t=w.time, wps=path[1:], last=a.p)
        entry['last'] = a.p
        wp = next_waypoint(a.p, entry['wps'], reach=6.0) or slot
        speed = a.walk * a.move_modifier
        if dist(a.p, slot) < 12:
            speed = min(speed, dist(a.p, slot))
        return step_toward(a, wp, speed), f'holder: to slot ({dist(a.p, slot):.0f})'

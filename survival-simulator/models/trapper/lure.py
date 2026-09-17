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
from .world import AGENT_RADIUS, WorldState, AgentView, PredatorView, PRED_CHARGE_RANGE

LEAD_DISTANCE = 125.0       # regulate the gap around this while leading (must stay > 90)
LEAD_MIN = 100.0            # below this a sprinting guide tops up the gap
ATTRACT_MIN = 110.0         # never approach an awake predator closer than this
WAIT_MIN = 125.0            # wait to be noticed only beyond this: its first pivot step eats ~10
GUIDE_RESERVE = 35.0        # abort attracting below this energy
GAZE_OFFSET = 0.12          # alternating facing offset that keeps it zig-zagging straight
RUN_IN = 250.0              # straight run along the axis before the corridor entry
CORRIDOR_MIN_GAP = 100.0    # turn our back only with at least this gap
FLYBY_GAP = 45.0            # gap at which a guide in front of a staffed mouth sprints aside
FLYBY_TICKS = 14            # sprint ticks along the face before the guide is released
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
    max_phase: str = 'ATTRACT'
    trace: list = field(default_factory=list)
    ticks: list = field(default_factory=list)   # last 40 ticks, for post-mortems
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
    if p.resting is not None:
        # the engine wakes it (and lets it act) in the same step once energy passes 100
        return bool(p.resting) and not (p.energy is not None and p.energy + 3.0 > 100.0)
    return 3 <= p.still_ticks < 33


def pred_next_speed(p: PredatorView):
    """Distance the predator will move this tick if it has a target: 15 sprinting, 11 when
    its energy is below 20% of 200. Without energy information fall back to its last speed."""
    if is_resting(p):
        return 0.0
    if p.energy is not None:
        return 15.0 if p.energy >= 40.0 else 11.0
    if p.resting or p.speed < 0.5:
        return 15.0          # just woke: it sprints
    return 15.0 if p.speed > 13.0 else 11.0


def heading_clear(p0, h, probe, rects, step=10.0):
    """We can move ``step`` along ``h`` and keep going to ``probe`` without touching a box.
    Tested from the next position, with the engine's own radius (5, strict): after a
    deflection we often stand within 6 of a wall, where a test from our own position
    would reject every heading."""
    nxt = add(p0, polar(h, step))
    if any(r.contains(nxt, AGENT_RADIUS) for r in rects):
        return False
    return path_clear(nxt, add(p0, polar(h, probe)), AGENT_RADIUS, rects)


def steer(a: AgentView, p: PredatorView, desired, rects, max_dev, probe=70.0, keep_los=True, slow=None):
    """Heading closest to ``desired`` within ``max_dev`` of directly-away-from-p that
    does not run into an obstacle within ``probe`` units and, when ``keep_los``,
    keeps the predator's line of sight to us after both move. When nothing inside the
    cone is clear (our back is to a wall) the clear heading anywhere on the circle with
    the largest away component is used: running along the wall beats standing still."""
    away = heading_of(sub(a.p, p.p))
    best = None
    fallback = None
    p_next = add(p.p, mul(unit(sub(a.p, p.p)), 10.0))
    for k in range(-30, 30):
        h = away + math.pi * k / 30
        inside = abs(wrap(h - away)) <= max_dev + 1e-9
        clear = heading_clear(a.p, h, probe, rects)
        if not clear and inside:
            clear = heading_clear(a.p, h, min(probe, 40.0), rects)     # a short clear run still helps inside the cone
            if not clear:
                continue
            short = True
        else:
            short = False
        if not clear:
            continue
        end = add(a.p, polar(h, probe))
        if inside:
            cost = abs(wrap(h - desired)) + (0.8 if short else 0.0)
            if not path_clear(add(a.p, polar(h, 10.0)), end, 28.0, rects):
                cost += 0.6          # a heading that hugs an obstacle deflects the pivoting predator behind us
            if slow is not None and (slow(end) < 0.6 or slow(add(a.p, polar(h, probe * 0.5))) < 0.6):
                cost += 1.2          # river or swamp ahead: we would crawl at 3-5 per tick
            if keep_los and not los_clear(p_next, add(a.p, polar(h, 12.0)), rects):
                cost += 1.5
            if best is None or cost < best[0]:
                best = (cost, h)
        comp = math.cos(h - away) - (0.3 if slow is not None and slow(end) < 0.6 else 0.0)
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

    def __init__(self, world: WorldState, held=(), baits=()):
        self.world = world
        self.held = set(held)
        self.baits = set(baits)      # agents standing as baits: a predator that targets one is delivered

    # ------------------------------------------------------------------ hearing tap
    @staticmethod
    def _tap_budget(a: AgentView):
        """Enough sprint energy for a tap (approach to hearing range and the run back out)."""
        return a.can_sprint and a.energy - a.max_energy / 5 > 120.0

    def _tap(self, d: Delivery, a: AgentView, p: PredatorView, gap):
        """It has lost us and we cannot get in front of it: run straight at it until it hears us
        (60 through walls), then the ordinary flee/lead takes over. Costs about a second of sprint."""
        w = self.world
        h = heading_of(sub(p.p, a.p))
        if heading_clear(a.p, h, min(gap, 40.0), w.rects):
            goal = p.p
        else:
            path = plan(w.rects, w.width, w.height, a.p, p.p, radius=6.0, slow=w.biome_at)
            goal = path[1] if path and len(path) > 1 else p.p
        d.decision = f'tap: closing to hearing range ({gap:.0f})'
        return step_toward(a, goal, speed_for(a, True))

    # ------------------------------------------------------------------ flyby (gap sites)
    def flyby_side(self, site: Site, a: AgentView):
        """Unit vector along the obstacle face with the longer clear sprint from the flyby point,
        away from other predators when both are clear; None if neither side is clear for 40."""
        w = self.world
        tangent = (-site.normal[1], site.normal[0])
        best = None
        for sgn in (1.0, -1.0):
            side = mul(tangent, sgn)
            run = 0.0
            for length in (40.0, 60.0, 90.0, 120.0):
                end = add(a.p, mul(side, length))
                if 6 <= end[0] <= w.width - 6 and 6 <= end[1] <= w.height - 6 and path_clear(add(a.p, mul(side, 8.0)), end, AGENT_RADIUS, w.rects):
                    run = length
                else:
                    break
            if run < 40.0:
                continue
            threat = min((dist(q.p, add(a.p, mul(side, 60.0))) for q in w.predators if q.pid not in self.held), default=1e9)
            score = run + min(threat, 200.0)
            if best is None or score > best[0]:
                best = (score, side)
        return best[1] if best else None

    def _flyby(self, d: Delivery, a: AgentView, p: PredatorView, gap):
        side = d.flee_heading
        if side is None:
            side = self.flyby_side(d.site, a)
            d.flee_heading = side
        d.stall_ticks += 1
        if side is None:
            d.decision = f'flyby: no clear side, standing (gap {gap:.0f})'
            return self._facing(d, a, p, hold(a))
        if d.stall_ticks >= FLYBY_TICKS:
            d.done = 'delivered'
        d.decision = f'flyby: sprinting aside ({d.stall_ticks}), gap {gap:.0f}'
        return step_toward(a, add(a.p, mul(side, 40.0)), speed_for(a, True))

    def act(self, d: Delivery, a: AgentView, p: PredatorView | None):
        out = self._act(d, a, p)
        if p is not None:
            w = self.world
            d.ticks.append((round(w.time, 1), d.phase[0], round(dist(a.p, p.p)), round(a.energy), round(p.speed, 1),
                            round(a.x), round(a.y), round(p.x), round(p.y), round(math.degrees(p.heading)),
                            int(bool(p.resting)), None if p.energy is None else round(p.energy), d.decision[:70],
                            round(math.degrees(a.heading)),
                            None if out is None else (round(out['move_distance'], 1), round(math.degrees(out['move_direction'])), round(math.degrees(out['turn_angle'])))))
            if len(d.ticks) > 40:
                d.ticks.pop(0)
        return out

    def _act(self, d: Delivery, a: AgentView, p: PredatorView | None):
        w = self.world
        # another loose predator close by: let the society's flee logic act for us
        for q in w.predators:
            if p is not None and q.pid == p.pid or q.pid in self.held or is_resting(q):
                continue
            dq = dist(q.p, a.p)
            if dq < (200 if d.phase == 'ATTRACT' else 150) and d.phase not in ('FRONT', 'ENTER'):
                # a second loose predator this close ends the delivery while a flee can still
                # succeed; deferring until it is on top of us cost most guides their lives
                d.done = 'failed:other_predator'
                d.decision = f'aborting: predator {q.pid} at {dq:.0f}'
                return DEFER
        if p is None:
            d.lost_ticks += 1
            if d.lost_ticks > 30:
                d.done = 'failed:predator_lost'
            d.decision = 'predator unknown; holding'
            return hold(a)
        d.lost_ticks = 0
        order = ('ATTRACT', 'OPEN', 'LEAD', 'CORRIDOR', 'FRONT', 'FLYBY', 'ENTER')
        if order.index(d.phase) > order.index(d.max_phase):
            d.max_phase = d.phase
        target = predator_target(w, p)
        following = target == a.id
        gap = dist(a.p, p.p)
        if target is not None and target in self.baits and d.phase in ('CORRIDOR', 'FRONT', 'FLYBY') and not d.become_bait:
            # it has taken the bait: we are free (keep sprinting aside if we were)
            d.done = 'delivered'
            d.decision = f'delivered: it targets bait {target}'
            if d.phase == 'FLYBY' and d.flee_heading is not None:
                return step_toward(a, add(a.p, mul(d.flee_heading, 40.0)), speed_for(a, True))
            return self._facing(d, a, p, hold(a))
        if int(round(w.time * 10)) % 10 == 0:
            off = abs(wrap(heading_of(sub(a.p, p.p)) - p.heading))
            d.trace.append((round(w.time, 1), d.phase[0], round(gap), target, int(los_clear(p.p, a.p, w.rects)), round(a.energy),
                            round(p.speed, 1), round(math.degrees(off))))
            if len(d.trace) > 150:
                d.trace.pop(0)
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
                if d.stall_ticks > 40:
                    d.phase = 'ATTRACT'
                    d.stall_ticks = 0
                    d.event(w.time, 'lost_attention', target=target)
                    return self._attract(d, a, p, target, gap)
                if target is None and gap >= PRED_CHARGE_RANGE:
                    # it lost us. Facing away or walking off, it will not find us again by itself:
                    # tap it (run into hearing range) when we can afford the sprint
                    off = abs(wrap(heading_of(sub(a.p, p.p)) - p.heading))
                    receding = off > math.radians(75) or (d.stall_ticks > 6 and p.speed > 0.5)
                    if receding and gap <= 260 and self._tap_budget(a):
                        return self._tap(d, a, p, gap)
                    # usually behind an obstacle corner we just rounded: stay put and visible
                    # instead of walking on, it comes round within a few ticks
                    if not los_clear(p.p, a.p, w.rects):
                        d.decision = f'lead: waiting for it to come round the corner ({d.stall_ticks})'
                        return self._facing(d, a, p, hold(a))
                    # clear line but outside its cone: the nearest point of its forward ray, at least 100
                    # from it, reached at a sprint if we can afford it (it turns away fast once lost)
                    ray = (math.cos(p.heading), math.sin(p.heading))
                    along = max(100.0, dot(sub(a.p, p.p), ray))
                    goal = add(p.p, mul(ray, along))
                    if dist(a.p, goal) > 2.0 and los_clear(p.p, goal, w.rects):
                        d.decision = f'lead: re-entering its cone ({d.stall_ticks})'
                        return self._facing(d, a, p, step_toward(a, goal, speed_for(a, a.energy > 150)))
                    d.decision = f'lead: out of its sight, holding ({d.stall_ticks})'
                    return self._facing(d, a, p, hold(a))
            else:
                d.stall_ticks = 0
            if gap < PRED_CHARGE_RANGE and (following or target is not None or gap < 62):
                # inside its charge range while it chases us (or is about to hear us): reopen the gap
                d.phase = 'OPEN'
                d.event(w.time, 'too_close', gap=round(gap, 1))
                return self._open(d, a, p, gap)
            if gap < PRED_CHARGE_RANGE:
                return self._tap(d, a, p, gap) if self._tap_budget(a) else self._facing(d, a, p, hold(a))
            return self._lead(d, a, p, gap)
        if d.phase == 'CORRIDOR':
            return self._corridor(d, a, p, gap, following)
        if d.phase == 'FRONT':
            return self._front(d, a, p, gap)
        if d.phase == 'FLYBY':
            return self._flyby(d, a, p, gap)
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
        speed = (a.sprint_speed if self._tap_budget(a) else a.walk) * a.move_modifier
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
            h = steer(a, p, heading_of(sub(a.p, p.p)), w.rects, math.radians(60), slow=w.biome_at)
            return self._facing(d, a, p, self._move_heading(a, h, speed_for(a, a.energy > 150)))
        if bearing < 0.35 and WAIT_MIN <= gap <= 240:
            d.decision = f'{label}: in its cone at {gap:.0f}, waiting'
            return self._facing(d, a, p, hold(a))
        goal, k = self._intercept_point(a, p, 140.0)
        if goal is None or (d.phase == 'ATTRACT' and k > 15):
            d.stall_ticks += 1
            if d.phase == 'ATTRACT' and gap <= 260 and self._tap_budget(a) and not is_resting(p):
                return self._tap(d, a, p, gap)
            if d.phase == 'ATTRACT' and d.stall_ticks > 10:
                d.done = 'failed:cannot_intercept'
                d.decision = 'attract: cannot intercept'
            else:
                d.decision = f'{label}: cannot intercept yet; waiting'
            return self._facing(d, a, p, hold(a))
        d.stall_ticks = 0
        if not d.waypoints or dist(d.waypoints[-1], goal) > 25:
            path = plan(w.rects, w.width, w.height, a.p, goal, radius=6.0, avoid=[(p.p, berth)], slow=w.biome_at)
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
        """Reopen the gap. Inside 90 it charges straight at 15/tick: only a sprint (20) gains
        on it, so sprint directly away while we can; in its pivot band walking away holds the
        gap and a sprint reopens it 9/tick. Without a sprint left the society's flee (which
        knows refuges and joint threats) takes over."""
        w = self.world
        d.defer_ticks += 1
        if d.defer_ticks > 150:
            d.done = 'failed:cannot_open'
        away = heading_of(sub(a.p, p.p))
        if not a.can_sprint:
            d.decision = f'open: society flees ({gap:.0f})'
            return DEFER
        max_dev = math.radians(30 if gap < PRED_CHARGE_RANGE else 45)
        h = steer(a, p, away, w.rects, max_dev, probe=90.0, keep_los=gap >= PRED_CHARGE_RANGE, slow=w.biome_at)
        d.decision = f'open: sprinting away ({gap:.0f}, dev {math.degrees(wrap(h - away)):.0f})'
        return self._facing(d, a, p, self._move_heading(a, h, a.sprint_speed * a.move_modifier))

    def _lead_path(self, a: AgentView, site: Site):
        """Waypoints to the corridor entry: join the approach axis at about our own
        distance from the trap (never further out than the run-in point, never
        inside the corridor), then walk the axis to the corridor start."""
        w = self.world
        out = dot(sub(a.p, site.front_mid), site.normal)
        from .sites import CORRIDOR
        entry_out = max(CORRIDOR + 30.0, min(CORRIDOR + RUN_IN, out - 40.0))
        entry = add(site.front_mid, mul(site.normal, entry_out))
        # a predator ~125 behind keeps line of sight round a corner only if we pass it at a
        # good distance (chord of the arc must clear the corner): try 55, then 32, then 16
        path = None
        for radius in (55.0, 32.0, 16.0):
            path = plan(w.rects, w.width, w.height, a.p, entry, radius=radius, slow=w.biome_at)
            if path is not None:
                break
        if path is None:
            path = [a.p, entry]
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
        sprinting_pred = pred_next_speed(p) >= 14.0
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
            h = steer(a, p, desired, w.rects, max(dev_s, dev), slow=w.biome_at)
            d.decision = f'lead: sprint-steer, gap {gap:.0f}, needed {math.degrees(needed):.0f}, dev {math.degrees(wrap(h - away)):.0f}'
            return self._facing(d, a, p, self._move_heading(a, h, sp))
        h = steer(a, p, desired, w.rects, dev, slow=w.biome_at)
        if gap < LEAD_MIN and a.can_sprint and (sprinting_pred or gap < LEAD_MIN - 8):
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
        flyby = site.kind == 'gap' and not d.become_bait
        if dist(a.p, goal) < 2.0:
            if flyby:
                # in front of a staffed mouth: face it until it is close, then sprint aside;
                # if it kills us here it takes the bait anyway
                if gap > FLYBY_GAP and not is_resting(p):
                    d.decision = f'flyby point: waiting for it ({gap:.0f})'
                    return self._facing(d, a, p, hold(a))
                if is_resting(p):
                    d.decision = f'flyby point: it rests at {gap:.0f}'
                    return self._facing(d, a, p, hold(a))
                d.phase = 'FLYBY'
                d.stall_ticks = 0
                d.flee_heading = None
                d.event(w.time, 'flyby', gap=round(gap, 1))
                return self._flyby(d, a, p, gap)
            d.phase = 'ENTER' if d.become_bait else 'FRONT'
            d.event(w.time, d.phase.lower(), gap=round(gap, 1))
            return self._enter(d, a, p) if d.become_bait else self._front(d, a, p, gap)
        speed = a.walk * a.move_modifier
        if is_resting(p):
            speed = 0.0
        elif gap > 140:
            speed *= 0.5
        if flyby:
            # walk the axis facing it (it pivots and closes only 0.6/tick); sprint if it gets close
            if gap < 60 and a.can_sprint and dist(a.p, goal) > 20:
                speed = speed_for(a, True)
            d.decision = f'corridor: gap {gap:.0f}, {dist(a.p, goal):.0f} to the flyby point'
            act = step_toward(a, goal, speed) if speed > 0 else hold(a)
            return self._facing(d, a, p, act)
        # wall: walk down the axis with our back to it
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
            h = steer(a, p, heading_of(sub(a.p, p.p)), w.rects, math.radians(60), slow=w.biome_at)
            return self._facing(d, a, p, self._move_heading(a, h, a.walk * a.move_modifier))
        # stand on its forward ray where it will see us on waking: pick the nearest such point
        # with a clear line of sight (an obstacle between would make it wake blind and wander)
        best = None
        for rng in (135.0, 120.0, 150.0, 165.0, 105.0):
            for off in (0.0, 0.15, -0.15, 0.3, -0.3):
                q = add(p.p, polar(p.heading + off, rng))
                if not (5 <= q[0] <= w.width - 5 and 5 <= q[1] <= w.height - 5):
                    continue
                if any(r.contains(q, 6.0) for r in w.rects) or not los_clear(p.p, q, w.rects):
                    continue
                c = dist(a.p, q) + 60.0 * abs(off)
                if best is None or c < best[0]:
                    best = (c, q)
        if best is None:
            return self._approach_front(d, a, p, gap, 75.0, 'resting')
        q = best[1]
        if dist(a.p, q) > 4.0:
            d.decision = f'resting: moving to a visible spot on its ray ({dist(a.p, q):.0f} away)'
            path = plan(w.rects, w.width, w.height, a.p, q, radius=6.0, avoid=[(p.p, 75.0)], slow=w.biome_at)
            wp = path[1] if path and len(path) > 1 else q
            return self._facing(d, a, p, step_toward(a, wp, speed_for(a, a.energy > 180 and dist(a.p, q) > 60)))
        d.decision = f'resting: waiting on its ray at {gap:.0f}'
        return self._facing(d, a, p, hold(a))


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
        if entry is not None and entry['key'] == key and entry.get('wps') is None and w.time - entry['t'] < 2.0:
            return hold(a), 'holder: no path'          # do not search again every tick
        if entry is None or entry['key'] != key or entry.get('wps') is None or w.time - entry['t'] > 3.0 or dist(a.p, entry['last']) < 0.5:
            path = plan(w.rects, w.width, w.height, a.p, slot, radius=6.0, avoid=avoid, slow=w.biome_at)
            if not path:
                _holder_paths[a.id] = dict(key=key, t=w.time, wps=None, last=a.p)
                return hold(a), 'holder: no path'
            entry = _holder_paths[a.id] = dict(key=key, t=w.time, wps=path[1:], last=a.p)
        entry['last'] = a.p
        wp = next_waypoint(a.p, entry['wps'], reach=6.0) or slot
        speed = a.walk * a.move_modifier
        if dist(a.p, slot) < 12:
            speed = min(speed, dist(a.p, slot))
        return step_toward(a, wp, speed), f'holder: to slot ({dist(a.p, slot):.0f})'

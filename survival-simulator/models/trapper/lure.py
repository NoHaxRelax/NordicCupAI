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

from .geometry import free_point, add, sub, mul, dot, dist, unit, wrap, heading_of, polar, path_clear, los_clear
from .motion import action, step_toward, hold, speed_for, next_waypoint
from .paths import plan, path_length
from .predator_model import PredState, observed_agents
from .sites import Site
from . import predator_model as pm
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
LEASH_GAP = 50.0            # leash lead: gap kept in front of a charging predator (kill radius 15)
LEASH_HEAR = 55.0           # inside this it hears us through anything and never loses us
LEASH_RUN_IN = 200.0        # straight run down the axis before the mouth (predator lines up behind)
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
    leash: bool = False           # close-range lead (guide keeps 40-60 behind it, charge mode)
    path: list | None = None
    path_t: float = -1e9
    stage: str = 'approach'       # leash: 'approach' (to the run-in start) or 'run' (down the axis)
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

    # ------------------------------------------------------------------ leash lead
    def _leash_path(self, d: Delivery, a: AgentView, p: PredatorView):
        """Waypoints: around obstacles to the run-in start (on the axis, LEASH_RUN_IN out), then
        straight down the axis to the run-in end and the goal. The straight run puts the predator
        on the axis behind us, so it arrives at the mouth and not at its side. Other predators, the
        one behind us and the far mouth are kept clear of."""
        w = self.world
        site = d.site
        crowded = not d.become_bait and any(q.pid in self.held and dist(q.p, site.front_mid) < 60 for q in w.predators)
        goal = site.holder if d.become_bait else (add(site.front_mid, mul(site.normal, 62.0)) if crowded else site.front)
        run_far = add(site.front_mid, mul(site.normal, LEASH_RUN_IN))
        run_end = add(site.front_mid, mul(site.normal, 60.0))
        out = dot(sub(a.p, site.front_mid), site.normal)
        lateral = abs(dot(sub(a.p, site.front_mid), (-site.normal[1], site.normal[0])))
        behind = abs(wrap(heading_of(sub(p.p, a.p)) - heading_of(site.normal))) < 0.7
        ahead = abs(wrap(heading_of(sub(p.p, a.p)) - heading_of(mul(site.normal, -1.0)))) < 1.0
        # stages with hysteresis: the run down the axis starts at the run-in point with the
        # predator behind us and is only abandoned if it gets between us and the mouth
        if d.stage == 'run':
            if ahead and out > 40.0:
                d.stage = 'approach'
            elif lateral > 30.0 and out > 40.0:
                d.stage = 'approach'
        if d.stage != 'run' and lateral < 18.0 and 0.0 <= out <= LEASH_RUN_IN + 25.0 and (behind or out < 40.0):
            d.stage = 'run'
        if d.stage == 'run':
            return [run_end, goal] if out > 62.0 and dist(run_end, goal) > 5.0 else [goal]
        avoid = [(q.p, 60.0) for q in w.recent_predators() if q.pid != p.pid and q.pid not in self.held
                 and predator_target(w, q) != a.id]
        avoid.append((p.p, 70.0))
        if site.far_mouth is not None:
            avoid.append((site.far_mouth, 110.0))     # it must not get sucked into the far mouth
        avoid.append((site.front_mid, 70.0))          # approach the mouth only along the axis
        best = None
        for radius in (12.0, 8.0, 6.0):
            path = plan(w.rects, w.width, w.height, a.p, run_far, radius=radius, avoid=avoid, slow=w.biome_at)
            if path is None:
                continue
            length = path_length(path) * (1.0 if radius >= 12 else (1.08 if radius >= 8 else 1.2))
            if best is None or length < best[0]:
                best = (length, path)
        head = best[1][1:] if best is not None else [run_far]
        return head + [run_end, goal]

    def _charge_step(self, p: PredatorView, q, speed):
        """Where a predator that charges point q ends up this tick (engine: turn at most 0.3 rad
        toward it, move ``speed`` along the new heading, deflect around obstacles)."""
        ang = wrap(heading_of(sub(q, p.p)) - p.heading)
        turn = max(-0.3, min(0.3, ang * 0.5)) if abs(ang) > 0.05 else ang
        st = pm.PredState(p.x, p.y, p.heading, energy=200.0, resting=False)
        w = self.world
        lrects = pm.local_rects(w.rects, p.x, p.y)
        st2 = pm.move(st, speed, turn, w.biome_at(p.p), lrects, w.width, w.height)
        return (st2.x, st2.y)

    def _pivot_step(self, p: PredatorView, q, q_heading, speed):
        """Where a predator that sees an agent at q (facing it, so |rel_dir| <= pi/2) and is at
        least 90 away ends up this tick: it moves ``speed`` at 45 degrees off the line, on the side
        opposite the agent's look direction, then turns to face the agent. Returns (pos, heading)."""
        ang = wrap(heading_of(sub(q, p.p)) - p.heading)
        rel = wrap(heading_of(sub(p.p, q)) - q_heading)
        pivot = -1.0 if rel > 0 else (1.0 if rel < 0 else 0.0)
        move_dir = ang + pivot * math.pi / 4
        st = pm.PredState(p.x, p.y, p.heading, energy=200.0, resting=False)
        w = self.world
        lrects = pm.local_rects(w.rects, p.x, p.y)
        st2 = pm.move(st, speed, move_dir, w.biome_at(p.p), lrects, w.width, w.height)
        return (st2.x, st2.y), heading_of(sub(q, (st2.x, st2.y)))

    def _walk_choose(self, a: AgentView, p: PredatorView, desired, gap, v_pred, gaze_sign, others=()):
        """Walking guide (no sprint): stay in its pivot band (90-240, facing it, line of sight),
        as close to ``desired`` as the predicted gap allows. Returns (heading, real_speed, gap2)."""
        w = self.world
        away = heading_of(sub(a.p, p.p))
        floor = 104.0 if gap >= 106.0 else max(gap - 1.0, 88.0)
        ceil = 235.0
        walk = a.walk * a.move_modifier
        speeds = sorted({0.0, walk * 0.5, walk})
        headings = [away + math.radians(10) * k for k in range(-9, 10)] + [desired]
        best = None; nearest = None
        for real in speeds:
            for h in (headings if real > 0.05 else [desired]):
                if real > 0.05 and not heading_clear(a.p, h, max(real, 12.0) + 4.0, w.rects):
                    continue
                q = add(a.p, polar(h, real)) if real > 0.05 else a.p
                face = heading_of(sub(p.p, q)) + GAZE_OFFSET * gaze_sign
                pn, ph = self._pivot_step(p, q, face, v_pred)
                g2 = dist(q, pn)
                seen = los_clear(pn, q, w.rects)
                d_other = 1e9
                for o in others:
                    on = self._charge_step(o, q, pred_next_speed(o) * w.biome_at(o.p)) if predator_target(w, o) == a.id else add(o.p, (o.vx, o.vy))
                    d_other = min(d_other, dist(q, on))
                dev = abs(wrap(h - desired)) if real > 0.05 else math.pi / 2
                progress = real * math.cos(dev) if real > 0.05 else 0.0
                ok = floor <= g2 <= ceil and seen and d_other >= 95.0
                if ok:
                    cost = -progress + 0.15 * abs(g2 - 128.0)
                    if best is None or cost < best[0]:
                        best = (cost, h, real, g2)
                miss = (floor - g2) if g2 < floor else (g2 - ceil if g2 > ceil else 0.0)
                miss += (0.0 if seen else 30.0) + max(0.0, 95.0 - d_other) * 2.0
                if nearest is None or miss < nearest[0]:
                    nearest = (miss, h, real, g2)
        if best is not None:
            return best[1], best[2], best[3]
        if nearest is not None:
            return nearest[1], nearest[2], nearest[3]
        return away, 0.0, gap

    def _leash_choose(self, a: AgentView, p: PredatorView, desired, gap, v_pred, gap_target, cap, others=(), run=False, ceil=None):
        """Joint choice of heading (36 directions plus ``desired``) and speed so that the predicted
        gap after both move lands in the window [floor, hearing] and every other nearby predator
        stays beyond 30; among feasible moves prefer the one closest to ``desired`` and cheapest
        in energy. Returns (heading, real_speed, gap2)."""
        w = self.world
        away = heading_of(sub(a.p, p.p))
        close = gap < 35.0
        floor = max(gap_target - 10.0, 26.0) if gap >= gap_target - 10.0 else max(gap - 2.0, 18.0)
        if run:
            # on the run down the axis the predator swings in behind us by itself; do not sidestep
            # to keep the gap, accept it closer for a few ticks (it cannot kill beyond 15)
            floor = min(floor, 28.0) if gap >= 30.0 else max(gap - 2.0, 18.0)
        if close:
            floor = gap + 2.0          # too close: every move must open the gap
        ceil = LEASH_HEAR - 2.0 if ceil is None else ceil
        p_mod = w.biome_at(p.p)
        speeds = sorted({0.0, 5.0, 10.0, min(v_pred, cap), min(v_pred + 4.0, cap), cap})
        if run:
            headings = [desired + math.radians(12) * k for k in range(-3, 4)]
        else:
            headings = [away + math.radians(10) * k for k in range(-18, 18)] + [desired]
        # other predators: where each will be after this tick (charging us if it targets us)
        oth = []
        for q in others:
            tq = predator_target(w, q)
            sleeping = bool(q.resting) if q.resting is not None else is_resting(q)
            oth.append((q, tq == a.id and not sleeping, 0.0 if sleeping else pred_next_speed(q) * w.biome_at(q.p)))
        best = None; nearest = None
        for sp in speeds:
            real = sp * a.move_modifier
            cost_e = (sp * 0.05 if sp <= 10.0 else 0.5 + (sp - 10.0) * 0.5)
            for h in (headings if real > 0.05 else [desired]):
                if real > 0.05 and not heading_clear(a.p, h, max(real, 6.0) + 2.0, w.rects):
                    continue
                q = add(a.p, polar(h, real)) if real > 0.05 else a.p
                pn = self._charge_step(p, q, v_pred)
                g2 = dist(q, pn)
                d_other = 1e9
                for (o, chasing_us, v_o) in oth:
                    on = self._charge_step(o, q, v_o) if chasing_us else (o.p if v_o == 0.0 else add(o.p, (o.vx, o.vy)))
                    d_other = min(d_other, dist(q, on))
                slow_ahead = real > 0.05 and w.biome_at(q) < min(0.9, p_mod - 0.05)
                dev = abs(wrap(h - desired)) if real > 0.05 else math.pi / 2
                progress = real * math.cos(dev) if real > 0.05 else 0.0
                g2_eff = g2 - (0.8 * v_pred if real <= 0.05 else 0.0)     # standing: it keeps coming
                if floor <= g2_eff and g2 <= ceil and d_other >= 30.0 and not slow_ahead:
                    if close:
                        cost = -g2 * 3.0 - progress
                    else:
                        cost = -progress + 2.0 * cost_e + 0.3 * abs(g2 - gap_target) + (0.0 if d_other > 60 else (60 - d_other) * 0.4)
                    if best is None or cost < best[0]:
                        best = (cost, h, real, g2)
                miss = (floor - g2_eff) if g2_eff < floor else (g2 - ceil if g2 > ceil else 0.0)
                miss += max(0.0, 30.0 - d_other) * 2.0 + (8.0 if slow_ahead else 0.0)
                if gap < 45.0:
                    miss = -min(g2, d_other)        # in trouble: the move that ends farthest from any predator
                if nearest is None or miss < nearest[0] or (miss == nearest[0] and cost_e < nearest[4]):
                    nearest = (miss, h, real, g2, cost_e)
        if best is not None:
            return best[1], best[2], best[3]
        if nearest is not None:
            return nearest[1], nearest[2], nearest[3]
        return away, 0.0, gap

    def _leash(self, d: Delivery, a: AgentView, p: PredatorView):
        """Close-range lead. With our back to it the predator charges straight at us (15 per tick,
        turning at most 0.3 rad per tick) and, inside 60, hears us through walls; so we walk our
        planned path at its own speed, 50 ahead of it, sprinting only to correct. Facing and line
        of sight do not matter; other predators that join the chase are welcome."""
        w = self.world
        site = d.site
        gap = dist(a.p, p.p)
        target = predator_target(w, p)
        following = target == a.id
        d.phase = 'LEASH'
        if int(round(w.time * 10)) % 10 == 0:
            d.trace.append((round(w.time, 1), 'S', round(gap), target, 1, round(a.energy), round(p.speed, 1), 0))
            if len(d.trace) > 150:
                d.trace.pop(0)
        away = heading_of(sub(a.p, p.p))
        face_away = wrap(away - a.heading)

        crowded = not d.become_bait and any(q.pid in self.held and dist(q.p, site.front_mid) < 60 for q in w.predators)
        goal = site.holder if d.become_bait else (add(site.front_mid, mul(site.normal, 62.0)) if crowded else site.front)
        at_goal = dist(a.p, goal) < (1.0 if d.become_bait else 2.0)
        # inside the passage nothing can reach us: just settle on the holder point
        out = dot(sub(a.p, site.front_mid), site.normal)
        lateral = abs(dot(sub(a.p, site.front_mid), (-site.normal[1], site.normal[0])))
        if d.become_bait and not at_goal and -site.length + 1.0 < out < -1.0 and lateral < site.thickness / 2 - 1.0:
            d.decision = f'leash: inside the passage, settling on the holder ({dist(a.p, goal):.0f})'
            return step_toward(a, goal, min(a.walk * a.move_modifier, dist(a.p, goal)), face=site.front_mid)

        # ---- endgame
        if at_goal and d.become_bait:
            if target != a.id and gap > 55 and not bool(p.resting):
                # it lost us as we went in: stand in the mouth (it cannot enter) until it hears us
                q = add(site.front_mid, mul(site.normal, -2.0))
                d.decision = f'leash: bait, it lost us ({gap:.0f}); stepping to the mouth'
                return step_toward(a, q, min(a.walk * a.move_modifier, dist(a.p, q)), face=site.front_mid)
            d.decision = 'leash: inside the passage, holding as bait'
            d.done = 'delivered'
            return hold(a)
        if at_goal and not a.can_sprint and not d.become_bait:
            d.decision = f'leash: no sprint; standing at the mouth for it to take the bait ({gap:.0f})'
            return action(a.id, 0.0, 0.0, face_away)
        if at_goal:
            if gap <= FLYBY_GAP + 10 and not is_resting(p):
                d.phase = 'FLYBY'
                d.stall_ticks = 0
                d.flee_heading = None
                d.event(w.time, 'flyby', gap=round(gap, 1))
                return self._flyby(d, a, p, gap)
            if not following and gap > 80 and not bool(p.resting):
                d.decision = f'leash: at the flyby point but it is gone ({gap:.0f}); hooking'
                return step_toward(a, p.p, speed_for(a, gap < 130 and a.energy > 320))
            d.decision = f'leash: at the flyby point, waiting for it ({gap:.0f})'
            return action(a.id, 0.0, 0.0, face_away)

        # ---- sprint budget low or gone: lead from its pivot band instead (facing it, 96-235 away,
        # in sight); open the gap to it first while a sprint is still possible
        resting_now = bool(p.resting) if p.resting is not None else (3 <= p.still_ticks < 30)
        waking_now = bool(p.resting) and ((p.energy is not None and p.energy > 100.0) or (p.energy is None and p.still_ticks >= 30))
        if (not a.can_sprint or a.energy < 118.0) and d.become_bait and dist(a.p, site.holder) < 60 and not at_goal:
            d.decision = f'leash: no sprint, walking straight into the passage ({dist(a.p, site.holder):.0f})'
            return step_toward(a, site.holder, min(a.walk * a.move_modifier, dist(a.p, site.holder)), face=site.front_mid)
        if (not a.can_sprint or a.energy < 118.0) and not (d.become_bait and dist(a.p, site.holder) < 60):
            if gap < 92.0 and a.can_sprint and not resting_now:
                cap = a.sprint_speed
                h, real, g2 = self._leash_choose(a, p, away, gap, pred_next_speed(p) * w.biome_at(p.p), 110.0, cap, [], ceil=135.0)
                d.decision = f'leash: budget low, opening the gap for the walk lead ({gap:.0f}->{g2:.0f})'
                return self._move_heading(a, h, real) if real > 0.05 else action(a.id, 0.0, 0.0, face_away)
            if d.path is None or w.time - d.path_t > 1.5:
                d.path = self._leash_path(d, a, p)
                d.path_t = w.time
            wp = next_waypoint(a.p, d.path, reach=8.0) or goal
            return self._walk_lead(d, a, p, gap, wp, goal, waking_now)

        # ---- it already wants the bait: we are done, get out of its way
        if target is not None and target in self.baits and not d.become_bait:
            # it has taken the bait: we are delivered; keep sprinting clear (FLYBY continues the
            # escape for whoever drives us next)
            d.done = 'delivered'
            d.phase = 'FLYBY'
            d.stall_ticks = 0
            side = self.flyby_side(site, a, p) or (-site.normal[1], site.normal[0])
            d.flee_heading = side
            d.decision = f'leash: it targets bait {target}; sprinting clear'
            return step_toward(a, add(a.p, mul(side, 40.0)), speed_for(a, True))

        # ---- it rests: it cannot kill, and it wakes deaf to anything beyond 60. Sit at ~42 and
        # do not move until the tick it wakes on (energy above 100 at its step), then keep pace.
        resting = bool(p.resting) if p.resting is not None else (3 <= p.still_ticks < 30)
        waking = bool(p.resting) and ((p.energy is not None and p.energy > 100.0) or (p.energy is None and p.still_ticks >= 30))
        if resting and not waking:
            others_near = [q for q in w.recent_predators() if q.pid != p.pid and q.pid not in self.held
                           and not (bool(q.resting) if q.resting is not None else is_resting(q)) and dist(q.p, a.p) < 120]
            if others_near:
                # another predator about: keep clear of it while staying inside 55 of the sleeper
                cap = a.sprint_speed if a.can_sprint else a.walk
                h, real, g2 = self._leash_choose(a, p, away, gap, 0.0, 42.0, cap, others_near, ceil=75.0)
                d.decision = f'leash: it rests at {gap:.0f}, dodging predator {others_near[0].pid} ({dist(others_near[0].p, a.p):.0f})'
                return self._move_heading(a, h, real) if real > 0.05 else action(a.id, 0.0, 0.0, face_away)
            if gap > 48:
                d.decision = f'leash: it rests at {gap:.0f}, closing in'
                return step_toward(a, p.p, min(a.walk * a.move_modifier, gap - 42.0))
            if gap < 36:
                d.decision = f'leash: it rests at {gap:.0f}, stepping back'
                h = steer(a, p, away, w.rects, math.radians(60), keep_los=False, slow=w.biome_at)
                return self._move_heading(a, h, min(a.walk * a.move_modifier, 42.0 - gap))
            d.decision = f'leash: it rests at {gap:.0f}, waiting'
            return action(a.id, 0.0, 0.0, face_away)
        if waking:
            following = True     # it hears us the moment it acts; the predicted gap keeps us inside 60

        # ---- hook: it is not chasing us yet
        if not following:
            if gap > LEASH_HEAR:
                bearing = abs(wrap(heading_of(sub(a.p, p.p)) - p.heading))
                if bearing < math.radians(50) and gap < 240 and los_clear(p.p, a.p, w.rects) and not bool(p.resting):
                    # it is coming our way and will see us: let it come (free)
                    d.decision = f'leash: in its sight, letting it come ({gap:.0f})'
                    return action(a.id, 0.0, 0.0, face_away)
                if target is not None and target in w.agents and dist(w.agents[target].p, p.p) < gap - 10.0:
                    # it chases someone nearer: come to just inside that distance, not onto it
                    want = max(38.0, dist(w.agents[target].p, p.p) - 8.0)
                    d.decision = f'leash: undercutting agent {target} to {want:.0f} ({gap:.0f})'
                    return step_toward(a, p.p, min(speed_for(a, a.energy > 260), max(gap - want, 0.0)))
                others_near = [q for q in w.recent_predators() if q.pid != p.pid and q.pid not in self.held
                               and not (bool(q.resting) if q.resting is not None else is_resting(q)) and dist(q.p, a.p) < 140]
                if others_near:
                    cap = a.sprint_speed if a.can_sprint else a.walk
                    toward = heading_of(sub(p.p, a.p))
                    # a hook is a move toward it: use the chooser with a wide window and the others kept off
                    h, real, g2 = self._leash_choose(a, p, toward, gap, 0.0, 45.0, cap, others_near)
                    d.decision = f'leash: hooking it ({gap:.0f}) around predator {others_near[0].pid}'
                    return self._move_heading(a, h, real) if real > 0.05 else action(a.id, 0.0, 0.0, face_away)
                goal_h = p.p
                if not heading_clear(a.p, heading_of(sub(p.p, a.p)), min(gap, 40.0), w.rects):
                    path = plan(w.rects, w.width, w.height, a.p, p.p, radius=6.0, slow=w.biome_at)
                    goal_h = path[1] if path and len(path) > 1 else p.p
                # a sprint costs 5.5 per tick: only when rich, for the last stretch or when it walks
                # away from us (it is faster than our walk)
                sprint = a.energy > 260 and gap < 260 and (gap < 130 or bearing > math.radians(90))
                d.decision = f'leash: hooking it ({gap:.0f}){" at a sprint" if sprint else ""}'
                return step_toward(a, goal_h, speed_for(a, sprint))
            if target is not None and target in w.agents and dist(w.agents[target].p, p.p) < gap:
                want = max(38.0, dist(w.agents[target].p, p.p) - 8.0)
                if gap > want + 2.0:
                    d.decision = f'leash: undercutting agent {target} to {want:.0f} ({gap:.0f})'
                    return step_toward(a, p.p, min(a.walk * a.move_modifier, gap - want))
                if gap < 40.0:
                    d.decision = f'leash: nearer than agent {target} but too close ({gap:.0f}); stepping back'
                    h = steer(a, p, away, w.rects, math.radians(50), probe=40.0, keep_los=False, slow=w.biome_at)
                    return self._move_heading(a, h, min(a.walk * a.move_modifier, 40.0 - gap + 2.0))
                d.decision = f'leash: nearer than agent {target}, waiting ({gap:.0f})'
                return action(a.id, 0.0, 0.0, face_away)
            d.decision = f'leash: in its hearing, waiting for it to turn ({gap:.0f})'
            return action(a.id, 0.0, 0.0, face_away)

        # ---- path
        if d.path is None or w.time - d.path_t > 1.5:
            d.path = self._leash_path(d, a, p)
            d.path_t = w.time
        wp = next_waypoint(a.p, d.path, reach=8.0) or goal


        # ---- move: heading and speed chosen jointly on the predicted gap after both move
        v_pred = pred_next_speed(p) * w.biome_at(p.p)
        gap_target = LEASH_GAP
        cap = a.sprint_speed if a.can_sprint else a.walk
        desired = heading_of(sub(wp, a.p))
        others = [q for q in w.recent_predators() if q.pid != p.pid and dist(q.p, a.p) < 140
                  and (q.pid in self.held or not (bool(q.resting) if q.resting is not None else is_resting(q)))]
        h, real, g2 = self._leash_choose(a, p, desired, gap, v_pred, gap_target, cap, others, run=(d.stage == 'run'))
        if wp is goal or dist(wp, goal) < 1e-6:
            real = min(real, dist(a.p, goal))
        d.decision = (f'leash: gap {gap:.0f}->{g2:.0f}, v {real:.0f} (pred {v_pred:.0f}), dev {math.degrees(wrap(h - away)):.0f}, '
                      f'wp {tuple(round(v) for v in wp)}, {len(d.path)} wps, e {a.energy:.0f}')
        if real <= 0.05:
            return action(a.id, 0.0, 0.0, face_away)
        return self._move_heading(a, h, real)

    def _walk_lead(self, d: Delivery, a: AgentView, p: PredatorView, gap, wp, goal, waking):
        """Walking guide. Facing it beyond 90 it pivots (closes 10.6 per tick sprinting, 7.8
        walking) so a walker holds or gains the gap and can steer up to 39 degrees while it walks;
        inside 90 it charges at 15 and a walker cannot escape, so the band is kept at all cost.
        It must see us: within 235, in its cone (it re-faces us every tick) with line of sight."""
        w = self.world
        site = d.site
        v_pred = pred_next_speed(p) * w.biome_at(p.p)
        d.gaze = -d.gaze
        target = predator_target(w, p)
        following = target == a.id
        walk = a.walk * a.move_modifier
        others = [q for q in w.recent_predators() if q.pid != p.pid and dist(q.p, a.p) < 200
                  and (q.pid in self.held or not (bool(q.resting) if q.resting is not None else is_resting(q)))]
        resting = bool(p.resting) if p.resting is not None else (3 <= p.still_ticks < 30)

        def spot_on_ray(rng_out):
            best = None
            for strict in (True, False):
                for off in (0.0, 0.2, -0.2, 0.4, -0.4):
                    q = add(p.p, polar(p.heading + off, rng_out))
                    if not free_point(q, AGENT_RADIUS + 1.0, w.rects, w.width, w.height) or not los_clear(p.p, q, w.rects):
                        continue
                    if strict and (not free_point(q, 28.0, w.rects, w.width, w.height)
                                   or min(q[0], q[1], w.width - q[0], w.height - q[1]) < 90.0):
                        continue        # a pocket or the map edge: a walker cannot get out of there
                    c = dist(a.p, q) + 40.0 * abs(off)
                    if best is None or c < best[0]:
                        best = (c, q)
                if best is not None:
                    break
            return best[1] if best else None

        if resting and not waking:
            # it wakes and looks where it faces: stand on that ray at ~115 with line of sight
            q = spot_on_ray(125.0) or spot_on_ray(140.0)
            if q is not None and dist(a.p, q) > 3.0:
                d.decision = f'walk-lead: it rests ({gap:.0f}); moving onto its ray'
                path = plan(w.rects, w.width, w.height, a.p, q, radius=6.0, avoid=[(p.p, 70.0)], slow=w.biome_at)
                nxt = path[1] if path and len(path) > 1 else q
                return step_toward(a, nxt, min(walk, dist(a.p, q)))
            d.decision = f'walk-lead: it rests ({gap:.0f}); waiting on its ray'
            return self._facing(d, a, p, hold(a))
        if gap < 90.0 and not waking:
            # in its charge range without a sprint: straight away (it walks at 11 or sprints at 15;
            # the walk phase ends in a rest, which is our way out)
            away = heading_of(sub(a.p, p.p))
            h = steer(a, p, away, w.rects, math.radians(35), probe=60.0, keep_los=True, slow=w.biome_at)
            d.decision = f'walk-lead: inside 90 ({gap:.0f}), retreating'
            return self._facing(d, a, p, self._move_heading(a, h, walk))
        if not following and not waking:
            # it does not see us: get onto its forward ray at ~125 where it will
            bearing = abs(wrap(heading_of(sub(a.p, p.p)) - p.heading))
            if bearing < math.radians(28) and los_clear(p.p, a.p, w.rects) and gap <= 235.0:
                d.decision = f'walk-lead: in its sight, waiting ({gap:.0f})'
                return self._facing(d, a, p, hold(a))
            q = spot_on_ray(125.0) or spot_on_ray(150.0)
            if q is not None:
                d.decision = f'walk-lead: hooking from its front ({gap:.0f})'
                path = plan(w.rects, w.width, w.height, a.p, q, radius=6.0, avoid=[(p.p, 95.0)], slow=w.biome_at)
                nxt = path[1] if path and len(path) > 1 else q
                return self._facing(d, a, p, step_toward(a, nxt, min(walk, dist(a.p, q))))
            d.decision = f'walk-lead: no spot in its sight ({gap:.0f}); waiting'
            return self._facing(d, a, p, hold(a))
        desired = heading_of(sub(wp, a.p))
        h, real, g2 = self._walk_choose(a, p, desired, gap, v_pred, d.gaze, others)
        if wp is goal or dist(wp, goal) < 1e-6:
            real = min(real, dist(a.p, goal))
        d.decision = f'walk-lead: gap {gap:.0f}->{g2:.0f}, v {real:.0f} (pred {v_pred:.0f}), wp {tuple(round(v) for v in wp)}, e {a.energy:.0f}'
        act = self._move_heading(a, h, real) if real > 0.05 else hold(a)
        act['turn_angle'] = float(wrap(heading_of(sub(p.p, add(a.p, polar(h, real)))) + GAZE_OFFSET * d.gaze - a.heading))
        return act

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
    def flyby_side(self, site: Site, a: AgentView, p: PredatorView | None = None):
        """Unit vector along the obstacle face with the longer clear sprint from the flyby point,
        never toward the predator's side, away from other predators when both are clear; None if
        neither side is clear for 40."""
        w = self.world
        tangent = (-site.normal[1], site.normal[0])
        best = None
        for sgn in (1.0, -1.0):
            side = mul(tangent, sgn)
            if p is not None and dot(sub(p.p, a.p), side) > 12.0:
                continue          # it stands on that side
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
            side = self.flyby_side(d.site, a, p)
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
        if d.leash and p is not None:
            d.lost_ticks = 0
            if a.energy < 6.0 and d.phase != 'FLYBY' and not (d.become_bait and dist(a.p, d.site.holder) < 40):
                d.done = 'failed:guide_energy'
                d.decision = 'leash: starving, handing over to the society'
                return DEFER
            if d.phase == 'FLYBY':
                return self._flyby(d, a, p, dist(a.p, p.p))
            return self._leash(d, a, p)
        # another loose predator close by: let the society's flee logic act for us
        for q in w.recent_predators():
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
        order = ('ATTRACT', 'OPEN', 'LEAD', 'LEASH', 'CORRIDOR', 'FRONT', 'FLYBY', 'ENTER')
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
            if gap < 130 and a.energy - a.max_energy / 5 < 60.0 and not is_resting(p):
                # not enough sprint left to reopen the gap if it charges: hand over to the society's
                # flee while there is still room
                d.done = 'failed:guide_energy'
                d.decision = f'lead: sprint reserve gone at gap {gap:.0f}, giving up'
                return DEFER
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

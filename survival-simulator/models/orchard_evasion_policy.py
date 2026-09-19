"""Coordinated orchard foraging plus own-sighting predator evasion. No trapping.

Second seed family for the no-trapping optimization campaign. The foraging,
population and lineage-shared-map behaviour is the vendored
models/survival/orchard_population.py; this layer adds the predator response it
lacks entirely.

Evasion defaults come from the `with_predators_best` config measured on
survival-simulator/oscar-overnight-cpp (configs/best-configs.json). That
campaign's own numbers, on its separate C++ engine port and therefore not
verified against this engine, ranked flee-close/own-sightings above both a
wider-radius shared-sighting variant (1239 s vs 923 s mean survival) and every
bait/guide trapping variant it tried (all negative against this same baseline).
Treat those as the reason these defaults were chosen, not as measured results
for this repository.

Sharing predator sightings across a group is deliberately not implemented:
upstream measured it worse than own-sightings-only, so `pred_share` accepts 0
and rejects anything else rather than silently ignoring the setting.

Inputs stay the public per-agent observation dictionaries and simulation time.
The threat query reads only observation entries of type 'Predator' with their
reported distance/angle; no engine state, world seed or true coordinates.

Deflection layer (`wall_mode`, off by default)
---------------------------------------------
src/elements/predator.py charges only when `abs(agent_looking_dir) > pi/2` or the
agent is inside `hearing_radius*1.5` (90); otherwise it pivots, moving one
sprint step along `angle_to_agent + pivot_sign*pi/4` with
`pivot_sign = -sign(agent_looking_dir)`, then re-facing the agent. Three
identities make that controllable from public observations alone; all three were
checked against the shipped engine to float precision:

  1. `agent_looking_dir`, the quantity the predator gates and signs on, is
     bit-identical to this agent's own observed `angle` to that predator.
  2. The predator's facing in the agent's local frame is `angle + pi - rel_dir`.
  3. The predator's pivot step, expressed in the agent's local frame, is
     `15 * unit(angle + pi + pivot_sign*pi/4)` - the bearing from predator to
     agent rotated by +/-45 degrees, independent of `rel_dir`.

So holding `abs(angle)` at a small `wall_epsilon` keeps the predator in pivot
mode (no charge) and the *sign* of that angle picks which of two 45-degree
offsets it takes. A pivot step closes the gap by only `10.2` at 140 units
(the chord, not the `15*cos(45) = 10.6` projection), so an agent walking away
at 10 holds the band open almost indefinitely, and edge avoidance is
unreachable while a predator can see or hear an agent (`if agents: ... return`
precedes `if edges:`), so the predator cannot dodge what it is aimed at.

The sign matters in both directions: `np.sign(0.) == 0.` makes `pivot_sign` 0
and `move_dir` the bare bearing, i.e. a straight sprint in at the full 15 per
tick (measured). `wall_epsilon` is clamped away from 0 for exactly that reason.
The shipped `face` branch returns `turn = clamp(angle)`, which lands the facing
dead on the predator and triggers that case; it is harmless only because
pred_face_r defaults to 80, under the engine's 90-unit charge gate, so it never
fires where pivoting was available. notrap_config exposes pred_face_r over
(0., 250.). Nudging that turn off-axis is a real improvement but it would
change wall_mode=0 behaviour and diverge from the native port, so it is left
alone here; measured effect at pred_face_r=190 was within noise.

`wall_mode` selects what the sign is used for:
  0  off - behaviour is byte-identical to the pre-deflection policy.
  1  steer: pick the sign whose pivot step most reduces the predator's distance
     to a remembered wall point, filtered to wall points and retreat headings
     that lead away from this group's own orchard.
  2  hold only - ablation. Same pivot hold, sign frozen at +1, no wall point and
     no colony filter, so an A/B against mode 1 separates "deny the charge"
     from "aim at a wall".

Both modes steer only when this agent is the predator's own nearest visible
agent, because that is the only one `Predator.step` reacts to; see _is_target.

MEASURED RESULT (this engine, this repository): the control works and does not
pay. Geometrically, in scenarios built so the branch can always fire, the sign
choice cuts the predator's closest approach to the intended wall by 22 units
(44 of 46 engagements closer, 1 farther) and beats the frozen-sign ablation
(22 lower, 4 higher). But the predator ends up *nearer* the abandoned colony,
not farther (paired -29 units), because pinning it to a wall keeps it in the
neighbourhood instead of letting it wander off; and holding it in the band
raises the number of ticks it spends charging rather than pivoting.

Full-game paired A/B, fastsim engine, identical world seeds, 3000 s horizon,
40 seeds per arm, baseline mean score 1456.9 (per-seed sd 327):

    wall_mode=1 default band    -52.8  [-176, +71]   16/40 seeds better
    wall_mode=1 steer_max 30   -109.0  [-243, +25]   11/40
    wall_mode=1 engage_r 140     -7.0  [-182, +168]  16/40
    wall_mode=2 hold            -88.7  [-234, +56]   15/40
    wall_mode=2 hold, 30 ticks -114.7  [-256, +26]   13/40

Every variant is negative on both score and survival, and the only one that
reaches parity does so by barely engaging. Steering costs ~0.52 energy a tick
against ~0.005 for the post/watch tick it replaces - about 13% of the colony's
reproductive energy budget in the unguarded configuration. So: the handle is
real and precise, the objective it optimises is the wrong one.
That is why the default is 0 and why no C++ port exists -
`fastsim/_evasion.hpp` has no steer branch, and the native config reader
silently ignores unknown keys, so `wall_mode>0` with `--engine native` would
quietly run the off behaviour. fastsim/verify_evasion.py catches that
(23143 divergences with wall_mode=1, 0 with wall_mode=0) and is the gate.
"""
from __future__ import annotations

import math

from models.survival.orchard_population import MOVE_PENALTY, OrchardPolicy, wrap

# Floor of the deflection band. The predator's charge gate is a hard
# `distance < hearing_radius*1.5` == 90 measured after the agent has moved, and a
# pivot step closes at most 10.6, so 105 survives one fully blocked retreat tick
# without handing the predator a charge. Not tunable: it is a safety margin
# against an engine threshold, not a policy preference.
_WALL_MIN_R = 105.
# Sign changes tolerated per engagement before the sign is frozen. Re-optimising
# every tick oscillates: a greedy controller can walk the predator in to a 28
# unit gap and then pull it straight back out again.
_WALL_MAX_FLIPS = 2
# Remembered edges older than this are ignored. 25 s is what the base class's own
# _goto trusts its edge memory for; the memory itself is retained for 40 s.
_WALL_EDGE_AGE = 25.
# Engagement state is dropped after this long without a steering tick.
_WALL_RESET = 1.


class OrchardEvasionPolicy(OrchardPolicy):
    def __init__(self, seed=0, *, pred_mode=1, pred_r=70., pred_face_r=80., pred_sprint_r=40.,
                 pred_dodge_r=80., pred_dodge_ang=1.4, pred_share=0, pred_turn_max=1.0,
                 pred_evade_closest=0, pred_face_threat=0,
                 wall_mode=0, wall_engage_r=190., wall_target_gap=35., wall_epsilon=0.25,
                 steer_max_ticks=120, **kw):
        if pred_share:
            raise ValueError('pred_share>0 (group-shared predator sightings) is not implemented')
        super().__init__(seed=seed, **kw)
        self.PRED = dict(mode=int(pred_mode), r=float(pred_r), face_r=float(pred_face_r),
                         sprint_r=float(pred_sprint_r), dodge_r=float(pred_dodge_r),
                         dodge_ang=float(pred_dodge_ang), turn_max=float(pred_turn_max),
                         # Only react to a predator that is actually hunting THIS agent.
                         # Predator.step targets min(agents, key=distance), so in a 20-30
                         # agent colony most sightings belong to somebody else and every
                         # flee tick spent on one costs ~0.52 energy against ~0.005 for the
                         # post tick it replaces. The _is_target predicate that decides this
                         # already existed but only gated the (dormant) deflection branch.
                         # Off by default: it changes behaviour, so it is a mechanism to be
                         # measured, not an assumed improvement.
                         evade_closest=int(pred_evade_closest),
                         # Turn to FACE the predator while retreating, instead of turning
                         # the agent's back to it.
                         #
                         # src/elements/predator.py:37 gates the charge on
                         # `abs(agent_looking_dir) > pi/2 or distance < 90`, and
                         # agent_looking_dir is the predator's view of our bearing - the
                         # same number as our own observed `angle` to it. Turning away
                         # drives |angle| towards pi, which SATISFIES that gate and buys
                         # the predator a 15/tick charge; holding |angle| near 0 leaves it
                         # in the pivot branch, which closes only 10.6/tick. Facing also
                         # keeps the sighting inside our own pi/6 vision half-angle, so the
                         # evasion layer does not lose the predator mid-chase and revert to
                         # foraging.
                         #
                         # The move direction is unchanged - the agent still retreats along
                         # `away`; only the commanded turn differs.
                         face_threat=int(pred_face_threat))
        # `epsilon` is clamped, not rejected: 0 would make np.sign(0) == 0 and send the
        # predator straight at the agent at full sprint, and anything at or above the
        # agent's own vision half-angle (pi/6) loses the sighting the branch needs. The
        # upper clamp leaves room for the ~0.15 rad of bearing change one pivot step
        # adds after the turn has been committed.
        self.WALL = dict(mode=int(wall_mode), engage_r=float(wall_engage_r),
                         target_gap=float(wall_target_gap),
                         epsilon=max(0.05, min(0.35, float(wall_epsilon))),
                         max_ticks=int(steer_max_ticks))
        self.metrics.update(flee_ticks=0, face_ticks=0, sprint_ticks=0,
                            steer_ticks=0, steer_locked=0, steer_flips=0, steer_aborts=0,
                            steer_no_target=0, steer_not_target=0)

    def _threat(self, s):
        """Nearest predator in this agent's own observations, or None."""
        nearest = None
        for o in s['observations']:
            if o['type'] != 'Predator':
                continue
            if nearest is None or o['distance'] < nearest['distance']:
                nearest = o
        return nearest

    def _release_fruit(self, m):
        """Free a claimed fruit before fleeing/facing, or it sits reserved and
        unreachable by hungrier group-mates for as long as the threat lasts."""
        if m.fruit is not None:
            g = self.groups[m.group]
            if m.fruit in g.fruits:
                g.fruits[m.fruit].claimed = None
            m.fruit = None

    # ------------------------------------------------------------ deflection
    def _colony(self, m):
        """Group-frame point the maneuver must never drag a predator towards.

        Posts are the orchard itself and are preferred; without any, the other
        members' dead-reckoned positions stand in. A lone agent with no post has
        nothing to protect, so the filter is skipped (None).
        """
        g = self.groups[m.group]
        posts, others = [], []
        for aid in g.agents:
            om = self.minds[aid]
            if om.post is not None and om.post in g.trees:
                posts.append(g.trees[om.post].p)
            if aid != m.aid:
                others.append(om.pose.p)
        pts = posts or others
        if not pts:
            return None
        return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))

    def _wall_target(self, m, pred_p, retreat_dir):
        """Nearest remembered wall point to the predator that leads away from home.

        Mirrors the predator's own edge-avoidance geometry - the globally nearest
        point across all candidate segments, not a corner search - because the
        engine picks exactly that point and corners are not special to it.
        """
        pose = m.pose
        colony = self._colony(m)
        if colony is not None:
            cd = (colony[0]-pose.p[0], colony[1]-pose.p[1])
            cn = math.hypot(*cd)
            if cn > 1e-6:
                cd = (cd[0]/cn, cd[1]/cn)
                # Retreating towards home would tow the predator into the orchard
                # no matter which wall is chosen, so refuse the whole engagement.
                if math.cos(retreat_dir)*cd[0]+math.sin(retreat_dir)*cd[1] > 0.:
                    return None
            else:
                colony = None
        reach = 2.*self.WALL['engage_r']
        best = None
        for a, b, seen in m.edges:
            if self.time-seen >= _WALL_EDGE_AGE:
                continue
            vx, vy = b[0]-a[0], b[1]-a[1]
            L2 = vx*vx+vy*vy
            if L2 < 1e-9:
                continue
            t = ((pred_p[0]-a[0])*vx+(pred_p[1]-a[1])*vy)/L2
            t = max(0., min(1., t))
            pt = (a[0]+t*vx, a[1]+t*vy)
            gap = math.dist(pt, pred_p)
            if gap > reach:
                continue
            if colony is not None:
                wx, wy = pt[0]-pose.p[0], pt[1]-pose.p[1]
                wn = math.hypot(wx, wy)
                if wn < 1e-6 or (wx/wn)*cd[0]+(wy/wn)*cd[1] > 0.:
                    continue
            if best is None or gap < best[1]:
                best = (pt, gap)
        return best

    @staticmethod
    def _is_target(s, threat):
        """Is this agent the one the predator is actually chasing?

        src/elements/predator.py picks `min(agents, key=distance)`, so only its
        nearest visible agent has any control over it. In a 20-30 agent colony most
        sightings belong to somebody else, and steering on one of those spends
        movement and turn energy on a predator that never reacts. Own observations
        only: every neighbour's distance to the predator is computable in this
        agent's local frame. Conservative - a nearer agent this one cannot see
        still slips through - but it never wrongly suppresses the true target.
        """
        d, a = threat['distance'], threat['angle']
        px, py = d*math.cos(a), d*math.sin(a)
        for o in s['observations']:
            if o['type'] != 'Agent':
                continue
            ox, oy = o['distance']*math.cos(o['angle']), o['distance']*math.sin(o['angle'])
            if math.hypot(px-ox, py-oy) < d:
                return False
        return True

    def _steer(self, m, s, threat):
        """Hold the predator in pivot mode and aim its 45-degree step at a wall.

        Returns an action tuple or None to fall through to the unchanged
        flee/face/forage branches.
        """
        WALL, P = self.WALL, self.PRED
        distance, angle = threat['distance'], threat['angle']
        if distance <= max(_WALL_MIN_R, P['r']) or distance > WALL['engage_r']:
            return None
        if not self._is_target(s, threat):
            self.metrics['steer_not_target'] += 1
            return None
        st = getattr(m, 'wsteer', None)
        if st is not None and self.time-st['at'] > _WALL_RESET:
            st = None                      # contact lost long enough: new engagement
        if st is not None and st['done']:
            return None                    # tick cap already spent on this engagement
        pose = m.pose
        # Retreat straight away from the predator: it is the heading that holds the
        # band open longest (pivot closes 10.6/tick, walking away gives back 10).
        retreat = wrap(angle+math.pi)
        walk = min(s['speed'], s['sprint_speed'])
        held = WALL['mode'] != 1 or (st is not None and (st['locked'] or st['flips'] >= _WALL_MAX_FLIPS))
        target, gap = None, None
        if not held:
            # Activation gate (mode 1 only, and only while still re-optimising): a
            # wall has to be in sight and it has to be one that leads away from home.
            # An engagement already under way may keep aiming at a remembered wall -
            # holding the predator in pivot mode means facing it, which routinely
            # swings the agent's 60-degree cone off the wall it started on.
            if st is None and not any(o['type'] == 'Edge' for o in s['observations']):
                return None
            found = self._wall_target(m, pose.polar(threat), pose.theta+retreat)
            if found is None:
                self.metrics['steer_no_target'] += 1
                return None
            target, gap = found
        if st is None:
            st = dict(sign=0., flips=0, ticks=0, locked=False, done=False, at=self.time)
            m.wsteer = st
        # --- sign: which 45-degree pivot offset the predator takes this tick ---
        if held:
            sign = st['sign'] or 1.        # frozen: ablation, lock-in, or flip-flop
        else:
            pred_p = pose.polar(threat)
            bearing = math.atan2(pose.p[1]-pred_p[1], pose.p[0]-pred_p[0])
            best = None
            for cand in (1., -1.):
                a = bearing+cand*math.pi/4
                step = (pred_p[0]+15.*math.cos(a), pred_p[1]+15.*math.sin(a))
                d = math.dist(step, target)
                if best is None or d < best[1]:
                    best = (cand, d)
            sign = best[0]
            if st['sign'] and sign != st['sign']:
                st['flips'] += 1
                self.metrics['steer_flips'] += 1
                if st['flips'] >= _WALL_MAX_FLIPS:
                    sign = st['sign']      # oscillating: freeze on the incumbent
            if gap <= WALL['target_gap']:
                st['locked'] = True
                self.metrics['steer_locked'] += 1
        st['sign'] = sign
        st['at'] = self.time
        st['ticks'] += 1
        if st['ticks'] >= WALL['max_ticks']:
            st['done'] = True
            self.metrics['steer_aborts'] += 1
        # --- facing: put the predator's own view of us at -sign*epsilon so that
        # pivot_sign == -sign(agent_looking_dir) == sign. The engine moves with the
        # old facing and turns afterwards, so the retreat and the aim do not fight:
        # predict where the predator sits after our own step, then turn onto it.
        step = walk*MOVE_PENALTY.get(s['biome'], 1.0)
        px = distance*math.cos(angle)-step*math.cos(retreat)
        py = distance*math.sin(angle)-step*math.sin(retreat)
        turn = wrap(math.atan2(py, px)+sign*WALL['epsilon'])
        turn = max(-P['turn_max'], min(P['turn_max'], turn))
        self.metrics['steer_ticks'] += 1
        self._release_fruit(m)
        return walk, retreat, turn, 'steer'

    def _act(self, m, s, states):
        P = self.PRED
        if P['mode']:
            threat = self._threat(s)
            # Somebody else's predator: fall through to foraging untouched. Applied
            # once here rather than inside each branch so flee, deflection and face
            # all agree on whose predator this is.
            if threat is not None and P['evade_closest'] and not self._is_target(s, threat):
                threat = None
            if threat is not None:
                distance, angle = threat['distance'], threat['angle']
                clamp = lambda a: max(-P['turn_max'], min(P['turn_max'], a))
                if distance <= P['r']:
                    away = wrap(angle+math.pi)
                    # A predator outruns an agent in a straight line, so break across
                    # its approach instead of directly away once it is this close.
                    if distance <= P['dodge_r']:
                        away = wrap(away+(P['dodge_ang'] if angle >= 0. else -P['dodge_ang']))
                    sprinting = distance <= P['sprint_r']
                    reach = s['sprint_speed'] if sprinting else min(s['speed'], s['sprint_speed'])
                    self.metrics['flee_ticks'] += 1
                    self.metrics['sprint_ticks'] += sprinting
                    self._release_fruit(m)
                    # Retreat along `away` either way; only the facing differs.
                    return reach, away, clamp(angle if P['face_threat'] else away), 'flee'
                # Deflection sits below flee and above face: its band floor is
                # _WALL_MIN_R > the engine's 90-unit charge gate and >= pred_r, so a
                # predator already close enough to charge is still handled by flee.
                if self.WALL['mode']:
                    steer = self._steer(m, s, threat)
                    if steer is not None:
                        return steer
                if distance <= P['face_r']:
                    self.metrics['face_ticks'] += 1
                    self._release_fruit(m)
                    return 0., 0., clamp(angle), 'face'
        return super()._act(m, s, states)

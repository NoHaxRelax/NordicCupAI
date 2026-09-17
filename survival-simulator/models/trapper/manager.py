"""Trap manager: stations, roles and deliveries on top of a running society.

Roles (all read the WorldState only):

* bait      - stands on a bait slot of a station (wall: two slots +-10 along the
              face; gap: one slot 5 inside the mouth).
* successor - walks to the staging slot behind the bait, then takes the slot over;
              the old bait is released and returns to the society.
* guard     - (wall) waits behind the baits and intercepts predators that approach
              the protected side, leading them round to the front like any other
              delivery.
* guide     - runs one Delivery (see lure.py).
* released  - walks clear of the station and drops back into the society.

Everyone else is the society; the policy keeps them out of the held zones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import add, sub, mul, dot, dist, unit, wrap, heading_of, path_clear
from .lure import Lure, Holder, Delivery, predator_target, is_resting, run_in_point
from .motion import action, step_toward, hold, speed_for
from .paths import plan, path_length
from .sites import find_sites, Site, CORRIDOR
from .world import WorldState, AgentView, PredatorView, PRED_CHARGE_RANGE

DEFAULTS = dict(
    max_stations=2,          # stations kept staffed at once
    max_deliveries=2,        # concurrent deliveries
    guide_min_energy=70.0,   # a guide needs this much to start
    bait_min_energy=65.0,    # below this a bait asks for a successor
    bait_release_energy=40.0,   # below this a bait leaves to eat even without a successor
    successor_min_energy=110.0,
    holder_min_energy=110.0,
    standby_radius=170.0,    # a wall station's standby bait forages within this radius
    relevant_range=420.0,    # a free predator this close to an agent is worth trapping
    max_turn=math.radians(95),   # reject deliveries needing a bigger direction change
    prestaff_time=40.0,      # staff the best station from this time on
    min_workers=3,           # never take the last workers away from foraging
    zone_radius=95.0,        # society agents keep this far from held predators
    guard=True,
    two_baits=True,
)


@dataclass
class Station:
    site: Site
    slots: list                      # bait slot points
    baits: dict = field(default_factory=dict)      # aid -> slot index
    successor: int | None = None
    guard: int | None = None
    held: set = field(default_factory=set)         # predator ids currently held
    opened: float = 0.0
    last_held: float = -1e9
    events: list = field(default_factory=list)
    energy_hist: dict = field(default_factory=dict)  # aid -> last energy (senescence detection)

    @property
    def key(self):
        return self.site.key

    def staffed(self):
        return bool(self.baits)

    def free_slot(self):
        used = set(self.baits.values())
        for i in range(len(self.slots)):
            if i not in used:
                return i
        return None


class TrapManager:
    def __init__(self, **params):
        self.P = dict(DEFAULTS, **params)
        self.sites: list[Site] = []
        self.stations: dict[str, Station] = {}
        self.deliveries: dict[int, Delivery] = {}       # pid -> delivery
        self.roles: dict[int, tuple] = {}               # aid -> (role, key)
        self.released: dict[int, tuple] = {}            # aid -> (point, until)
        self.metrics = dict(deliveries=0, delivered=0, guide_lost=0, failed=0, handoffs=0, guard_interceptions=0,
                            held_predator_ticks=0, zone_pushes=0, releases=0)
        self.events = []
        self.decisions: dict[int, str] = {}
        self._sites_done = False
        self.time = 0.0
        self.pred_cooldown: dict[int, float] = {}       # pid -> time before another attempt
        self.agent_cooldown: dict[int, float] = {}      # aid -> time before another role
        self.senescent: set = set()                     # agents seen draining fast while idle
        self._energy_seen: dict[int, tuple] = {}        # aid -> (energy, x, y)
        self.last_actions: dict = {}                    # aid -> ActionRequest applied last tick (set by the policy)
        self.fleeing: set = set()                       # agents the society wants to flee this tick

    # ------------------------------------------------------------------ helpers
    def event(self, kind, **data):
        self.events.append(dict(t=round(self.time, 1), kind=kind, **data))

    def role_of(self, aid):
        return self.roles.get(aid, (None, None))[0]

    def _slots(self, site: Site):
        if site.kind == 'wall' and self.P['two_baits']:
            return [add(site.holder, mul(site.axis, -10.0)), add(site.holder, mul(site.axis, 10.0))]
        return [site.holder]

    def _station(self, site: Site) -> Station:
        st = self.stations.get(site.key)
        if st is None:
            st = self.stations[site.key] = Station(site=site, slots=self._slots(site), opened=self.time)
            self.event('station_opened', key=site.key, site_kind=site.kind)
        return st

    def _eligible(self, world: WorldState, exclude=(), min_energy=60.0, max_age=None, allow_senescent=False):
        out = []
        for a in world.agents.values():
            if a.id in exclude or self.role_of(a.id) is not None:
                continue
            if a.energy < min_energy:
                continue
            if max_age is not None and a.age > max_age:
                continue
            if not allow_senescent and a.id in self.senescent:
                continue
            if self.agent_cooldown.get(a.id, -1.0) > world.time:
                continue
            out.append(a)
        return out

    def _track_senescence(self, world: WorldState):
        """Compare each agent's energy drop with the cost of the action it actually took.
        An extra loss beyond passive drain + movement + turning means the hidden max
        age has passed (the engine then charges 0.01 x age per tick)."""
        for a in world.agents.values():
            prev = self._energy_seen.get(a.id)
            self._energy_seen[a.id] = (a.energy, a.x, a.y)
            act = self.last_actions.get(a.id)
            if prev is None or act is None or a.age < 50 or a.energy > prev[0] or act.spawn_agent:
                continue
            cap = a.sprint_speed if prev[0] >= a.max_energy / 5 else a.walk
            d = max(0.0, min(float(act.move_distance), cap))
            move_cost = d * 0.05 if d <= a.speed else a.speed * 0.05 + (d - a.speed) * 0.5
            turn_cost = min(math.pi, abs(float(act.turn_angle))) / (2 * math.pi)
            expected = 0.1 + move_cost + turn_cost
            if (prev[0] - a.energy) - expected > 0.15:
                if a.id not in self.senescent:
                    self.event('senescent', agent=a.id, age=round(a.age, 1))
                self.senescent.add(a.id)
        for aid in list(self.senescent):
            if aid not in world.agents:
                self.senescent.discard(aid)
                self._energy_seen.pop(aid, None)

    def _workers(self, world):
        return sum(1 for a in world.agents.values() if self.role_of(a.id) is None)

    def _held_centers(self):
        return [(st.site.held_center(), self.P['zone_radius']) for st in self.stations.values() if st.held or st.staffed()]

    # ------------------------------------------------------------------ status
    def _update_status(self, world: WorldState):
        held_now = {}
        for st in self.stations.values():
            st.held = set()
        for p in world.predators:
            for st in self.stations.values():
                if not st.staffed():
                    continue
                if st.site.in_front_zone(p.p, margin=15.0):
                    target = predator_target(world, p)
                    if target in st.baits or (target is None and is_resting(p)):
                        st.held.add(p.pid)
                        st.last_held = world.time
                        held_now[p.pid] = st.key
                        break
        self.metrics['held_predator_ticks'] += len(held_now)
        return held_now

    # ------------------------------------------------------------------ main
    def step(self, world: WorldState, fleeing=()):
        self.fleeing = set(fleeing)
        self.time = world.time
        self.decisions = {}
        if not self._sites_done and world.complete_map:
            self.sites = find_sites(world.rects, world.width, world.height)
            self._sites_done = True
            self.event('sites', walls=sum(s.kind == 'wall' for s in self.sites), gaps=sum(s.kind == 'gap' for s in self.sites))
        # drop roles of dead agents
        for aid in list(self.roles):
            if aid not in world.agents:
                role, key = self.roles.pop(aid)
                st = self.stations.get(key)
                if st is not None:
                    st.baits.pop(aid, None)
                    if st.successor == aid:
                        st.successor = None
                    if st.guard == aid:
                        st.guard = None
                    if role in ('bait', 'standby'):
                        self.event(f'{role}_died', key=key, agent=aid)
        for pid in list(self.deliveries):
            d = self.deliveries[pid]
            if d.guide not in world.agents and d.done is None:
                d.done = 'guide_captured'
                self.metrics['guide_lost'] += 1
                self.event('guide_captured', pid=pid, key=d.site.key, agent=d.guide, phase=d.phase, last=d.decision,
                           after=round(self.time - d.created, 1))
        self._track_senescence(world)
        held_now = self._update_status(world)
        # finished deliveries
        for pid in list(self.deliveries):
            d = self.deliveries[pid]
            if d.done == 'delivered' or (d.done == 'guide_captured'):
                if pid in held_now:
                    self.metrics['delivered'] += 1
                    self.event('delivered', pid=pid, key=d.site.key)
                    if d.become_bait and d.guide in world.agents:
                        st = self._station(d.site)
                        st.baits[d.guide] = 0
                        self.roles[d.guide] = ('bait', st.key)
                    else:
                        self.roles.pop(d.guide, None)
                    del self.deliveries[pid]
                elif d.done == 'guide_captured' or self.time - d.created > 400:
                    # guide gone but predator not held: give the predator a few seconds to settle
                    if self.time - getattr(d, 'ended', self.time) > 3.0:
                        self.metrics['failed'] += 1
                        self.event('delivery_failed', pid=pid, key=d.site.key, reason=d.done)
                        self.roles.pop(d.guide, None)
                        self.pred_cooldown[pid] = self.time + 8.0
                        del self.deliveries[pid]
                    elif not hasattr(d, 'ended'):
                        d.ended = self.time
            elif d.done is not None:
                self.metrics['failed'] += 1
                self.event('delivery_failed', pid=pid, key=d.site.key, reason=d.done, phase=d.phase, after=round(self.time - d.created, 1))
                self.roles.pop(d.guide, None)
                self.pred_cooldown[pid] = self.time + (15.0 if d.done.startswith('failed:cannot') else 8.0)
                self.agent_cooldown[d.guide] = self.time + 10.0
                del self.deliveries[pid]
            elif pid in held_now and d.phase in ('FRONT', 'CORRIDOR', 'ENTER'):
                pass
            elif world.predator(pid) is None:
                d.done = 'failed:predator_gone'
        self._staff(world)
        self._assign(world, held_now)
        return self._actions(world, held_now)

    # ------------------------------------------------------------------ staffing
    def _best_site(self, world: WorldState, exclude_keys=()):
        if not self.sites or not world.agents:
            return None
        cx = sum(a.x for a in world.agents.values()) / len(world.agents)
        cy = sum(a.y for a in world.agents.values()) / len(world.agents)
        best = None
        for s in self.sites:
            if s.key in exclude_keys:
                continue
            d = dist((cx, cy), s.holder)
            score = d + (0 if s.kind == 'wall' else 60)
            if best is None or score < best[0]:
                best = (score, s)
        return best[1] if best else None

    def _staff(self, world: WorldState):
        P = self.P
        # open the first station early so deliveries never wait for a holder
        active = [st for st in self.stations.values() if st.staffed() or any(r == ('bait', st.key) for r in self.roles.values())]
        if world.time >= P['prestaff_time'] and len(active) < 1 and len(self.stations) < P['max_stations']:
            site = self._best_site(world)
            if site is not None and self._workers(world) > P['min_workers']:
                self._station(site)
        for st in self.stations.values():
            site = st.site
            need_live = bool(st.held) or any(d.site.key == st.key and not d.become_bait for d in self.deliveries.values())
            # standby bait for wall stations: forages near the station, steps onto the slot when needed
            if site.kind == 'wall':
                standby = [aid for aid, (r, k) in self.roles.items() if r == 'standby' and k == st.key]
                if not st.baits and not standby and self._workers(world) > P['min_workers']:
                    cands = self._eligible(world, min_energy=P['holder_min_energy'], max_age=45.0)
                    cands.sort(key=lambda a: (abs(a.energy - 160.0) / 60.0 + a.age / 40.0 + dist(a.p, site.holder) / 200.0))
                    if cands:
                        self.roles[cands[0].id] = ('standby', st.key)
                        self.event('standby_assigned', key=st.key, agent=cands[0].id)
                        standby = [cands[0].id]
                if need_live and standby and not st.baits:
                    aid = standby[0]
                    st.baits[aid] = 0
                    self.roles[aid] = ('bait', st.key)
                    self.event('bait_activated', key=st.key, agent=aid)
                elif not need_live and st.baits and world.time - st.last_held > 25.0 and not st.held:
                    for aid in list(st.baits):
                        st.baits.pop(aid)
                        self.roles[aid] = ('standby', st.key)
                        self.event('bait_standby', key=st.key, agent=aid)
            else:
                # gap: the first guide becomes the bait; re-staff through the far mouth only if
                # predators are held there and the bait is gone
                if st.held and not st.baits and st.successor is None and self._workers(world) > P['min_workers']:
                    cands = self._eligible(world, min_energy=P['holder_min_energy'], max_age=45.0)
                    cands.sort(key=lambda a: dist(a.p, site.holder))
                    if cands:
                        st.baits[cands[0].id] = 0
                        self.roles[cands[0].id] = ('bait', st.key)
                        self.event('bait_assigned', key=st.key, agent=cands[0].id, slot=0)
            # successor when a bait weakens or ages
            for aid, slot_i in list(st.baits.items()):
                a = world.agents.get(aid)
                if a is None:
                    continue
                last = st.energy_hist.get(aid)
                st.energy_hist[aid] = a.energy
                drain = (last - a.energy) if last is not None else 0.0
                senescent = aid in self.senescent
                if st.held and (a.energy < P['bait_min_energy'] or senescent or a.age > 95) and st.successor is None and \
                        self._workers(world) > P['min_workers']:
                    cands = self._eligible(world, min_energy=P['successor_min_energy'], max_age=45.0)
                    cands.sort(key=lambda c: dist(c.p, site.successor))
                    if cands:
                        st.successor = cands[0].id
                        self.roles[cands[0].id] = ('successor', st.key)
                        st.successor_for = aid
                        self.event('successor_assigned', key=st.key, agent=cands[0].id, replaces=aid,
                                   reason='senescent' if senescent else ('energy' if a.energy < P['bait_min_energy'] else 'age'))
                    break
            # guard for staffed wall stations that hold something
            if P['guard'] and site.kind == 'wall' and site.guard is not None and st.staffed() and st.held and st.guard is None \
                    and self._workers(world) > P['min_workers']:
                cands = self._eligible(world, min_energy=P['guide_min_energy'], max_age=70.0)
                cands.sort(key=lambda c: dist(c.p, site.guard))
                if cands:
                    st.guard = cands[0].id
                    self.roles[cands[0].id] = ('guard', st.key)
                    self.event('guard_assigned', key=st.key, agent=cands[0].id)

    # ------------------------------------------------------------------ assignment
    def _assign(self, world: WorldState, held_now):
        P = self.P
        if len(self.deliveries) >= P['max_deliveries']:
            return
        busy_guides = {d.guide for d in self.deliveries.values()}
        free = [p for p in world.predators if p.pid not in held_now and p.pid not in self.deliveries]
        if not free:
            return
        agents = list(world.agents.values())

        def nearest_agent_dist(p):
            return min((dist(a.p, p.p) for a in agents), default=1e9)
        free.sort(key=nearest_agent_dist)
        for p in free:
            if len(self.deliveries) >= P['max_deliveries']:
                break
            if nearest_agent_dist(p) > P['relevant_range'] or self.pred_cooldown.get(p.pid, -1.0) > world.time:
                continue
            target = predator_target(world, p)
            best = None
            for st in self.stations.values():
                site = st.site
                if site.kind == 'wall':
                    # needs a bait or a standby that can step onto the slot in time
                    if not st.staffed() and not any(r == ('standby', st.key) for r in self.roles.values()):
                        continue
                else:
                    # a gap with a bait gone but predators held is re-staffed first
                    if st.held and not st.staffed():
                        continue
                entry = run_in_point(site)
                # guide candidate: the chased agent if eligible, else the nearest eligible agent
                cands = []
                if target is not None and target in world.agents and self.role_of(target) in (None, 'guard', 'standby') and \
                        world.agents[target].energy >= P['guide_min_energy'] * 0.6:
                    cands.append((world.agents[target], 0.0))
                for a in self._eligible(world, exclude=busy_guides, min_energy=P['guide_min_energy'], allow_senescent=True):
                    if a.id == target or self.role_of(a.id) == 'standby':
                        continue
                    # only guides that can reach its forward ray before it wanders off
                    goal, k = Lure(world)._intercept_point(a, p, 140.0)
                    if goal is None and not is_resting(p):
                        continue
                    cands.append((a, (k or 0) / 1.0 + dist(a.p, p.p) / 20.0))
                for a, attract_cost in cands:
                    if a.id in st.baits or a.id == st.successor:
                        continue
                    away = heading_of(sub(a.p, p.p))
                    to_entry = heading_of(sub(entry, a.p))
                    turn = abs(wrap(to_entry - away))
                    if turn > P['max_turn'] and dist(a.p, p.p) < 200:
                        continue
                    lead = dist(a.p, entry) + CORRIDOR
                    cost = attract_cost + lead / 10.0 + 60.0 * (turn / math.pi) ** 2
                    if a.id in self.senescent:
                        cost -= 40.0    # a senescent agent dies soon anyway: the cheapest guide
                    elif a.age > 60:
                        cost -= 10.0
                    if best is None or cost < best[0]:
                        best = (cost, st, a)
            if best is None:
                continue
            cost, st, a = best
            become_bait = st.site.kind == 'gap' and not st.staffed() and not st.held
            d = Delivery(site=st.site, guide=a.id, pid=p.pid, become_bait=become_bait, created=world.time)
            self.deliveries[p.pid] = d
            self.roles[a.id] = ('guide', st.key)
            busy_guides.add(a.id)
            self.metrics['deliveries'] += 1
            self.event('delivery_started', pid=p.pid, key=st.key, guide=a.id, cost=round(cost, 1), chased=target == a.id)

    # ------------------------------------------------------------------ actions
    def _actions(self, world: WorldState, held_now):
        out = {}
        lure = Lure(world, held=set(held_now))
        holder = Holder(world)
        avoid = self._held_centers()
        for aid, (role, key) in list(self.roles.items()):
            a = world.agents.get(aid)
            if a is None:
                continue
            st = self.stations.get(key) if key in self.stations else None
            if role == 'guide':
                d = next((d for d in self.deliveries.values() if d.guide == aid), None)
                if d is None:
                    self.roles.pop(aid, None)
                    continue
                p = world.predator(d.pid)
                act = lure.act(d, a, p)
                out[aid] = (act, f'guide[{d.phase}] {d.decision}')
            elif role == 'bait' and st is not None:
                site = st.site
                slot = st.slots[st.baits.get(aid, 0)]
                at_slot = dist(a.p, slot) < 1.0
                threat = self._wild_threat(world, st, a, held_now)
                if threat is not None:
                    out[aid] = (self._flee(a, threat, world), 'bait: fleeing a wild predator')
                    continue
                if not at_slot and aid in self.fleeing:
                    continue          # the society flees for us while we walk to the slot
                if at_slot and a.energy < self.P['bait_release_energy'] and st.successor is None:
                    # nobody is coming: leave before starving, the trap can be re-staffed later
                    st.baits.pop(aid, None)
                    self.roles[aid] = ('released', key)
                    self.released[aid] = (self._exit_point(site, slot), world.time + 15.0)
                    self.event('bait_abandoned', key=key, agent=aid, energy=round(a.energy, 1))
                    continue
                if site.kind == 'gap' and not at_slot:
                    act, why = self._gap_approach(world, holder, a, site, slot, held_now)
                else:
                    act, why = holder.act(a, slot, site, avoid=[c for c in avoid if dist(c[0], slot) > 60])
                out[aid] = (act, why)
            elif role == 'successor' and st is not None:
                site = st.site
                target_bait = getattr(st, 'successor_for', None)
                if target_bait not in st.baits:
                    # the bait we were to replace is gone: take its slot directly
                    slot_i = st.free_slot()
                    if slot_i is None:
                        self.roles.pop(aid, None); st.successor = None
                        continue
                    st.baits[aid] = slot_i
                    self.roles[aid] = ('bait', key)
                    st.successor = None
                    self.event('successor_promoted', key=key, agent=aid)
                    continue
                if aid in self.fleeing and dist(a.p, site.holder) > 40:
                    continue
                stage = site.successor
                if site.kind == 'wall':
                    stage = add(st.slots[st.baits[target_bait]], mul(site.normal, -12.0))
                    act, why = holder.act(a, stage, site, avoid=avoid)
                else:
                    act, why = self._gap_approach(world, holder, a, site, stage, held_now)
                if dist(a.p, stage) < 1.0:
                    # hand over: old bait released, we take the slot
                    slot_i = st.baits.pop(target_bait)
                    st.baits[aid] = slot_i
                    self.roles[aid] = ('bait', key)
                    st.successor = None
                    st.successor_for = None
                    self.roles[target_bait] = ('released', key)
                    self.released[target_bait] = (self._exit_point(site, st.slots[slot_i]), world.time + 15.0)
                    self.metrics['handoffs'] += 1
                    self.event('handoff', key=key, old=target_bait, new=aid)
                    why = 'successor: took over'
                out[aid] = (act, why)
            elif role == 'standby' and st is not None:
                # society forages; only pull the agent back when it strays too far
                if aid in self.fleeing:
                    continue
                if a.energy < 45.0:
                    self.roles.pop(aid, None)
                    self.agent_cooldown[aid] = world.time + 40.0
                    self.event('standby_released', key=key, agent=aid, energy=round(a.energy, 1))
                    continue
                if dist(a.p, st.site.holder) > self.P['standby_radius']:
                    act, why = holder.act(a, st.site.holder, st.site, avoid=avoid)
                    out[aid] = (act, 'standby: returning toward the station')
            elif role == 'released':
                if aid in self.fleeing:
                    self.roles.pop(aid, None); self.released.pop(aid, None)
                    self.agent_cooldown[aid] = world.time + 60.0
                    continue
                point, until = self.released.get(aid, (a.p, 0.0))
                if dist(a.p, point) < 8.0 or world.time > until:
                    self.roles.pop(aid, None)
                    self.released.pop(aid, None)
                    self.agent_cooldown[aid] = world.time + 60.0
                    self.metrics['releases'] += 1
                    continue
                out[aid] = (step_toward(a, point, a.walk * a.move_modifier), 'released: leaving the station')
            elif role == 'guard' and st is not None:
                site = st.site
                intruder = self._intruder(world, st, held_now)
                if intruder is not None and st.guard == aid and intruder.pid not in self.deliveries and \
                        len(self.deliveries) < self.P['max_deliveries']:
                    d = Delivery(site=site, guide=aid, pid=intruder.pid, created=world.time)
                    self.deliveries[intruder.pid] = d
                    self.roles[aid] = ('guide', key)
                    st.guard = None
                    self.metrics['guard_interceptions'] += 1
                    self.metrics['deliveries'] += 1
                    self.event('guard_intercept', key=key, agent=aid, pid=intruder.pid)
                    p = world.predator(intruder.pid)
                    act = lure.act(d, a, p)
                    out[aid] = (act, f'guide[{d.phase}] {d.decision}')
                    continue
                station = site.guard
                if aid in self.fleeing and dist(a.p, station) > 5:
                    continue
                act, why = holder.act(a, station, site, avoid=avoid)
                out[aid] = (act, why.replace('holder', 'guard'))
        return out

    def _exit_point(self, site: Site, slot):
        if site.kind == 'wall':
            return add(slot, mul(site.normal, -90.0))
        far = site.far_mouth if site.far_mouth is not None else site.front_mid
        return add(far, mul(site.normal, -45.0))

    def _gap_approach(self, world: WorldState, holder: Holder, a: AgentView, site: Site, goal, held_now):
        """Reach a point inside a gap passage through the far mouth, never through the
        predator crowd at the front mouth; wait outside if a loose predator sits there."""
        far = site.far_mouth if site.far_mouth is not None else site.front_mid
        outside = add(far, mul(site.normal, -45.0))
        inside_passage = abs(dot(sub(a.p, site.front_mid), site.axis)) < site.length + 5 and \
            abs(dot(sub(a.p, site.front_mid), (-site.axis[1], site.axis[0]))) < site.thickness / 2 + 1.0 and \
            dot(sub(a.p, site.front_mid), site.axis) > -1.0
        if inside_passage:
            return step_toward(a, goal, min(a.walk * a.move_modifier, dist(a.p, goal)), face=goal), 'gap: walking the passage'
        loose = [p for p in world.predators if p.pid not in held_now and dist(p.p, far) < 70]
        if loose and dist(a.p, outside) < 12:
            return hold(a), 'gap: waiting outside the far mouth'
        if dist(a.p, outside) < 6:
            return step_toward(a, far, a.walk * a.move_modifier, face=far), 'gap: entering the far mouth'
        act, why = holder.act(a, outside, site, avoid=[(site.held_center(), self.P['zone_radius'])])
        return act, why.replace('holder', 'gap route')

    def _wild_threat(self, world: WorldState, st: Station, a: AgentView, held_now):
        """A predator that is not held and can reach the bait: on the protected side within 45."""
        for p in world.predators:
            if p.pid in held_now or is_resting(p):
                continue
            if dist(p.p, a.p) < 45 and dot(sub(p.p, st.site.front_mid), st.site.normal) < 0:
                return p
        return None

    def _intruder(self, world: WorldState, st: Station, held_now):
        best = None
        for p in world.predators:
            if p.pid in held_now:
                continue
            behind = dot(sub(p.p, st.site.front_mid), st.site.normal) < -10
            d = dist(p.p, st.site.holder)
            if behind and d < 260 and (best is None or d < best[0]):
                best = (d, p)
        return best[1] if best else None

    def _flee(self, a: AgentView, p: PredatorView, world: WorldState):
        h = heading_of(sub(a.p, p.p))
        return step_toward(a, add(a.p, (40 * math.cos(h), 40 * math.sin(h))), speed_for(a, True))

    def society_override(self, world: WorldState, a: AgentView):
        """Keep ordinary agents out of the held zones."""
        for st in self.stations.values():
            if not (st.held or st.staffed()):
                continue
            c = st.site.held_center()
            d = dist(a.p, c)
            if d < self.P['zone_radius'] and self.role_of(a.id) is None:
                self.metrics['zone_pushes'] += 1
                h = heading_of(sub(a.p, c)) if d > 1e-6 else 0.0
                return step_toward(a, add(a.p, (60 * math.cos(h), 60 * math.sin(h))), a.walk * a.move_modifier), 'zone: leaving the trap area'
        return None

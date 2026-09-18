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

from .geometry import free_point, add, sub, mul, dot, dist, unit, wrap, heading_of, path_clear, polar
from .lure import Lure, Holder, Delivery, predator_target, is_resting, run_in_point, DEFER, steer
from .motion import action, step_toward, hold, speed_for
from .paths import plan, path_length
from .sites import find_sites, Site, CORRIDOR
from .refuge import Refugee, plan_refuge, inside_slot
from .world import AGENT_RADIUS, WorldState, AgentView, PredatorView, PRED_CHARGE_RANGE

DEFAULTS = dict(
    max_stations=2,          # stations kept staffed at once
    max_deliveries=2,        # concurrent deliveries
    guide_min_energy=70.0,   # a guide needs this much to start
    bait_min_energy=65.0,    # below this a bait asks for a successor
    bait_release_energy=40.0,   # below this a bait leaves to eat even without a successor
    successor_min_energy=180.0,
    holder_min_energy=180.0,
    standby_radius=170.0,    # a wall station's standby bait forages within this radius
    relevant_range=420.0,    # a free predator this close to an agent is worth trapping
    max_turn=math.radians(50),   # base steering allowance for a walking guide; sprint energy widens it
    attract_wanderers=False,     # only chased agents become guides (attracting wanderers wastes energy)
    bait_eta_slack=40,           # ticks a bait may arrive after the guide reaches the corridor
    prestaff_time=40.0,      # staff the best station from this time on
    min_workers=3,           # never take the last workers away from foraging
    zone_radius=95.0,        # society agents keep this far from held predators
    guard=True,
    two_baits=False,
    explicit_deliveries=True,    # guide-led deliveries (lure.py)
    refuge=True,                 # chased agents run into the nearest gap
    site_kinds=('gap',),         # narrow gaps first; walls are the backup ('wall', 'gap')
    prestaff=False,              # speculative staffing (senescent baits walking in) cost more than it held
    bait_idle_time=40.0,         # an unheld gap bait leaves after this long without any predator near
    n_traps=2,                   # designated traps: 1 near the map centre, or 2 on opposite sides
    bait_min_life=30.0,          # seconds of life a bait must have left when it arrives
    swap_lead_time=40.0,         # call the replacement when the bait's life falls below walk time + this
    swap_force_life=8.0,         # swap even while predators are awake when the bait has this little life left
    staff_range=230.0,           # staff a station in advance only with a loose predator this close to its mouth
    turn_bonus=math.radians(50), # extra steering allowance for a guide with spare sprint energy
    lead_max=900.0,              # longest lead (guide to entry + corridor + run-in) worth starting
    gap_reserve=True,            # replacement enters behind a dying bait and moves up when it dies (fixture 6/6)
    leash_min_energy=330.0,      # measured: ~200 + 0.09 x lead at p90, plus 100 to sprint at the end
    leash_lead_max=900.0,        # longest leash lead (guide to entry + corridor + run-in)
    hold_bait_min_life=120.0,    # life on arrival for a bait replacing one at a station that holds predators
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
    staffed_since: float = -1e9
    events: list = field(default_factory=list)
    energy_hist: dict = field(default_factory=dict)  # aid -> last energy (senescence detection)
    successor_for: int | None = None
    leaving: int | None = None       # gap: the bait walking out through the far mouth (swap in progress)
    left_at: float = -1e9

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
        self.designated: list[Site] = []       # the one or two traps everything is lured to
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
        self.refugees: dict[int, dict] = {}             # aid -> dict(site, waypoints, slot)
        self.senescent: set = set()                     # agents seen draining fast while idle
        self._refuge_tried: dict = {}                   # aid -> last time a refuge plan was attempted
        self._held_prev: dict = {}
        self._last_why: dict = {}
        self._last_pos: dict = {}
        self._hold_start: dict = {}
        self._energy_seen: dict[int, tuple] = {}        # aid -> (energy, x, y)
        self.last_actions: dict = {}                    # aid -> ActionRequest applied last tick (set by the policy)
        self.fleeing: set = set()                       # agents the society wants to flee this tick

    # ------------------------------------------------------------------ helpers
    def event(self, kind, **data):
        self.events.append(dict(t=round(self.time, 1), kind=kind, **data))

    def role_of(self, aid):
        return self.roles.get(aid, (None, None))[0]

    def _slots(self, site: Site):
        if site.kind == 'wall':
            # two bait slots 10 either side of the face midpoint (Lucas's funnel layout); the
            # second is where a successor stands before the old bait leaves
            return [add(site.holder, mul(site.axis, -10.0)), add(site.holder, mul(site.axis, 10.0))]
        if site.kind == 'gap':
            # front bait and a reserve 10 deeper: the reserve enters behind a bait that is about to
            # die and moves up when it does, so the predators never lose a target
            if not self.P['gap_reserve']:
                return [site.holder]
            return [site.holder, sub(site.holder, mul(site.normal, 10.0))]
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
                    in_place = {b for b, k in st.baits.items() if b in world.agents and k < len(st.slots)
                                and (dist(world.agents[b].p, st.slots[k]) < 3.0 or dist(world.agents[b].p, st.slots[0]) < 14.0)}
                    # a predator resting at the mouth (including the tick it wakes on) still holds
                    resting = bool(p.resting) if p.resting is not None else is_resting(p)
                    if target in in_place or (target is None and resting and in_place):
                        st.held.add(p.pid)
                        st.last_held = world.time
                        held_now[p.pid] = st.key
                        break
        self.metrics['held_predator_ticks'] += len(held_now)
        for st in self.stations.values():
            if len(st.held) > self.metrics.get('max_held_one_station', 0):
                self.metrics['max_held_one_station'] = len(st.held)
        # hold episodes: start/end events with the reason the hold ended
        prev = self._held_prev
        for pid, key in held_now.items():
            if pid not in prev:
                self._hold_start[pid] = world.time
                self.event('hold_started', pid=pid, key=key)
        for pid, key in prev.items():
            if pid not in held_now:
                st = self.stations.get(key)
                p = world.predator(pid)
                if p is None:
                    why = 'predator_gone'
                elif st is None or not st.staffed():
                    why = 'bait_gone'
                elif not any(b in world.agents and dist(world.agents[b].p, st.site.holder) < 3.0 for b in st.baits):
                    why = 'bait_moving'
                    b = next(iter(st.baits))
                    ba = world.agents.get(b)
                    self.event('bait_moving_detail', key=key, agent=b, why=self._last_why.get(b), fleeing=b in self.fleeing,
                               off=round(dist(ba.p, st.site.holder), 1) if ba else None, energy=round(ba.energy) if ba else None,
                               pred=(round(p.x), round(p.y)), target=predator_target(world, p))
                elif (bool(p.resting) if p.resting is not None else is_resting(p)):
                    why = 'resting_elsewhere'
                else:
                    why = 'predator_left'
                extra = {}
                if p is not None and st is not None:
                    dvec = sub(p.p, st.site.front_mid)
                    tangent = (-st.site.normal[1], st.site.normal[0])
                    extra = dict(out=round(dot(dvec, st.site.normal)), lateral=round(abs(dot(dvec, tangent))),
                                 target=predator_target(world, p), penergy=None if p.energy is None else round(p.energy))
                self.event('hold_ended', pid=pid, key=key, why=why, held_s=round(world.time - self._hold_start.get(pid, world.time), 1), **extra)
        self._held_prev = dict(held_now)
        return held_now

    # ------------------------------------------------------------------ main
    def step(self, world: WorldState, fleeing=()):
        self.fleeing = set(fleeing)
        self.time = world.time
        self.decisions = {}
        if not self._sites_done and world.complete_map:
            self.sites = self._usable_sites(world, find_sites(world.rects, world.width, world.height, kinds=self.P['site_kinds']))
            self._sites_done = True
            self._designate(world)
            self.event('sites', walls=sum(s.kind == 'wall' for s in self.sites), gaps=sum(s.kind == 'gap' for s in self.sites))
        elif not world.complete_map and len(world.rects) != getattr(self, '_rects_seen', -1) and int(world.time * 10) % 20 == 0:
            # estimated map: refresh sites every 2 s while rectangles keep appearing; keep stations whose site persists
            self._rects_seen = len(world.rects)
            fresh = self._usable_sites(world, find_sites(world.rects, world.width, world.height, kinds=self.P['site_kinds']))
            known = {s.key: s for s in self.sites}
            self.sites = fresh
            self._designate(world)
            for key, st in list(self.stations.items()):
                match = next((s for s in fresh if dist(s.holder, st.site.holder) < 3.0 and s.kind == st.site.kind), None)
                if match is not None:
                    st.site = match
                    self.stations[match.key] = self.stations.pop(key)
                    for aid, (r, k) in list(self.roles.items()):
                        if k == key:
                            self.roles[aid] = (r, match.key)
            if len(fresh) != len(known):
                self.event('sites', walls=sum(s.kind == 'wall' for s in fresh), gaps=sum(s.kind == 'gap' for s in fresh))
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
                        last = self._last_pos.get(aid)
                        self.event(f'{role}_died', key=key, agent=aid,
                                   at_slot=(last is not None and st is not None and min(dist(last[0], sl) for sl in st.slots) < 3.0),
                                   energy=last[1] if last else None, cause=('predator' if last and any(dist(q.p, last[0]) < 40 for q in world.predators) else 'starvation') if last else None)
        for pid in list(self.deliveries):
            d = self.deliveries[pid]
            if d.guide not in world.agents and d.done is None:
                d.done = 'guide_captured'
                self.metrics['guide_lost'] += 1
                self.event('guide_captured', pid=pid, key=d.site.key, agent=d.guide, phase=d.phase, last=d.decision,
                           after=round(self.time - d.created, 1), max_phase=d.max_phase, trace=d.trace[-8:], ticks=list(d.ticks))
        self._track_senescence(world)
        held_now = self._update_status(world)
        # finished deliveries
        for pid in list(self.deliveries):
            d = self.deliveries[pid]
            if d.done in ('delivered', 'guide_captured'):
                if not hasattr(d, 'ended'):
                    d.ended = self.time
                if d.done == 'delivered' and not d.become_bait and d.guide in world.agents and self.roles.get(d.guide, (None,))[0] == 'guide' \
                        and self.time - d.ended > 2.0:
                    self.roles.pop(d.guide, None)       # its part is done; the society takes it back
                if d.become_bait and d.done == 'delivered' and d.guide in world.agents:
                    st = self._station(d.site)
                    if d.guide not in st.baits:
                        for other in list(st.baits):
                            # a bait still walking in becomes the replacement instead
                            st.baits.pop(other)
                            if st.successor is None:
                                st.successor = other
                                self.roles[other] = ('successor', st.key)
                            else:
                                self.roles.pop(other, None)
                        st.baits[d.guide] = 0
                        st.staffed_since = self.time
                        self.roles[d.guide] = ('bait', st.key)
                if pid in held_now:
                    self.metrics['delivered'] += 1
                    self.event('delivered', pid=pid, key=d.site.key, guide_alive=d.guide in world.agents, leash=d.leash,
                               energy=round(world.agents[d.guide].energy) if d.guide in world.agents else None, after=round(self.time - d.created, 1))
                    if not (d.become_bait and d.guide in world.agents):
                        self.roles.pop(d.guide, None)
                        self.agent_cooldown[d.guide] = self.time + 8.0
                    del self.deliveries[pid]
                elif self.time - d.ended > 12.0:
                    # the predator did not settle on the bait
                    self.metrics['failed'] += 1
                    self.event('delivery_failed', pid=pid, key=d.site.key, reason=d.done, max_phase=d.max_phase, trace=d.trace[-8:])
                    if not (d.become_bait and d.guide in world.agents):
                        self.roles.pop(d.guide, None)
                    self.pred_cooldown[pid] = self.time + 8.0
                    del self.deliveries[pid]
            elif d.done is not None:
                self.metrics['failed'] += 1
                self.event('delivery_failed', pid=pid, key=d.site.key, reason=d.done, phase=d.phase, after=round(self.time - d.created, 1), max_phase=d.max_phase, trace=d.trace[-8:], ticks=list(d.ticks))
                self.roles.pop(d.guide, None)
                self.pred_cooldown[pid] = self.time + (15.0 if d.done.startswith('failed:cannot') else 8.0)
                self.agent_cooldown[d.guide] = self.time + 10.0
                del self.deliveries[pid]
            elif pid in held_now and d.phase in ('FRONT', 'CORRIDOR', 'ENTER'):
                pass
            elif world.predator(pid) is None:
                d.done = 'failed:predator_gone'
        self._transfer_guides(world, held_now)
        self._staff(world)
        self._assign(world, held_now)
        out = self._actions(world, held_now)
        self._last_pos = {aid: (world.agents[aid].p, round(world.agents[aid].energy)) for aid in self.roles if aid in world.agents}
        self._last_why = {aid: why for aid, (act, why) in out.items()}
        return out

    @staticmethod
    def _usable_sites(world: WorldState, sites):
        """Drop sites whose approach or holding points lie in a river or swamp."""
        out = []
        for s in sites:
            pts = (s.front_mid, s.corridor_start, run_in_point(s), s.holder, s.front)
            if min(world.biome_at(q) for q in pts) >= 0.6:
                out.append(s)
        return out

    def _transfer_guides(self, world: WorldState, held_now):
        """A predator that switches to another of our agents mid-delivery is handed to that
        agent (the first capable one), instead of the old guide chasing it."""
        for pid, d in list(self.deliveries.items()):
            if d.done is not None or d.phase not in ('ATTRACT', 'OPEN', 'LEAD'):
                continue
            p = world.predator(pid)
            if p is None:
                continue
            target = predator_target(world, p)
            if target is None or target == d.guide or target not in world.agents:
                continue
            new = world.agents[target]
            if self.role_of(target) is not None:
                continue
            if new.energy < 150.0 or new.id in self.senescent:
                # it now chases someone who cannot sprint: the delivery is over either way
                d.done = 'failed:stolen'
                continue
            old = d.guide
            self.roles.pop(old, None)
            self.agent_cooldown[old] = world.time + 5.0
            d.guide = target
            d.waypoints = []
            d.stall_ticks = 0
            d.defer_ticks = 0
            d.phase = 'LEAD' if dist(new.p, p.p) >= PRED_CHARGE_RANGE + 12 else 'OPEN'
            if d.phase == 'LEAD':
                d.waypoints = Lure(world)._lead_path(new, d.site)
            self.roles[target] = ('guide', d.site.key)
            self.metrics['guide_transfers'] = self.metrics.get('guide_transfers', 0) + 1
            self.event('guide_transferred', pid=pid, key=d.site.key, old=old, new=target, gap=round(dist(new.p, p.p)))

    # ------------------------------------------------------------------ staffing
    def _designate(self, world: WorldState):
        """Pick the traps the colony will use: with one, the best site nearest the map centre; with
        two, a second one at least 500 from the first (on the other side). Sites already in use
        (staffed or holding) keep their designation on a refreshed (estimated) map."""
        n = int(self.P['n_traps'])
        cands = [s for s in self.sites if s.kind == 'gap']
        if not cands:
            self.designated = []
            return
        keep = [s for s in cands if s.key in self.stations and (self.stations[s.key].staffed() or self.stations[s.key].held)]
        centre = (world.width / 2, world.height / 2)
        chosen = list(keep[:n])
        while len(chosen) < n:
            best = None
            for s in cands:
                if any(s.key == c.key for c in chosen):
                    continue
                if chosen and min(dist(s.front_mid, c.front_mid) for c in chosen) < 500.0:
                    continue
                cost = s.score + 0.06 * dist(s.front_mid, centre)
                if best is None or cost < best[0]:
                    best = (cost, s)
            if best is None:
                break
            chosen.append(best[1])
        if [s.key for s in chosen] != [s.key for s in self.designated]:
            self.designated = chosen
            self.event('traps_designated', keys=[s.key for s in chosen])

    def _best_site(self, world: WorldState, exclude_keys=()):
        if not self.sites or not world.agents:
            return None
        cx = sum(a.x for a in world.agents.values()) / len(world.agents)
        cy = sum(a.y for a in world.agents.values()) / len(world.agents)
        best = None
        for s in (getattr(self, 'designated', None) or self.sites):
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
        if P['explicit_deliveries'] and world.time >= P['prestaff_time'] and len(active) < 1 and len(self.stations) < P['max_stations']:
            site = self._best_site(world)
            if site is not None and self._workers(world) > P['min_workers']:
                self._station(site)
        for st in self.stations.values():
            site = st.site
            need_live = bool(st.held) or any(d.site.key == st.key and not d.become_bait for d in self.deliveries.values())
            # wall: baits are assigned when a delivery starts (see _assign); release an idle one
            if site.kind == 'wall':
                if st.baits and not st.held and not need_live and world.time - max(st.last_held, st.staffed_since) > 25.0:
                    for aid in list(st.baits):
                        st.baits.pop(aid)
                        self.roles.pop(aid, None)
                        self.agent_cooldown[aid] = world.time + 20.0
                        self.event('bait_released', key=st.key, agent=aid, why='idle')
            else:
                # gap: keep the station staffed (through the far mouth) so a guide can fly past the
                # mouth and leave the predator to the bait; also re-staff when the bait is gone
                # staff only while a loose predator is near enough to matter: an idle bait costs the
                # colony a forager for nothing
                near = self._predator_near(world, site, held_now=set(st.held))
                want = (st.held or (P['prestaff'] and world.time >= P['prestaff_time'] and near and st is self._primary_station())) \
                    and st.leaving is None
                if want and not st.baits and st.successor is None and self._workers(world) > P['min_workers']:
                    cands = self._bait_candidates(world, site.successor, min_life=P['bait_min_life'], long_hold=bool(st.held))
                    if cands:
                        st.baits[cands[0].id] = 0
                        st.staffed_since = world.time + 30.0   # allow for the walk in
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
                spare = P['min_workers'] + (1 if site.kind == 'gap' else 0)
                room = site.kind == 'gap' or len(st.baits) < len(st.slots)
                if site.kind == 'gap':
                    # a replacement is called when the bait's remaining life would not cover the walk in
                    # (it waits at the stage for up to swap_lead_time), and only while the station
                    # holds something or a loose predator is near
                    useful = st.held or self._predator_near(world, site, held_now=set(st.held))
                    has_reserve = any(k == 1 for k in st.baits.values())
                    cands = self._bait_candidates(world, site.successor, exclude={aid}, min_life=P['bait_min_life'] + P['swap_lead_time'],
                                                  long_hold=bool(st.held), hurry=self._life_s(a) < 45.0) \
                        if useful and st.baits.get(aid) == 0 and not has_reserve and st.successor is None and self._workers(world) > spare else []
                    walk_s = 1.3 * dist(cands[0].p, site.successor) / 100.0 if cands else 0.0
                    tired = self._life_s(a) < walk_s + P['swap_lead_time']
                else:
                    cands = [c for c in self._eligible(world, min_energy=P['successor_min_energy'], max_age=45.0)]
                    cands.sort(key=lambda c: dist(c.p, site.successor))
                    tired = a.energy < P['bait_min_energy'] or senescent or a.age > 95
                if (st.held or site.kind == 'gap') and st.baits.get(aid) == 0 and room and tired and \
                        st.successor is None and self._workers(world) > spare:
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

    def _drain(self, a: AgentView):
        """Energy per second while standing: 1 plus the engine's 0.1 x age once past the hidden max
        age (60-120 s). An agent past 60 that is not yet senescent may turn any second: half rate."""
        if a.id in self.senescent:
            return 1.0 + 0.1 * a.age
        if a.age >= 60.0:
            return 1.0 + 0.05 * a.age
        return 1.0

    def _life_s(self, a: AgentView):
        return a.energy / self._drain(a)

    def _arrival_life(self, a: AgentView, point):
        """Seconds of life left after walking to ``point`` (walking adds 5/s and a 1.5x detour)."""
        walk_s = 1.5 * dist(a.p, point) / 100.0
        return (a.energy - walk_s * (self._drain(a) + 5.0)) / self._drain(a)

    def _bait_candidates(self, world: WorldState, point, exclude=(), min_life=30.0, long_hold=False, hurry=False):
        """Agents to send as bait to ``point``. Speculative staffing (nothing held yet) takes senescent
        agents first (the colony loses them anyway), then the oldest, each with ``min_life`` seconds
        left after the walk (a healthy one twice that, so it can walk back to food). Once predators
        are held (``long_hold``) the bait should last: the healthiest agent with the most life is sent."""
        out = []
        for a in self._eligible(world, exclude=exclude, min_energy=60.0, allow_senescent=True):
            life = self._arrival_life(a, point)
            senescent = a.id in self.senescent
            if long_hold:
                # a bait that should last: young (its hidden max age is at least 60) and well fed;
                # a senescent one only as a stopgap
                young = a.age <= 45.0 and a.energy >= 200.0
                if not young and not (senescent and life >= min_life):
                    continue
                out.append(((0 if young else 1), -a.energy, a))
            else:
                if life < (min_life if senescent else max(2 * min_life, 60.0)):
                    continue
                # senescent agents first (the colony loses them anyway), nearest of those; a
                # healthy agent only when no senescent one can make it
                out.append(((0 if senescent else 1), dist(a.p, point) / 100.0 if (hurry or senescent) else -a.age, a))
        out.sort(key=lambda t: t[:-1])
        return [t[-1] for t in out]

    def _leashable(self, a: AgentView, site: Site):
        """A guide for the close-range lead: sprint energy, a sprint that beats 15, not senescent."""
        return (site.kind == 'gap' and a.energy >= self.P['leash_min_energy'] and a.id not in self.senescent
                and a.age < 58.0 and a.sprint_speed >= 18.0 and a.walk >= 9.0)

    def _predator_near(self, world: WorldState, site: Site, held_now=(), rng=None):
        """A loose, awake predator within ``rng`` (default staff_range) of the mouth."""
        rng = self.P['staff_range'] if rng is None else rng
        for q in world.recent_predators():
            if q.pid in held_now or is_resting(q):
                continue
            if dist(q.p, site.front_mid) < rng:
                return True
        return False

    def _primary_station(self):
        """The station kept staffed in advance: the open one with the best (lowest) site score."""
        opened = [st for st in self.stations.values() if st.site.kind == 'gap']
        if not opened:
            return None
        return min(opened, key=lambda st: st.site.score)

    # ------------------------------------------------------------------ assignment
    def _bait_eta(self, world: WorldState, a: AgentView, slot):
        path = plan(world.rects, world.width, world.height, a.p, slot, radius=6.0, slow=world.biome_at)
        if path is None:
            return None
        return path_length(path) / max(a.walk * a.move_modifier, 1e-6)

    def _assign(self, world: WorldState, held_now):
        """Start deliveries. A predator chasing one of our agents is delivered by that agent
        (it is already fleeing); wanderers are only attracted when ``attract_wanderers`` is on.
        Sites are ranked by the steering the guide needs (a walking guide can only retreat within
        a cone around "directly away") plus the lead length; a wall site also needs a bait that
        can reach the slot before the guide reaches the corridor."""
        P = self.P
        if not P['explicit_deliveries'] or len(self.deliveries) >= P['max_deliveries']:
            return
        busy_guides = {d.guide for d in self.deliveries.values()}
        free = [p for p in world.recent_predators() if p.pid not in held_now and p.pid not in self.deliveries]
        if not free or not self.sites:
            return
        agents = list(world.agents.values())

        def nearest_agent_dist(p):
            return min((dist(a.p, p.p) for a in agents), default=1e9)
        free.sort(key=nearest_agent_dist)
        loose = [q for q in world.recent_predators() if q.pid not in held_now and not is_resting(q)]
        for p in free:
            if len(self.deliveries) >= P['max_deliveries']:
                break
            if nearest_agent_dist(p) > P['relevant_range'] or self.pred_cooldown.get(p.pid, -1.0) > world.time:
                continue
            others = [q for q in loose if q.pid != p.pid]
            target = predator_target(world, p)
            if target is not None and target in world.agents and \
                    any(dist(q.p, world.agents[target].p) < 220 for q in others):
                continue        # a second loose predator near the guide: no delivery survives that
            # guide candidates
            cands = []
            if target is not None and target in world.agents and self.role_of(target) in (None, 'guard') and \
                    world.agents[target].energy >= P['guide_min_energy'] * 0.6 and target not in busy_guides and \
                    dist(world.agents[target].p, p.p) >= 100.0:
                # an agent chased inside 100 is in the society's hands (it flees); we pick it up once
                # the gap has recovered
                cands.append((world.agents[target], 0.0))
            if P['attract_wanderers'] and target is None:
                for a in self._eligible(world, exclude=busy_guides, min_energy=P['guide_min_energy'], allow_senescent=True):
                    goal, k = Lure(world)._intercept_point(a, p, 140.0)
                    if (goal is None or k > 15) and not is_resting(p):
                        continue
                    cands.append((a, (k or 0) / 1.0 + dist(a.p, p.p) / 20.0))
            if not cands:
                continue
            best = None
            for a, attract_cost in cands:
                away = heading_of(sub(a.p, p.p))
                gap = dist(a.p, p.p)
                # steering allowance: base cone, widened by spare sprint energy (sprint-steer)
                allowance = P['max_turn'] + min(1.0, max(0.0, (a.energy - a.max_energy / 5 - 40.0) / 160.0)) * P['turn_bonus']
                if gap > 200:
                    allowance += math.radians(30)     # a far predator leaves room to swing first
                for site in (getattr(self, 'designated', None) or self.sites):
                    st = self.stations.get(site.key)
                    if st is not None and (a.id in st.baits or a.id == st.successor):
                        continue
                    if site.kind == 'gap' and st is not None and st.held and not st.staffed():
                        continue
                    entry = run_in_point(site)
                    if any(dist(q.p, entry) < 220 or dist(q.p, site.front_mid) < 220 for q in others):
                        continue
                    to_entry = heading_of(sub(entry, a.p))
                    axis_dir = heading_of(mul(site.normal, -1.0))
                    turn = abs(wrap(to_entry - away)) + 0.5 * abs(wrap(axis_dir - to_entry))
                    leashable = self._leashable(a, site)
                    if turn > allowance and not leashable:
                        continue
                    if leashable and st is not None and st.staffed() and not site.extra.get('flyby_clear', True):
                        continue        # no room to fly past a staffed mouth here
                    lead = dist(a.p, entry) + CORRIDOR + 250.0
                    if leashable:
                        # the real route (around obstacles, slow ground costed) decides the energy need
                        run_far = add(site.front_mid, mul(site.normal, 200.0))
                        route = plan(world.rects, world.width, world.height, a.p, run_far, radius=12.0, slow=world.biome_at)
                        if route is None:
                            continue
                        lead = path_length(route) + 260.0
                    # arranged-arena measurement: a leash costs ~200 + 0.09 x lead at the 90th percentile
                    # and the guide must keep 100 to sprint at the end
                    if lead > (min(P['leash_lead_max'], (a.energy - 300.0) / 0.09) if leashable else P['lead_max']):
                        continue
                    lead_ticks = lead / 8.0
                    bait = None
                    if site.kind == 'wall':
                        if st is not None and st.staffed():
                            bait_cost = 0.0
                        else:
                            if self._workers(world) <= P['min_workers']:
                                continue
                            slot = self._slots(site)[0]
                            options = []
                            for b in self._eligible(world, exclude=busy_guides | {a.id}, min_energy=P['holder_min_energy'], max_age=45.0):
                                if dist(b.p, slot) > lead_ticks * 10.0 + 200:
                                    continue
                                eta = self._bait_eta(world, b, slot)
                                if eta is None or eta > lead_ticks + P['bait_eta_slack']:
                                    continue
                                options.append((eta, b))
                            if not options:
                                continue
                            eta, bait = min(options, key=lambda o: o[0])
                            bait_cost = eta / 10.0
                    else:
                        bait_cost = 0.0
                    cost = attract_cost + lead / 10.0 + 60.0 * (turn / math.pi) ** 2 + bait_cost + 0.5 * site.score
                    if best is None or cost < best[0]:
                        best = (cost, site, a, bait, round(math.degrees(turn)))
            if best is None:
                continue
            cost, site, a, bait, turn_deg = best
            st = self._station(site)
            if bait is not None:
                st.baits[bait.id] = 0
                st.staffed_since = world.time + 30.0
                self.roles[bait.id] = ('bait', st.key)
                self.event('bait_assigned', key=st.key, agent=bait.id, slot=0, eta=round(self._bait_eta(world, bait, st.slots[0]) or 0))
            bait_in_place = any(b in world.agents and dist(world.agents[b].p, site.holder) < 3.0 for b in st.baits)
            become_bait = site.kind == 'gap' and not bait_in_place
            d = Delivery(site=site, guide=a.id, pid=p.pid, become_bait=become_bait, created=world.time,
                         leash=self._leashable(a, site))
            self.deliveries[p.pid] = d
            self.roles[a.id] = ('guide', st.key)
            busy_guides.add(a.id)
            self.metrics['deliveries'] += 1
            self.event('delivery_started', pid=p.pid, key=st.key, guide=a.id, cost=round(cost, 1), chased=target == a.id, turn=turn_deg,
                       gap=round(dist(a.p, p.p)), leash=d.leash, energy=round(a.energy))

    # ------------------------------------------------------------------ actions
    def _actions(self, world: WorldState, held_now):
        out = {}
        baits = {b for st in self.stations.values() for b in st.baits if b in world.agents and dist(world.agents[b].p, st.slots[st.baits[b]] if st.baits[b] < len(st.slots) else st.site.holder) < 3.0}
        lure = Lure(world, held=set(held_now), baits=baits)
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
                if act is DEFER:
                    continue
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
                if site.kind == 'gap' and st.baits.get(aid, 0) >= 1 and 0 not in st.baits.values():
                    st.baits[aid] = 0
                    slot = st.slots[0]
                    st.staffed_since = world.time
                    self.metrics['handoffs'] += 1
                    self.event('bait_moved_up', key=key, agent=aid)
                if site.kind == 'gap' and at_slot:
                    exit_point = self._gap_exit(world, site, held_now)
                    life = self._life_s(a)
                    # a senescent bait cannot be saved by eating: it serves until it dies
                    starving = a.energy < self.P['bait_release_energy'] and aid not in self.senescent
                    near = any(dist(q.p, site.front_mid) < self.P['relevant_range'] for q in world.predators if q.pid not in held_now)
                    idle = not st.held and not near and world.time - max(st.last_held, st.staffed_since) > self.P['bait_idle_time'] \
                        and aid not in self.senescent
                    my_slot = st.baits.get(aid, 0)
                    reserve = next((b for b, k in st.baits.items() if k == 1 and b != aid), None)
                    reserve_in = reserve is not None and reserve in world.agents and dist(world.agents[reserve].p, st.slots[1]) < 3.0
                    front = next((b for b, k in st.baits.items() if k == 0 and b != aid), None)
                    if my_slot == 0:
                        # the front bait: with a reserve behind it, it serves until it dies
                        leave = (starving or idle) and not reserve_in and world.time - st.opened > 3.0
                        swap = False
                    else:
                        # the reserve: leaves when the front bait will outlast the wait
                        fa = world.agents.get(front) if front is not None else None
                        swap = False
                        leave = fa is not None and self._life_s(fa) > 60.0 and world.time - st.staffed_since > 30.0 \
                            and not (st.held and self._life_s(fa) < 45.0)
                        leave = leave or starving
                    if exit_point is not None and leave:
                        st.baits.pop(aid, None)
                        self.roles[aid] = ('released', key)
                        self.released[aid] = (exit_point, world.time + 12.0)
                        st.leaving = aid
                        st.left_at = world.time
                        self.event('bait_left', key=key, agent=aid, slot=my_slot, why='starving' if starving else ('idle' if my_slot == 0 else 'reserve_idle'), energy=round(a.energy))
                        continue
                if site.kind == 'wall' and at_slot and st.baits.get(aid) == 0 and 1 in st.baits.values():
                    other = next(b for b, k in st.baits.items() if k == 1)
                    ob = world.agents.get(other)
                    if ob is not None and dist(ob.p, st.slots[1]) < 1.5 and (a.energy < self.P['bait_min_energy'] or aid in self.senescent or a.age > 95):
                        # the successor is standing next to us: hand over and go eat
                        st.baits.pop(aid, None)
                        self.roles[aid] = ('released', key)
                        self.released[aid] = (self._exit_point(site, slot), world.time + 12.0)
                        self.metrics['handoffs'] += 1
                        self.event('handoff', key=key, old=aid, new=other)
                        continue
                if site.kind == 'wall' and at_slot and st.baits.get(aid) == 1 and 0 not in st.baits.values():
                    st.baits[aid] = 0
                    slot = st.slots[0]
                    self.event('bait_moved_up', key=key, agent=aid)
                if site.kind == 'gap' and not at_slot:
                    act, why = self._gap_approach(world, holder, a, site, slot, held_now)
                else:
                    act, why = holder.act(a, slot, site, avoid=[c for c in avoid if dist(c[0], slot) > 60])
                out[aid] = (act, why)
            elif role == 'successor' and st is not None:
                site = st.site
                slot_i = st.free_slot()
                if slot_i is None:
                    # both slots taken: the old bait is still alive, wait nearby (outside the held zone)
                    slot_i = 1 if len(st.slots) > 1 else 0
                stage = st.slots[slot_i]
                if aid in self.fleeing and dist(a.p, site.holder) > 40:
                    continue
                if site.kind == 'gap':
                    old_bait = world.agents.get(st.leaving) if st.leaving is not None else None
                    passage_clear = (old_bait is None or dist(old_bait.p, site.holder) > site.length + 6.0
                                     or world.time - st.left_at > 15.0)
                    if passage_clear and st.leaving is not None:
                        st.leaving = None
                    passage_clear = passage_clear and (slot_i == 1 or not st.baits)
                    if not passage_clear:
                        # wait at the stage outside the far mouth until the old bait has walked out;
                        # with a loose predator near the far mouth wait further back instead
                        loose_far = [q for q in world.predators if q.pid not in held_now and not is_resting(q)
                                     and site.far_mouth is not None and dist(q.p, site.far_mouth) < 130]
                        stage_pt = site.successor
                        if loose_far:
                            stage_pt = add(site.successor, mul(site.normal, -90.0))
                        act, why = holder.act(a, stage_pt, site, avoid=avoid)
                        if dist(a.p, stage_pt) < 2.0:
                            act, why = hold(a), ('successor: staged at the far mouth' if not loose_far else 'successor: waiting back, predator at the far mouth')
                        out[aid] = (act, why)
                        if int(world.time * 10) % 30 == 0:
                            self.event('successor_trace', key=key, agent=aid, why=why, d=round(dist(a.p, site.successor)), energy=round(a.energy))
                        continue
                    act, why = self._gap_approach(world, holder, a, site, stage, held_now)
                    if int(world.time * 10) % 30 == 0:
                        self.event('successor_trace', key=key, agent=aid, why=why, d=round(dist(a.p, stage)), energy=round(a.energy))
                else:
                    act, why = holder.act(a, stage, site, avoid=avoid)
                if dist(a.p, stage) < 1.0 and slot_i not in st.baits.values():
                    st.baits[aid] = slot_i
                    st.staffed_since = world.time
                    self.roles[aid] = ('bait', key)
                    st.successor = None
                    st.successor_for = None
                    self.metrics['handoffs'] += 1
                    self.event('successor_in_place', key=key, agent=aid, slot=slot_i)
                    why = 'successor: in place'
                out[aid] = (act, why)
            elif role == 'refugee':
                r = self.refugees.get(aid)
                if r is None:
                    self.roles.pop(aid, None)
                    continue
                site = r['site']
                p = min(world.predators, key=lambda q: dist(q.p, a.p), default=None)
                st2 = self.stations.get(site.key)
                occupied = st2 is not None and any(b in world.agents and b != aid and dist(world.agents[b].p, site.holder) < 3.0 for b in st2.baits)
                if occupied and r.get('side') is None:
                    r['side'] = Lure(world).flyby_side(site, a) or (-site.normal[1], site.normal[0])
                act, why, arrived = Refugee(world).act(a, p, site, r['waypoints'], r['slot'], occupied=occupied and r.get('front', True), side=r.get('side'))
                if arrived and occupied:
                    r['flyby'] = r.get('flyby', 0) + 1
                    if r['flyby'] >= 12:
                        self.roles.pop(aid, None); self.refugees.pop(aid, None)
                        self.agent_cooldown[aid] = world.time + 8.0
                        self.metrics['refuge_flybys'] = self.metrics.get('refuge_flybys', 0) + 1
                        self.event('refuge_flyby', key=site.key, agent=aid)
                        continue
                    out[aid] = (act, why)
                    continue
                if arrived:
                    st2 = self._station(site)
                    st2.baits[aid] = r['slot']
                    st2.staffed_since = world.time
                    if st2.successor is not None and self._life_s(a) >= 60.0:
                        self.roles.pop(st2.successor, None)
                        self.agent_cooldown[st2.successor] = world.time + 10.0
                        self.event('successor_released', key=st2.key, agent=st2.successor, why='refugee is the bait')
                        st2.successor = None
                        st2.successor_for = None
                    while len(st2.slots) <= r['slot']:
                        st2.slots.append(inside_slot(site, len(st2.slots)))
                    self.roles[aid] = ('bait', st2.key)
                    self.refugees.pop(aid, None)
                    self.metrics['refuges'] = self.metrics.get('refuges', 0) + 1
                    self.event('refuge_entered', key=st2.key, agent=aid, slot=r['slot'])
                    continue
                # give up if no predator is near any more and we are still far from the mouth
                if (p is None or dist(p.p, a.p) > 260) and dist(a.p, site.front_mid) > 120 and dist(a.p, site.far_mouth or site.front_mid) > 120:
                    self.roles.pop(aid, None); self.refugees.pop(aid, None)
                    self.event('refuge_abandoned', agent=aid)
                    continue
                out[aid] = (act, why)
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
                    if act is not DEFER:
                        out[aid] = (act, f'guide[{d.phase}] {d.decision}')
                    continue
                station = site.guard
                if aid in self.fleeing and dist(a.p, station) > 5:
                    continue
                act, why = holder.act(a, station, site, avoid=avoid)
                out[aid] = (act, why.replace('holder', 'guard'))
        return out

    def _gap_exit(self, world: WorldState, site: Site, held_now):
        """A clear way out of a gap: the exit point beyond the far mouth, else out of the front
        mouth, else None."""
        for mouth, point in ((site.far_mouth, site.extra.get('exit')), (site.front_mid, add(site.front_mid, mul(site.normal, 70.0)))):
            if mouth is None or point is None:
                continue
            if mouth is site.far_mouth and not site.far_mouth_open:
                continue
            if not any(dist(q.p, mouth) < 80 and not is_resting(q) for q in world.recent_predators()):
                return point
        return None

    def _exit_point(self, site: Site, slot):
        if site.kind == 'wall':
            return add(slot, mul(site.normal, -90.0))
        far = site.far_mouth if site.far_mouth is not None else site.front_mid
        return add(far, mul(site.normal, -45.0))

    def _gap_approach(self, world: WorldState, holder: Holder, a: AgentView, site: Site, goal, held_now):
        """Reach a point inside a gap passage through the far mouth, never through the
        predator crowd at the front mouth; wait back if a loose predator sits at the far mouth
        (predators held at the front mouth do not count, even on a short passage)."""
        far = site.far_mouth if site.far_mouth is not None else site.front_mid
        back = mul(site.normal, -1.0)
        inside_passage = abs(dot(sub(a.p, site.front_mid), site.axis)) < site.length + 5 and \
            abs(dot(sub(a.p, site.front_mid), (-site.axis[1], site.axis[0]))) < site.thickness / 2 + 1.0 and \
            dot(sub(a.p, site.front_mid), site.axis) > -1.0
        if inside_passage:
            return step_toward(a, goal, min(a.walk * a.move_modifier, dist(a.p, goal)), face=goal), 'gap: walking the passage'
        held_here = set(held_now) if isinstance(held_now, (set, dict)) else set()
        loose = [p for p in world.recent_predators() if dist(p.p, far) < 75 and not is_resting(p)
                 and p.pid not in held_here and dot(sub(p.p, site.front_mid), site.normal) < 5.0]
        d_far = dist(a.p, far)
        st = self.stations.get(site.key)
        front = world.agents.get(next((b for b, k in (st.baits.items() if st else []) if k == 0), -1)) if st else None
        urgent = front is not None and self._life_s(front) < 10.0
        if loose and not urgent:
            wait_pt = add(far, mul(back, 110.0))
            if dist(a.p, wait_pt) < 8.0:
                return hold(a), 'gap: waiting back, a loose predator is at the far mouth'
            act, why = holder.act(a, wait_pt, site, avoid=[(site.held_center(), self.P['zone_radius'])])
            return act, 'gap: falling back, a loose predator is at the far mouth'
        if d_far < 45.0 and path_clear(a.p, far, AGENT_RADIUS + 0.5, world.rects):
            return step_toward(a, far, min(a.walk * a.move_modifier, d_far + 2.0), face=far), 'gap: entering the far mouth'
        # approach the far mouth from behind: aim at a point 30 out of it, around the front mouth crowd
        # and around loose predators on the way
        approach = add(far, mul(back, 30.0))
        avoid = [(add(site.front_mid, mul(site.normal, 45.0)), 75.0)]
        avoid += [(q.p, 70.0) for q in world.recent_predators() if q.pid not in held_here and not is_resting(q) and dist(q.p, a.p) < 250]
        sprint = a.can_sprint and a.energy > 200.0 and front is not None and self._life_s(front) < 25.0
        act, why = holder.act(a, approach, site, avoid=avoid, sprint=sprint)
        return act, why.replace('holder', 'gap route')

    def _wild_threat(self, world: WorldState, st: Station, a: AgentView, held_now):
        """A predator that is not held and can reach the bait: on the protected side within 45."""
        for p in world.recent_predators():
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

    def _clear_the_lead(self, world: WorldState, a: AgentView):
        """A bystander in front of a predator that a guide is leading would steal the chase
        (the predator takes the nearest agent it sees). Step sideways out of its cone instead
        of fleeing straight ahead of it."""
        for pid, d in self.deliveries.items():
            if d.done is not None or d.phase not in ('LEAD', 'CORRIDOR') or d.guide == a.id:
                continue
            p = world.predator(pid)
            g = world.agents.get(d.guide)
            if p is None or g is None:
                continue
            axis = unit(sub(g.p, p.p))
            rel = sub(a.p, p.p)
            along = dot(rel, axis)
            lateral = rel[0] * -axis[1] + rel[1] * axis[0]
            if along < -20 or along > 280 or abs(lateral) > 130:
                continue
            side = 1.0 if lateral >= 0 else -1.0
            out = (-axis[1] * side, axis[0] * side)                 # perpendicular, away from the pair's line
            heading = heading_of(add(mul(out, 0.8), mul(unit(rel), 0.4)))
            h = steer(a, p, heading, world.rects, math.radians(80), keep_los=False, slow=world.biome_at)
            self.metrics['lead_clears'] = self.metrics.get('lead_clears', 0) + 1
            return step_toward(a, add(a.p, polar(h, 60.0)), speed_for(a, dist(a.p, p.p) < 70)), 'clearing the lead: stepping out of its cone'
        return None

    def society_override(self, world: WorldState, a: AgentView, fleeing=False):
        """Refuge flight for chased agents; keep ordinary agents out of the held zones."""
        if self.role_of(a.id) is None and self.deliveries:
            o = self._clear_the_lead(world, a)
            if o is not None:
                return o
        if fleeing and self.P['refuge'] and self.role_of(a.id) is None and self.sites and \
                self.agent_cooldown.get(a.id, -1.0) <= world.time:
            p = min(world.predators, key=lambda q: dist(q.p, a.p), default=None)
            danger = p is not None and (dist(p.p, a.p) < 100 or (p.speed > 13 and not a.can_sprint and dist(p.p, a.p) < 160))
            if danger and not is_resting(p) and self._refuge_tried.get(a.id, -1.0) <= world.time - 0.4:
                self._refuge_tried[a.id] = world.time
                held = {pid for st in self.stations.values() for pid in st.held}
                best = plan_refuge(world, a, p, getattr(self, 'designated', None) or self.sites, held)
                if best is not None:
                    site, wps, is_front, cost = best
                    st = self._station(site)
                    taken = set(st.baits.values())
                    for r in self.refugees.values():
                        if r['site'].key == site.key:
                            taken.add(r['slot'])
                    slot = 0
                    while slot in taken:
                        slot += 1
                    self.refugees[a.id] = dict(site=site, waypoints=wps, slot=slot, front=is_front)
                    self.roles[a.id] = ('refugee', site.key)
                    self.event('refuge_run', key=site.key, agent=a.id, slot=slot, front=is_front, gap=round(dist(p.p, a.p)), cost=round(cost))
                    act, why, _ = Refugee(world).act(a, p, site, wps, slot)
                    return act, why
        for st in self.stations.values():
            if not st.held:
                continue
            c = st.site.held_center()
            d = dist(a.p, c)
            if d < self.P['zone_radius'] and self.role_of(a.id) is None:
                self.metrics['zone_pushes'] += 1
                h = heading_of(sub(a.p, c)) if d > 1e-6 else 0.0
                return step_toward(a, add(a.p, (60 * math.cos(h), 60 * math.sin(h))), a.walk * a.move_modifier), 'zone: leaving the trap area'
        return None

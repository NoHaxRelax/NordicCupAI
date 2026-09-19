"""Native-game colony: Nikolaj explores; Oscar gathers; our geometry traps.

Only normal per-agent DTOs and simulation time enter this controller. Estimated
coordinates are never replaced by evaluator coordinates. No engine imports.
"""
import math
import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from src.utils.DTOs import ActionRequest
from models.exploration.expert_policy import ExpertPolicy, load_config
from models.exploration.global_planner import load_planner_config
from models.exploration.navigation import Navigator
from models.exploration.world_estimator import rotate
from models.survival.oscar_orchard import OrchardPolicy, MOVE_PENALTY
from models.entrapment.observed_trap_sites import observed_rectangles, available_sites
from models.entrapment.my_guide import guide
from models.entrapment.bystander_avoidance import avoid_predators
from models.entrapment.bait_nursery import BaitNursery
from models.entrapment.guide_lookahead import chase_step
from models.entrapment.guide_coordinator import GuideCoordinator
from models.entrapment.bait_travel import ObservedBaitTravel
from models.entrapment.foraging_safety import ForagingSafety
from models.entrapment.guide_assignment import detectable_observer


def action_for(aid, **kwargs):
    values = dict(agent_id=aid, move_distance=0., move_direction=0., turn_angle=0., spawn_agent=False)
    values.update(kwargs)
    return ActionRequest(**values)


def remaining_life(energy, age):
    """Conservative idle lifetime: earliest possible senescence is 60 seconds.

    Native old-age loss is .01*age/tick at 10 Hz. A small energy reserve and
    half-tick allowance cover discrete stepping. Food is never assumed.
    """
    energy = max(0., energy - 2.)
    young = min(energy, max(0., 60.-age))
    energy -= young
    age += young
    b = 1. + .1*age + .01
    old = 2*energy / (b + math.sqrt(b*b + .2*energy)) if energy else 0.
    return max(0., young + old - .1)


def local(pose, point):
    return tuple(map(float, rotate(np.asarray(point)-pose.position, -pose.heading)))


@dataclass
class Track:
    key: int
    group: int
    position: np.ndarray
    seen: float
    observers: dict = field(default_factory=dict)
    guide_id: int | None = None
    memory: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    completed_at: float | None = None
    held_since: float | None = None
    guide_last_saw: float | None = None


class EntrapmentPolicy:
    def __init__(self, seed=0, *, bait_overlap_seconds=20., bait_reserve_seconds=0., survival_settings=None,
                 release_trap_food=False, nursery_size=0, bait_food_lead_seconds=6.,
                 guide_lookahead_ticks=3, share_guide_paths=True,
                 guide_preferred_distance=(100.,120.), guide_reacquire_close=False,
                 guide_contact_forecast=False, guide_orbit_recovery=False, guide_coordination=False,
                 bait_terrain_estimate=False, safe_foraging=False, guide_chased_only=False):
        if not math.isfinite(bait_overlap_seconds) or bait_overlap_seconds < 0.:
            raise ValueError('bait_overlap_seconds must be finite and nonnegative')
        self.bait_overlap_seconds = float(bait_overlap_seconds)
        if not math.isfinite(bait_food_lead_seconds) or bait_food_lead_seconds < 0.:
            raise ValueError('bait_food_lead_seconds must be finite and nonnegative')
        self.bait_food_lead_seconds = float(bait_food_lead_seconds)
        self.bait_navigation = {}
        self.bait_terrain_estimate = bait_terrain_estimate
        self.safe_foraging = safe_foraging
        self.guide_chased_only = guide_chased_only
        self.foraging_safety = ForagingSafety()
        self.bait_travel = ObservedBaitTravel()
        self.guide_corridors = []
        if guide_lookahead_ticks not in (0,3):
            raise ValueError('guide_lookahead_ticks must be 0 or 3')
        self.guide_lookahead_ticks = guide_lookahead_ticks
        self.guide_reacquire_close = guide_reacquire_close
        self.guide_contact_forecast = guide_contact_forecast
        self.guide_orbit_recovery = guide_orbit_recovery
        self.guide_coordination = guide_coordination
        self.guide_coordinator = GuideCoordinator()
        low,high = guide_preferred_distance
        if not all(math.isfinite(x) for x in (low,high)) or not 0. < low <= high:
            raise ValueError('guide_preferred_distance must be finite, positive and ordered')
        self.guide_preferred_distance = (float(low),float(high))
        self.share_guide_paths = share_guide_paths
        if not math.isfinite(bait_reserve_seconds) or bait_reserve_seconds < 0.:
            raise ValueError('bait_reserve_seconds must be finite and nonnegative')
        self.bait_reserve_seconds = float(bait_reserve_seconds)
        self.reserved_bait = None
        self.release_trap_food = release_trap_food
        self.nursery = BaitNursery(nursery_size)
        config = load_config()
        # Nikolaj's survey-gap harvesting is deliberately disabled. Our detector
        # alone selects traps. Stay in his exploration phase until we find one.
        config = config.model_copy(update={'harvest': config.harvest.model_copy(update={'enabled': False})})
        planner = load_planner_config().model_copy(update={'population_after_alignment': False})
        self.explorer = ExpertPolicy(config, planner)
        settings = json.loads((Path(__file__).parent / 'survival/oscar_best_config.json').read_text())
        settings.update(survival_settings or {})
        self.orchard = OrchardPolicy(seed=seed, **settings)
        self.site = None
        self.site_group = None
        self.site_frame = None
        self.bait = None
        self.incoming = None
        self.bait_progress = None
        self.bait_retry_after = {}
        self.retired_baits = set()
        self.entered_rear = set()
        self.navigator = Navigator(clearance=5.05)
        self.tracks = {}
        self.next_track = 0
        self.events = []
        self.roles = {}
        self.next_site_check = 0.
        self.map_stats = {}
        self.now = 0.
        self.metrics = dict(site_discoveries=0, guide_assignments=0, guide_deaths=0, guide_releases=0,
                            guide_handovers=0,
                            delivery_arrivals=0, bait_arrivals=0, overlapping_replacements=0,
                            estimated_unbaited_seconds=0., no_viable_bait_ticks=0)
        self.last_time = None

    @property
    def estimator(self):
        return self.explorer.planner.estimator

    def event(self, kind, **data):
        self.events.append(dict(time=round(self.now, 3), kind=kind, **data))

    def _find_site(self):
        if self.now < self.next_site_check: return
        self.next_site_check = self.now + 2.
        candidates = []
        self.map_stats = {}
        for gid, group in self.estimator.groups.items():
            static = observed_rectangles(group)
            self.map_stats[gid] = dict(edges=len(group.edges), anchored=group.anchored,
                                      rectangles=0 if static is None else len(static['obstacles']))
            if static is None: continue
            sites = available_sites(static, keep_corner_sites=(self.site is not None and
                                    self.site.get('site_kind') == 'corner_pocket'))
            self.map_stats[gid]['our_sites'] = len(sites)
            candidates.extend((gid, group.frame_revision, site) for site in sites)
        if self.site is not None:
            # Observe and revalidate, including newly discovered blocking walls.
            same = next((c for c in candidates if c[0] == self.site_group
                         and c[1] == self.site_frame and math.dist(c[2]['goal'], self.site['goal']) <
                         (.02 if self.site.get('site_kind') == 'corner_pocket' else 2.)), None)
            if same:
                return
            self.event('site_invalidated_by_mapping')
            self.site = None
            self.bait = self.incoming = None
            self.reserved_bait = None
            self.bait_progress = None
            self.retired_baits.clear()
            self.entered_rear.clear()
            self.tracks.clear()
        if candidates:
            gid, revision, site = max(candidates, key=lambda c: c[2]['overlap'])
            self.site, self.site_group, self.site_frame = site, gid, revision
            self.navigator.reset()
            self.metrics['site_discoveries'] += 1
            self.event('our_trap_discovered', group=gid, site=site)

    def _arrival(self, aid):
        pose = self.estimator.poses.get(aid)
        return (pose is not None and pose.group_id == self.site_group and
                math.dist(pose.position, self.site['goal']) <= 2.)

    def _select_bait(self, states, excluded):
        choices = []
        rear, goal = np.array(self.site['replacement_entry']), np.array(self.site['goal'])
        for aid, s in states.items():
            pose = self.estimator.poses[aid]
            if (aid in excluded or aid in self.nursery.members or pose.group_id != self.site_group or pose.uncertainty > 8.
                    or self.bait_retry_after.get(aid, 0.) > self.now): continue
            # Route through the rear entrance, never through the predator mass.
            plan = self.navigator.steer(aid, pose.position, rear, self.now)
            if plan.blocked or not math.isfinite(plan.remaining): continue
            distance = plan.remaining + float(np.linalg.norm(rear-goal))
            estimate = self._bait_travel(aid,s,distance,include_goal=True)
            travel,walk_cost = estimate['seconds'],estimate['walk_cost']
            life = remaining_life(s['energy']-walk_cost, s['age'])
            if life < travel + 15.: continue
            old = self.orchard.minds[aid].old or s['age'] >= 55.
            deadline = math.inf if self.bait is None else remaining_life(states[self.bait]['energy'], states[self.bait]['age'])
            choices.append((travel+5. < deadline, aid in self.nursery.children,
                            old and self.bait_reserve_seconds > 0., -travel, old, life-travel,
                            s['energy'], aid, travel))
        return max(choices)[-2:] if choices else None

    def _bait_travel(self, aid, state, distance, *, include_goal):
        if not math.isfinite(distance):
            return dict(seconds=math.inf,walk_cost=math.inf,distance=distance,known_fraction=0.,method='blocked')
        if not self.bait_terrain_estimate:
            return dict(seconds=distance/(max(.1,min(state['speed'],state['sprint_speed']))*.3*10.),
                        walk_cost=distance/.3*.05+6.,distance=distance,known_fraction=0.,method='all_river')
        pose = self.estimator.poses[aid]
        route = self.navigator.routes[aid]
        points = [pose.position]+route.points
        if include_goal: points.append(self.site['goal'])
        return self.bait_travel.estimate(points,state)

    def _bait_roles(self, states):
        if self.site is None: return
        group = self.estimator.groups.get(self.site_group)
        if group is None: return
        self.navigator.update(group, self.now)
        if self.bait_terrain_estimate: self.bait_travel.update(group,self.now)
        self.navigator.prune(states)
        self.retired_baits.intersection_update(states)
        self.entered_rear.intersection_update(states)
        if self.bait not in states:
            # An older overlapping bait may still be holding the trap.
            holding = [aid for aid in self.retired_baits if self._arrival(aid)]
            self.bait = max(holding, key=lambda aid: remaining_life(states[aid]['energy'], states[aid]['age']), default=None)
            self.retired_baits.discard(self.bait)
        if self.incoming not in states:
            self.incoming = None
            self.bait_progress = None
        if self.incoming is not None and self._arrival(self.incoming):
            previous = self.bait
            self.bait, self.incoming = self.incoming, None
            self.bait_progress = None
            self.metrics['bait_arrivals'] += 1
            if previous is not None and self._arrival(previous):
                self.retired_baits.add(previous)
                self.metrics['overlapping_replacements'] += 1
            self.event('bait_arrived', agent=self.bait, previous=previous,
                       energy=states[self.bait]['energy'], age=states[self.bait]['age'],
                       conservative_lifetime_seconds=remaining_life(states[self.bait]['energy'], states[self.bait]['age']))
        # Retry with another candidate if the current traveller gets stuck.
        if self.incoming is not None:
            aid = self.incoming
            pose = self.estimator.poses[aid]
            entered = aid in self.entered_rear
            target = self.site['goal'] if entered else self.site['replacement_entry']
            plan = self.navigator.steer(aid, pose.position, target, self.now)
            distance = plan.remaining
            # Avoidance and ageing can invalidate the dispatch estimate.
            # Reassign early when another viable agent can beat that deadline.
            remaining = distance + (0. if entered else math.dist(target, self.site['goal']))
            estimate = self._bait_travel(aid,states[aid],remaining,include_goal=not entered)
            travel = estimate['seconds']
            life = remaining_life(states[aid]['energy']-estimate['walk_cost'], states[aid]['age'])
            deadline = math.inf if self.bait is None else remaining_life(states[self.bait]['energy'], states[self.bait]['age'])
            if not entered and (travel+5. >= deadline or life < travel+15.):
                alternative = self._select_bait(states, self.retired_baits | {self.bait, aid}
                                               | {t.guide_id for t in self.tracks.values()})
                if alternative is not None and alternative[1]+5. < deadline and (alternative[1] < travel or life < travel+15.):
                    self.event('replacement_reassigned', agent=aid, replacement=alternative[0])
                    self.incoming = None
                    self.bait_progress = None
                    self.bait_retry_after[aid] = self.now+15.
                    self.navigator.release(aid)
            last = self.bait_progress
            if last is None or last[0] != entered or distance < last[1]-2.:
                self.bait_progress = (entered, distance, self.now)
            stalled = self.now-self.bait_progress[2] > 8.
            if stalled and self.incoming is not None:
                self.event('replacement_stalled', agent=aid)
                self.bait_retry_after[aid] = self.now+15.
                self.incoming = None
                self.bait_progress = None
                self.navigator.release(aid)
                self.entered_rear.discard(aid)
        # Dispatch for an estimated overlap at the bait itself. The rear is
        # only a route waypoint; replacements never wait there.
        if self.incoming is None:
            excluded = self.retired_baits | {self.bait} | {t.guide_id for t in self.tracks.values()}
            selected = self._select_bait(states, excluded)
            active_life = 0. if self.bait is None else remaining_life(states[self.bait]['energy'], states[self.bait]['age'])
            food_lead = (self.bait_food_lead_seconds if selected is not None and
                         states[selected[0]]['energy'] < states[selected[0]]['max_energy']-30. else 0.)
            due = self.bait is None or (selected is not None and
                    active_life <= selected[1]+self.bait_overlap_seconds+food_lead)
            # Reserve a donor while it still gathers food. It cannot reproduce
            # or become a guide until the handoff, but never waits at the entry.
            self.reserved_bait = (selected[0] if selected is not None and not due
                                  and self.bait_reserve_seconds > 0.
                                  and active_life <= selected[1]+self.bait_reserve_seconds else None)
            if selected is None:
                self.metrics['no_viable_bait_ticks'] += 1
            elif due:
                self.incoming = selected[0]
                self.bait_progress = None
                self.event('replacement_dispatched', agent=self.incoming, current=self.bait,
                           conservative_travel_seconds=selected[1], active_lifetime_seconds=active_life,
                           target_overlap_seconds=self.bait_overlap_seconds, food_lead_seconds=food_lead)
        if self.incoming is not None: self.reserved_bait = None
        if self.reserved_bait is not None: self.roles[self.reserved_bait] = 'bait_candidate'
        for aid in self.retired_baits: self.roles[aid] = 'retired_bait'
        if self.bait is not None: self.roles[self.bait] = 'bait'
        if self.incoming is not None: self.roles[self.incoming] = 'replacement_bait'
        ready = self.bait is not None and self._arrival(self.bait)
        if not ready and self.last_time is not None:
            self.metrics['estimated_unbaited_seconds'] += self.now-self.last_time

    def _track_predators(self, states):
        self.tracks = {key: t for key, t in self.tracks.items()
                       if self.now-t.seen < (30. if t.guide_id in states else 8.)}
        for track in self.tracks.values():
            track.observers = {}
            if track.guide_id is not None and track.guide_id not in states:
                self.metrics['guide_deaths'] += 1
                self.event('guide_lost', agent=track.guide_id, track=track.key)
                track.guide_id = None
        for aid, s in states.items():
            pose = self.estimator.poses[aid]
            if pose.uncertainty > 12.: continue
            used = set()
            for obs in s['observations']:
                if obs['type'] != 'Predator': continue
                point = pose.position + rotate((obs['distance']*math.cos(obs['angle']),
                                                obs['distance']*math.sin(obs['angle'])), pose.heading)
                matches = [t for t in self.tracks.values() if t.group == pose.group_id and t.key not in used
                           and math.dist(point, t.position) < min(90., 20.+150.*max(0., self.now-t.seen))]
                if matches:
                    track = min(matches, key=lambda t: math.dist(point, t.position))
                    track.position, track.seen = point, self.now
                else:
                    track = Track(self.next_track, pose.group_id, point, self.now)
                    self.next_track += 1
                    self.tracks[track.key] = track
                used.add(track.key)
                track.observers[aid] = obs
        if self.site is None or self.bait is None or not self._arrival(self.bait): return
        if self.guide_coordination:
            self.guide_coordinator.update(self.estimator.groups[self.site_group],states,self.now,self.tracks.values())
        assigned = {t.guide_id for t in self.tracks.values() if t.guide_id is not None}
        for track in self.tracks.values():
            if track.group != self.site_group: continue
            near_bait = math.dist(track.position, self.site['goal']) <= 40.
            track.held_since = (self.now if track.held_since is None else track.held_since) if near_bait else None
            if (track.guide_id in states and track.held_since is not None
                    and self.now-track.held_since >= 10. and self.now-track.seen < .2):
                self.event('guide_released', agent=track.guide_id, track=track.key)
                self.metrics['guide_releases'] += 1
                track.guide_id = None
                track.memory = {}
                track.completed_at = None
            if track.guide_id in states:
                if track.guide_id in track.observers:
                    track.guide_last_saw = self.now
                if not self.guide_coordination:
                    self.roles[track.guide_id] = 'guide'
                    continue
            # Predators already near the trap do not need a second delivery.
            if math.dist(track.position, self.site['goal']) < 60.:
                if track.guide_id in states: self.roles[track.guide_id] = 'guide'
                continue
            available = [aid for aid in track.observers if aid not in self.roles and aid not in assigned]
            if self.guide_chased_only:
                edges = self.estimator.groups[track.group].edges
                available = [aid for aid in available if detectable_observer(track.observers[aid],states[aid],
                    [(local(self.estimator.poses[aid],e.start),local(self.estimator.poses[aid],e.end)) for e in edges])]
            previous = track.guide_id if track.guide_id in states else None
            if self.guide_coordination:
                if not track.edges:
                    track.edges = [(tuple(e.start),tuple(e.end)) for e in self.estimator.groups[track.group].edges]
                fit = []
                for candidate in available:
                    if candidate in self.nursery.members: continue
                    pose = self.estimator.poses[candidate]
                    self.guide_coordinator.observe(track.key,candidate,track.observers[candidate],
                        local(pose,self.site['goal']),[(local(pose,a),local(pose,b)) for a,b in track.edges],
                        states[candidate],round(self.now*10))
                    if self.guide_coordinator.assess(candidate,states[candidate],pose,
                                                    self.site['handoff'],self.now)['viable']:
                        fit.append(candidate)
                available = fit
                if previous is not None:
                    old_pose = self.estimator.poses[previous]
                    old_fit = self.guide_coordinator.assess(previous,states[previous],old_pose,
                                                           self.site['handoff'],self.now)['viable']
                    blind = track.guide_last_saw is None or self.now-track.guide_last_saw > 1.
                    needs_relief = not old_fit or blind or track.memory.get('_predator_not_following',False)
                    # Never release an old guide merely because somebody can
                    # reach it: require consecutive compatible predator motion
                    # toward a nearer, fit observer. Identities are inferred.
                    available = [candidate for candidate in available if needs_relief
                        and self.guide_coordinator.following.get((track.key,candidate),False)
                        and track.observers[candidate]['distance']+old_pose.uncertainty+
                            self.estimator.poses[candidate].uncertainty+2. < math.dist(old_pose.position,track.position)]
                    if not available:
                        self.roles[previous] = 'guide'
                        continue
            if not available: continue
            aid = max(available, key=lambda a: ((self.guide_coordinator.following.get((track.key,a),False)
                                                if self.guide_coordination else False),
                                               -track.observers[a]['distance'] if self.guide_chased_only else 0.,
                                               self.orchard.minds[a].old or states[a]['age'] >= 55.,
                                               states[a]['energy'], -track.observers[a]['distance']))
            track.guide_id = aid
            # Snapshot of observed static geometry gives guiding a stable fixed
            # frame. New local edges still enter native contact avoidance.
            track.edges = [(tuple(e.start), tuple(e.end)) for e in self.estimator.groups[track.group].edges]
            track.memory = {}
            track.completed_at = None
            track.guide_last_saw = self.now
            if previous is not None:
                self.metrics['guide_handovers'] += 1
                self.event('guide_handover', previous=previous, agent=aid, track=track.key,
                           reason='nearer_fit_observer_with_consecutive_following_motion')
                # Keep the old guide out of another assignment this tick; its
                # ordinary action will receive the normal predator avoidance.
            assigned.add(aid)
            self.roles[aid] = 'guide'
            self.metrics['guide_assignments'] += 1
            self.event('guide_assigned', agent=aid, track=track.key,
                       selection='nearest_detectable_observer' if self.guide_chased_only else 'older_observer',
                       viability=self.guide_coordinator.assessments.get(aid) if self.guide_coordination else None)

    def _bait_action(self, aid, s, states):
        pose = self.estimator.poses[aid]
        if aid in self.retired_baits or self._arrival(aid): return action_for(aid)
        rear = self.site['replacement_entry']
        # A past visit to the rear waypoint is not a permanent exemption from
        # predator avoidance. A detour can bring the replacement back around
        # the exposed front; send it to the rear again and restore avoidance.
        if (aid in self.entered_rear and
                float((pose.position-np.asarray(self.site['mouth'])) @ np.asarray(self.site['inward'])) < 0.):
            self.entered_rear.discard(aid)
            self.navigator.release(aid)
            self.event('replacement_left_rear_route',agent=aid)
        if math.dist(pose.position, rear) < 3.: self.entered_rear.add(aid)
        target = self.site['goal'] if aid in self.entered_rear else rear
        plan = self.navigator.steer(aid, pose.position, target, self.now)
        info = dict(target=list(target), phase='inside_rear' if aid in self.entered_rear else 'rear_approach',
                    route_status=plan.status, fruit_skips={}, fruit_selected=None)
        self.bait_navigation[aid] = info
        if plan.blocked or plan.waypoint is None:
            # Retry when geometry changes or a transient collision clears.
            if int(self.now*10) % 30 == 0: self.navigator.release(aid)
            return action_for(aid, turn_angle=.2)
        waypoint = plan.waypoint
        remaining = plan.remaining+(0. if aid in self.entered_rear else math.dist(rear,self.site['goal']))
        estimate = self._bait_travel(aid,s,remaining,include_goal=aid not in self.entered_rear)
        info['travel_estimate'] = estimate
        # No food waits or assumed food gains: even if the fruit disappears,
        # this short detour must leave enough energy and handoff time.
        if (aid not in self.entered_rear and self.bait in states
                and s['energy'] < s['max_energy']):
            deadline = remaining_life(states[self.bait]['energy'], states[self.bait]['age'])
            info['bait_deadline_seconds'] = deadline
            segment = waypoint-pose.position
            length2 = float(segment @ segment)
            options = []
            for o in s['observations']:
                if o['type'] != 'Fruit' or o['distance'] > 60.: continue
                def skip(reason):
                    info['fruit_skips'][reason] = info['fruit_skips'].get(reason,0)+1
                fruit = pose.position+rotate(np.array([o['distance']*math.cos(o['angle']),
                                                       o['distance']*math.sin(o['angle'])]), pose.heading)
                fraction = float(np.clip((fruit-pose.position) @ segment/max(length2, 1e-9), 0., 1.))
                if np.linalg.norm(fruit-pose.position-fraction*segment) > 20.:
                    skip('off_route'); continue
                if not self.navigator._clear(pose.position, fruit) or not self.navigator._clear(fruit, waypoint):
                    skip('wall'); continue
                extra = max(0., float(np.linalg.norm(fruit-pose.position)+np.linalg.norm(waypoint-fruit)-math.sqrt(length2)))
                if extra > 30.:
                    skip('long_detour'); continue
                travel = estimate['seconds']+extra/(max(.1,min(s['speed'],s['sprint_speed']))*.3*10.)+.2
                life = remaining_life(s['energy']-estimate['walk_cost']-extra/.3*.05,s['age'])
                if travel+5. < deadline and life >= travel+15.:
                    options.append((extra, o['distance'], fruit))
                else:
                    skip('handoff_deadline' if travel+5. >= deadline else 'energy_without_fruit')
            if options:
                waypoint = min(options, key=lambda x: x[:2])[2]
                info['fruit_selected'] = waypoint.tolist()
        else:
            info['fruit_ineligible'] = ('already_inside_rear' if aid in self.entered_rear else
                                        'no_current_bait' if self.bait not in states else 'full_energy')
        x, y = local(pose, waypoint)
        angle = math.atan2(y, x)
        distance = min(s['speed'], math.hypot(x, y)/MOVE_PENALTY[s['biome']])
        return action_for(aid, move_distance=distance, move_direction=angle,
                             turn_angle=max(-.3, min(.3, angle)))

    def _guide_action(self, track, s):
        aid = track.guide_id
        pose = self.estimator.poses[aid]
        agent = dict(s)
        track.memory['_lookahead_ticks'] = self.guide_lookahead_ticks
        track.memory['_reacquire_close'] = self.guide_reacquire_close
        track.memory['_contact_forecast'] = self.guide_contact_forecast
        track.memory['_orbit_recovery'] = self.guide_orbit_recovery
        track.memory['_preferred_predator_distance'] = self.guide_preferred_distance
        track.memory['_terrain_samples'] = [
            (local(pose,sample.position),sample.biome,sample.uncertainty)
            for sample in self.estimator.groups[pose.group_id].biomes.values()
            if sample.uncertainty<=8. and math.dist(sample.position,pose.position)<=100.]
        # Use association based exclusively on this agent's observation. All
        # observations remain available for collision/survival steering.
        target = track.observers.get(aid)
        context = dict(tick=round(self.now*10), time=self.now, dt=.1,
                       handoff=local(pose, self.site['handoff']), mouth=local(pose, self.site['mouth']),
                       target_predator=target)
        value = guide(local(pose, self.site['goal']),
                      [(local(pose, a), local(pose, b)) for a, b in track.edges],
                      agent, context, track.memory)
        debug = track.memory.get('debug', {})
        if isinstance(debug, dict) and debug.get('mode') == 'hold_at_delivery' and track.completed_at is None:
            track.completed_at = self.now
            self.metrics['delivery_arrivals'] += 1
            self.event('guide_at_delivery', agent=aid, track=track.key)
        return action_for(aid, **value)

    def _shared_predators(self, aid):
        """Current teammate sightings transformed into this agent's local frame."""
        pose = self.estimator.poses.get(aid)
        if pose is None or pose.uncertainty > 8.: return []
        result = []
        for track in self.tracks.values():
            if (track.group != pose.group_id or self.now-track.seen > .15
                    or aid in track.observers or not track.observers):
                continue
            x, y = local(pose, track.position)
            distance = math.hypot(x, y)
            if distance > 275.: continue
            obs = dict(type='Predator', distance=distance, angle=math.atan2(y, x))
            observer, sighting = next(iter(track.observers.items()))
            source = self.estimator.poses.get(observer)
            if source is not None and 'rel_dir' in sighting:
                toward_observer = math.atan2(source.position[1]-track.position[1],
                                             source.position[0]-track.position[0])
                heading = toward_observer-sighting['rel_dir']
                bearing = math.atan2(pose.position[1]-track.position[1],
                                     pose.position[0]-track.position[0])-heading
                obs['rel_dir'] = math.atan2(math.sin(bearing), math.cos(bearing))
            result.append(obs)
        return result

    def _plan_guides(self, states):
        """Plan guides first so every other agent sees the same intended move."""
        actions = {}
        self.guide_corridors = []
        if self.site is None or self.bait is None:
            return actions
        for track in self.tracks.values():
            aid = track.guide_id
            if aid not in states or self.roles.get(aid) != 'guide':
                continue
            action = self._guide_action(track,states[aid])
            actions[aid] = action
            pose = self.estimator.poses[aid]
            if self.now-track.seen > .15 or pose.uncertainty > 8.:
                continue
            debug = track.memory.get('debug',{})
            forecast = debug.get('forecast',{}) if isinstance(debug,dict) else {}
            path = forecast.get('predator_path')
            if path:
                points = [pose.position+rotate(p,pose.heading) for p in path]
            else:
                # An intentional delivery hold or recovery can bypass search.
                # Extrapolate its actual first command conservatively instead.
                delta = rotate((action.move_distance*MOVE_PENALTY[states[aid]['biome']]*math.cos(action.move_direction),
                                action.move_distance*MOVE_PENALTY[states[aid]['biome']]*math.sin(action.move_direction)),pose.heading)
                points = [track.position]
                for step in range(1,4):
                    points.append(np.asarray(chase_step(points[-1],pose.position+step*delta)))
            self.guide_corridors.append(dict(guide=aid,group=pose.group_id,
                                             frame_revision=self.estimator.groups[pose.group_id].frame_revision,
                                             points=[list(map(float,p)) for p in points]))
        return actions

    def _guided_paths(self, aid, *, rear_approach=False):
        if not self.share_guide_paths:
            return []
        pose = self.estimator.poses.get(aid)
        if pose is None or pose.uncertainty > 8.:
            return []
        result = []
        for corridor in self.guide_corridors:
            if corridor['guide'] == aid or corridor['group'] != pose.group_id:
                continue
            points = corridor['points']
            if rear_approach and all(math.dist(p,self.site['goal']) <= 40. for p in points):
                continue
            if min(math.dist(p,pose.position) for p in points) <= 320.:
                result.append([local(pose,p) for p in points])
        return result

    def __call__(self, states_list, sim_time):
        self.now = sim_time
        states = {s['agent_id']: s for s in states_list}
        if not states: return []
        exploration = {a.agent_id: a for a in self.explorer.actions_for_step(states_list, sim_time)}
        heir_done = {aid: m.heir_done for aid, m in self.orchard.minds.items()}
        unavailable = self.retired_baits | {self.bait, self.incoming} | {t.guide_id for t in self.tracks.values()}
        if self.safe_foraging: self.foraging_safety.update(self,states)
        orchard = dict(self.orchard(states_list, sim_time,
                                    unavailable_agents=unavailable if self.release_trap_food else (),
                                    destination_allowed=self.foraging_safety if self.safe_foraging else None))
        self.roles = {}
        self._find_site()
        if self.site is not None:
            self.navigator.update(self.estimator.groups[self.site_group], self.now)
        self.nursery.update(self, states, 0. if self.bait not in states else
                            remaining_life(states[self.bait]['energy'], states[self.bait]['age']))
        self._bait_roles(states)
        self._track_predators(states)
        guide_actions = self._plan_guides(states)
        self.bait_navigation = {}
        actions = {}
        for aid, s in states.items():
            role = self.roles.get(aid)
            if role in ('bait', 'replacement_bait', 'retired_bait'):
                action = self._bait_action(aid, s, states)
                if role == 'replacement_bait' and not self._arrival(aid):
                    pose = self.estimator.poses[aid]
                    bait_local = local(pose, self.site['goal'])
                    shared = self._shared_predators(aid)
                    avoidance_state = s
                    rear = np.asarray(self.site['replacement_entry'])
                    behind = float((pose.position-np.asarray(self.site['other_mouth'])) @ np.asarray(self.site['inward'])) >= 0.
                    rear_approach = (behind and math.dist(pose.position, rear) <= 25.
                                     and self.navigator._clear(pose.position, rear))
                    if aid in self.entered_rear or rear_approach:
                        # On the final rear approach, permit sensing range of
                        # the observed trap occupants, but avoid other predators.
                        def occupant(o):
                            return (o['type'] == 'Predator' and math.dist(
                                (o['distance']*math.cos(o['angle']),
                                 o['distance']*math.sin(o['angle'])), bait_local) <= 40.)
                        avoidance_state = dict(s, observations=[o for o in s['observations'] if not occupant(o)])
                        shared = [o for o in shared if not occupant(o)]
                    # The blanket keep-out zone is for gatherers. Bait must
                    # reach the rear waypoint; use actual shared sightings here.
                    action, avoiding = avoid_predators(action, avoidance_state, None, shared,
                        self._guided_paths(aid,rear_approach=aid in self.entered_rear or rear_approach))
                    if aid in self.bait_navigation:
                        self.bait_navigation[aid]['avoiding_predator'] = avoiding
                    if avoiding:
                        self.navigator.release(aid)
            elif role == 'guide' and self.site is not None and self.bait is not None:
                action = guide_actions[aid]
            else:
                role = ('nursery_farmer' if aid in self.nursery.members else 'nursery_child'
                        if aid in self.nursery.children else 'bait_candidate'
                        if aid == self.reserved_bait else 'explorer' if self.site is None else 'gatherer')
                action = (self.nursery.action(self,aid,s)
                          if aid in self.nursery.members | self.nursery.children else None)
                if action is None:
                    action = exploration[aid] if self.site is None else orchard[aid]
                bait_local = None
                if self.site is not None and self.bait is not None:
                    pose = self.estimator.poses.get(aid)
                    if pose is not None and pose.group_id == self.site_group:
                        bait_local = local(pose, self.site['goal'])
                action, avoiding = avoid_predators(action, s, bait_local, self._shared_predators(aid), self._guided_paths(aid))
                if avoiding and aid in self.nursery.members:
                    action = action.model_copy(update={'spawn_agent': False})
                if avoiding and aid != self.reserved_bait and aid not in self.nursery.members | self.nursery.children:
                    role = 'avoiding_predator'
                self.roles[aid] = role
            if role in ('bait', 'replacement_bait', 'retired_bait', 'guide', 'bait_candidate', 'nursery_child'):
                action = action.model_copy(update={'spawn_agent': False})
            if orchard[aid].spawn_agent and not action.spawn_agent:
                self.orchard.minds[aid].heir_done = heir_done.get(aid, False)
            distance = min(max(0., action.move_distance), s['sprint_speed'])
            if s['energy'] < s['max_energy']/5: distance = min(distance, s['speed'])
            actions[aid] = action.model_copy(update={'move_distance': distance})
        # Both maps must integrate the action actually returned to the game.
        actual = list(actions.values())
        self.explorer.planner.remember_actions(actual)
        self.explorer.population.remember_actions(actual, sim_time)
        self.explorer.harvest.remember_actions(actual, sim_time)
        self.orchard.last_spawners = []
        for aid, a in actions.items():
            s, m = states[aid], self.orchard.minds[aid]
            m.last_action = (a.move_distance, a.move_direction, a.turn_angle, s['biome'],
                             s['energy'], s['speed'], s['sprint_speed'], s['max_energy'])
            cost = .05*min(a.move_distance, s['speed']) + .5*max(0., a.move_distance-s['speed'])
            cost += min(math.pi, abs(a.turn_angle))/(2*math.pi)
            m.spawned_ok = a.spawn_agent and s['energy']-cost > 100.
            if orchard[aid].spawn_agent and not m.spawned_ok:
                m.heir_done = heir_done.get(aid, False)
            if m.spawned_ok:
                self.orchard.last_spawners.append(aid)
                if aid in self.nursery.members:
                    self.nursery.record_birth(self, aid, s)
            self.explorer.crowd_tracker.remember_turn(aid, a.turn_angle)
            if aid in self.explorer._escape_memories:
                self.explorer._escape_memories[aid].last_turn = a.turn_angle
        self.last_time = sim_time
        return list(actions.items())

    def snapshot(self):
        return dict(phase='exploration' if self.site is None else 'orchard_and_entrapment',
                    bait_overlap_seconds=self.bait_overlap_seconds,
                    bait_food_lead_seconds=self.bait_food_lead_seconds,
                    bait_terrain_estimate=self.bait_terrain_estimate,
                    safe_foraging=self.safe_foraging,foraging_safety=self.foraging_safety.snapshot() if self.safe_foraging else None,
                    guide_chased_only=self.guide_chased_only,
                    guide_lookahead_ticks=self.guide_lookahead_ticks, share_guide_paths=self.share_guide_paths,
                    guide_preferred_distance=self.guide_preferred_distance,
                    guide_reacquire_close=self.guide_reacquire_close,
                    guide_contact_forecast=self.guide_contact_forecast,
                    guide_orbit_recovery=self.guide_orbit_recovery,
                    guide_coordination=self.guide_coordination,
                    guide_coordinator=self.guide_coordinator.snapshot() if self.guide_coordination else None,
                    bait_navigation=self.bait_navigation, guide_corridors=self.guide_corridors,
                    bait_reserve_seconds=self.bait_reserve_seconds, reserved_bait=self.reserved_bait,
                    release_trap_food=self.release_trap_food,
                    nursery=self.nursery.snapshot(),
                    site=self.site, site_group=self.site_group, bait=self.bait, incoming=self.incoming,
                    roles=self.roles.copy(), metrics=self.metrics.copy(), map=self.map_stats,
                    estimated_agents={aid: dict(position=p.position.tolist(), heading=p.heading,
                        uncertainty=p.uncertainty, group=p.group_id) for aid, p in self.estimator.poses.items()},
                    guides={t.guide_id: dict(track=t.key, debug=t.memory.get('debug'))
                            for t in self.tracks.values() if t.guide_id is not None})

"""Opt-in experiments extending the current integrated production policy.

All geometry here comes from public observations and estimated poses. Never
pass an environment, world seed, real coordinates or spectator trap metrics.
"""
from __future__ import annotations

import math

from models.core import EntrapmentPolicy, action_for, local
from models.experiment_config import orchard_kwargs, validate
from models.exploration.expert_policy import ExpertPolicy, ExpertConfig
from models.exploration.global_planner import PlannerConfig
from models.exploration.navigation import Navigator
from models.exploration.exploration import _path_clear
from models.exploration.policy_inputs import observed_edges
from models.survival.oscar_orchard import OrchardPolicy, MOVE_PENALTY, point_segment, wrap
from models.entrapment.bystander_avoidance import avoid_predators


def action_cost(action, state):
    return (.05 * min(action.move_distance, state['speed'])
            + .5 * max(0., action.move_distance - state['speed'])
            + min(math.pi, abs(action.turn_angle)) / math.tau)


def remember_orchard(orchard, actions, states):
    orchard.last_spawners = []
    for aid, action in actions.items():
        s, m = states[aid], orchard.minds[aid]
        m.last_action = (action.move_distance, action.move_direction, action.turn_angle, s['biome'],
                         s['energy'], s['speed'], s['sprint_speed'], s['max_energy'])
        m.spawned_ok = action.spawn_agent and s['energy'] - action_cost(action, s) > 100.
        if m.spawned_ok:
            orchard.last_spawners.append(aid)


def population_and_idle(actions, states, orchard, config, now, eligible, metrics):
    """Optional bounded rescue births and cheaper idle scans, after movement."""
    f, p = config['features'], config['safety']
    if f['late_conservation']:
        fraction = min(1., max(0., (now-p['conservation_start']) /
                                   (p['conservation_end']-p['conservation_start'])))
        factor = 1. - fraction*(1.-p['idle_turn_fraction'])
        for aid in eligible:
            a = actions[aid]
            if a.move_distance == 0 and a.turn_angle and factor < 1:
                actions[aid] = a.model_copy(update={'turn_angle': a.turn_angle*factor})
                metrics['conserved_scan_ticks'] = metrics.get('conserved_scan_ticks', 0) + 1
    if not f['emergency_reproduction']:
        return
    young = sum(s['age'] < p['young_age'] and not orchard.minds[aid].old for aid, s in states.items())
    target = max(p['emergency_min_young'], math.ceil(orchard._cap()*p['emergency_target_fraction']))
    requested = sum(a.spawn_agent for a in actions.values())
    slots = min(p['emergency_births_per_tick'], max(0, target-young-requested))
    candidates = []
    for aid in eligible:
        s, a = states[aid], actions[aid]
        left = s['energy'] - action_cost(a, s)
        if (not a.spawn_agent and s['age'] <= p['emergency_parent_max_age']
                and s['biome'] != 'river' and left > 100.+p['emergency_reserve']):
            candidates.append((left, orchard._fitness(s), aid))
    for _, _, aid in sorted(candidates, reverse=True)[:slots]:
        actions[aid] = actions[aid].model_copy(update={'spawn_agent': True})
        metrics['rescue_birth_requests'] = metrics.get('rescue_birth_requests', 0) + 1


class RiskAwareOrchard(OrchardPolicy):
    def __init__(self, owner, seed, **kwargs):
        self.owner = owner
        super().__init__(seed=seed, **kwargs)

    def safe_food(self, aid, point):
        if not self.owner.experiment['features']['risk_aware_food']:
            return True
        d, angle = self.minds[aid].pose.local(point)
        target = (d*math.cos(angle), d*math.sin(angle))
        # Convert through this agent's local frame, not by equating the two maps.
        return all(point_segment(target, a, b) >= radius
                   for a, b, radius in self.owner.danger_zones(aid))

    def _site_ok(self, group, tree, aid):
        return super()._site_ok(group, tree, aid) and self.safe_food(aid, tree.p)

    def _assign_fruits(self, group, states):
        if not self.owner.experiment['features']['risk_aware_food']:
            return super()._assign_fruits(group, states)
        pairs = []
        for aid in group.agents:
            mind, state = self.minds[aid], states[aid]
            if mind.fruit is not None:
                fruit = group.fruits.get(mind.fruit)
                if fruit is None or fruit.claimed != aid or not self.safe_food(aid, fruit.p):
                    if fruit is not None and fruit.claimed == aid:
                        fruit.claimed = None
                    mind.fruit = None
            if mind.fruit is not None or state['age'] > self.P['no_eat_age']:
                continue
            reach = 60. if mind.old else self.P['fruit_reach']
            for fruit in group.near_fruits(mind.pose.p, reach):
                if (fruit.claimed is not None or not self._ready(fruit, state['energy'], mind.old)
                        or not self.safe_food(aid, fruit.p)):
                    continue
                heir = (not mind.heir_done and state['age'] >= self.P['heir_age']-5.
                        and state['energy'] < self.P['heir_reserve']+20.)
                full = state['energy'] > state['max_energy']-30.
                bucket = 10 if mind.old or aid in self.culled else (0 if heir else 9 if full else int(state['energy']//60))
                pairs.append((bucket, -self._fitness(state), math.dist(fruit.p, mind.pose.p), aid, fruit.id))
        taken = set()
        for _, _, _, aid, fid in sorted(pairs):
            if aid not in taken and group.fruits[fid].claimed is None:
                group.fruits[fid].claimed = aid
                self.minds[aid].fruit = fid
                taken.add(aid)


def configure_guide(values):
    """One policy per fresh process: these are policy heuristics, not engine globals."""
    from models.entrapment import my_guide, guide_steering, guide_pathfinding, predator_following
    for name in ('hearing_target', 'vision_target', 'half_cone_target', 'safe_distance',
                 'bait_buffer', 'contact_buffer', 'trapped_radius'):
        setattr(guide_steering, name.upper(), values[name])
    # my_guide imports this name by value.
    my_guide.HEARING_TARGET = values['hearing_target']
    for name in ('lost_wait_ticks', 'reacquire_arrival_distance', 'delivery_arrival_distance'):
        setattr(my_guide, name.upper(), values[name])
    for name in ('position_tolerance', 'heading_tolerance', 'mismatch_ticks'):
        setattr(predator_following, name.upper(), values[name])
    # The path planner binds its default at function definition time.
    # Explicit radius=10 in predator-motion inference remains unchanged.
    guide_pathfinding.RoutePlanner.__init__.__defaults__ = (values['predator_clearance'], 0.)


class ExperimentalEntrapment(EntrapmentPolicy):
    def __init__(self, config, seed=0):
        self.experiment = validate(config)
        super().__init__(seed=seed, include_corner_pockets=config['features']['corner_pockets'])
        expert = ExpertConfig.model_validate(config['expert'])
        if config['features']['consistent_escape']:
            expert = expert.model_copy(update={'perception': expert.perception.model_copy(
                update={'predator_danger_radius': config['safety']['danger_radius']})})
        if config['features']['escape_memory']:
            expert = expert.model_copy(update={'memory': expert.memory.model_copy(
                update={'predator_escape_seconds': config['safety']['memory_seconds']})})
        self.explorer = ExpertPolicy(expert, PlannerConfig.model_validate(config['planner']))
        self.orchard = RiskAwareOrchard(self, seed, **orchard_kwargs(config['trapping_orchard']))
        self.navigator = Navigator(**config['navigator'])
        configure_guide(config['guide'])
        self.current_states = {}
        self.track_frames = {}
        self.guide_started, self.guide_contact, self.track_cooldown = {}, {}, {}
        self.geometry_checked = {}
        self.extra_metrics = {}

    def _select_bait(self, states, excluded):
        if (self.experiment['features']['role_budget']
                and len(set(states)-excluded) <= self.experiment['safety']['minimum_workers']):
            return None
        return super()._select_bait(states, excluded)

    def _guide_action(self, track, state):
        if self.experiment['features']['refresh_guide_geometry']:
            group = self.estimator.groups.get(track.group)
            if group is None or group.frame_revision != self.site_frame:
                return action_for(track.guide_id, turn_angle=.2)
            checked = self.geometry_checked.get(track.key, -math.inf)
            if self.now-checked >= self.experiment['safety']['guide_geometry_interval']:
                revision = (group.frame_revision, getattr(group, '_edge_revision', 0), len(group.edges))
                if track.memory.get('_observed_geometry_revision') != revision:
                    track.edges = [(tuple(e.start), tuple(e.end)) for e in group.edges]
                    # Geometry determines the guide's fixed frame. Old navigation
                    # and following caches cannot survive a changed frame.
                    track.memory = {'_observed_geometry_revision': revision}
                    self.extra_metrics['guide_geometry_refreshes'] = self.extra_metrics.get('guide_geometry_refreshes', 0)+1
                self.geometry_checked[track.key] = self.now
        return super()._guide_action(track, state)

    def _track_predators(self, states):
        previous = {t.key: t.guide_id for t in self.tracks.values()}
        super()._track_predators(states)
        self.track_frames = {k: v for k, v in self.track_frames.items() if k in self.tracks}
        self.geometry_checked = {k: v for k, v in self.geometry_checked.items() if k in self.tracks}
        for key, track in self.tracks.items():
            group = self.estimator.groups.get(track.group)
            if group is not None and track.seen == self.now:
                self.track_frames[key] = group.frame_revision
        if not self.experiment['features']['role_budget']:
            return
        p = self.experiment['safety']
        reserved = sum(role in ('bait', 'replacement_bait', 'retired_bait') for role in self.roles.values())
        limit = max(0, min(math.floor(len(states)*p['guide_fraction']), len(states)-reserved-p['minimum_workers']))
        accepted = 0
        # Keep established guides before accepting additional assignments.
        tracks = sorted(self.tracks.values(), key=lambda t: (previous.get(t.key) != t.guide_id, t.key))
        for track in tracks:
            aid = track.guide_id
            if aid is None:
                continue
            key = (track.key, aid)
            self.guide_started.setdefault(key, self.now)
            self.guide_contact.setdefault(key, self.now)
            if aid in track.observers:
                self.guide_contact[key] = self.now
            stalled = track.completed_at is None and (
                self.now-self.guide_started[key] > p['guide_timeout']
                or self.now-self.guide_contact[key] > p['guide_lost_seconds'])
            if accepted >= limit or stalled or self.now < self.track_cooldown.get(track.key, -math.inf):
                self.roles.pop(aid, None)
                track.guide_id = None
                track.memory = {}
                track.completed_at = None
                self.guide_started.pop(key, None)
                self.guide_contact.pop(key, None)
                if stalled:
                    self.track_cooldown[track.key] = self.now+p['guide_cooldown']
                self.extra_metrics['guide_releases'] = self.extra_metrics.get('guide_releases', 0)+1
                self.event('experimental_guide_released', agent=aid, track=track.key, stalled=stalled)
            else:
                accepted += 1
        active = {(t.key, t.guide_id) for t in self.tracks.values() if t.guide_id is not None}
        self.guide_started = {k: v for k, v in self.guide_started.items() if k in active}
        self.guide_contact = {k: v for k, v in self.guide_contact.items() if k in active}
        self.track_cooldown = {k: v for k, v in self.track_cooldown.items() if v > self.now and k in self.tracks}

    def danger_zones(self, aid):
        """Local circles/capsules; reject uncertain or incompatible shared frames."""
        p, flags = self.experiment['safety'], self.experiment['features']
        state = self.current_states[aid]
        radius = p['danger_radius'] if flags['consistent_escape'] else self.explorer.config.perception.predator_danger_radius
        zones = []
        for obs in state['observations']:
            if obs['type'] == 'Predator':
                point = (obs['distance']*math.cos(obs['angle']), obs['distance']*math.sin(obs['angle']))
                zones.append((point, point, radius))
        pose = self.estimator.poses.get(aid)
        if pose is None or pose.uncertainty > p['uncertainty_limit']:
            return zones
        group = self.estimator.groups.get(pose.group_id)
        if group is None:
            return zones
        if flags['shared_danger']:
            for key, track in self.tracks.items():
                if (track.group != pose.group_id or self.now-track.seen > p['shared_ttl']
                        or self.track_frames.get(key) != group.frame_revision):
                    continue
                point = local(pose, track.position)
                zones.append((point, point, p['shared_radius'] + pose.uncertainty*p['uncertainty_padding']))
                guide_pose = self.estimator.poses.get(track.guide_id)
                if (guide_pose is not None and guide_pose.group_id == pose.group_id
                        and guide_pose.uncertainty <= p['uncertainty_limit']):
                    zones.append((point, local(pose, guide_pose.position), p['lure_radius']))
        if (flags['trap_exclusion'] and self.site is not None and self.site_group == pose.group_id
                and self.site_frame == group.frame_revision and self.bait in self.current_states
                and self._arrival(self.bait)):
            mouth = local(pose, self.site['mouth'])
            goal = local(pose, self.site['goal'])
            zones.append((mouth, goal, p['trap_radius']))
        return zones

    def steer_safe(self, aid, state, action, zones):
        p, flags = self.experiment['safety'], self.experiment['features']
        terrain = MOVE_PENALTY[state['biome']]
        requested = action.move_distance*terrain
        desired = (requested*math.cos(action.move_direction), requested*math.sin(action.move_direction))
        close = any(point_segment((0., 0.), a, b) < radius for a, b, radius in zones)
        # Distance between two short segments, including crossings.
        def segment_distance(a, b, c, d):
            from models.survival.oscar_orchard import segments_cross
            if segments_cross(a, b, c, d):
                return 0.
            return min(point_segment(a, c, d), point_segment(b, c, d),
                       point_segment(c, a, b), point_segment(d, a, b))
        horizon = max(1., 10*p['lookahead_seconds'])
        future = (desired[0]*horizon, desired[1]*horizon)
        crossing = any(segment_distance((0., 0.), future, a, b) < radius for a, b, radius in zones)
        escaping = self.roles[aid] == 'avoiding_predator'
        if not (close or crossing or escaping):
            return action
        edges = observed_edges(state['observations']) if flags['safe_steering'] else ()
        cap = state['speed']
        if close and state['energy'] >= max(state['max_energy']/5, p['sprint_reserve']):
            cap = state['sprint_speed']
        angles = [action.move_direction] + [i*math.tau/p['angle_candidates'] for i in range(p['angle_candidates'])]
        lengths = {0., min(cap, action.move_distance), min(cap, state['speed']/2), min(cap, state['speed']), cap}
        best = None
        for length in sorted(lengths):
            for angle in angles if length else (action.move_direction,):
                distance = length*terrain
                point = (distance*math.cos(angle), distance*math.sin(angle))
                if not _path_clear(angle, distance, edges, p['wall_margin']):
                    continue
                no_approach = all(segment_distance((0., 0.), point, a, b) >=
                                  min(point_segment((0., 0.), a, b), radius)-1e-6 for a, b, radius in zones)
                clearance = min((point_segment(point, a, b)-radius for a, b, radius in zones), default=0.)
                turn = max(-.5, min(.5, wrap(angle))) if length else action.turn_angle
                candidate = action.model_copy(update={'move_distance': length, 'move_direction': wrap(angle),
                                                       'turn_angle': turn, 'spawn_agent': False})
                cost = action_cost(candidate, state)
                rank = (cost+1 >= state['energy'], not no_approach, clearance < 0,
                        -clearance if clearance < 0 else 0., math.dist(point, desired)+.2*cost)
                if best is None or rank < best[0]:
                    best = rank, candidate
        self.extra_metrics['safety_override_ticks'] = self.extra_metrics.get('safety_override_ticks', 0)+1
        self.roles[aid] = 'avoiding_predator'
        return best[1] if best else action_for(aid, turn_angle=.3)

    def __call__(self, states_list, sim_time):
        """The production dispatch order, with optional hooks before final memory."""
        self.now = sim_time
        self.decision_trace = {}
        states = {s['agent_id']: s for s in states_list}
        self.current_states = states
        if not states:
            return []
        exploration = {a.agent_id: a for a in self.explorer.actions_for_step(states_list, sim_time)}
        orchard = dict(self.orchard(states_list, sim_time))
        self.roles = {}
        self._find_site()
        self._bait_roles(states)
        self._track_predators(states)
        flags, p = self.experiment['features'], self.experiment['safety']
        actions = {}
        for aid, state in states.items():
            role = self.roles.get(aid)
            if role in ('bait', 'replacement_bait', 'retired_bait'):
                action = self._bait_action(aid, state)
            elif role == 'guide' and self.site is not None and self.bait is not None:
                track = next(t for t in self.tracks.values() if t.guide_id == aid)
                action = self._guide_action(track, state)
            else:
                role = 'explorer' if self.site is None else 'gatherer'
                action = exploration[aid] if self.site is None else orchard[aid]
                self._trace(aid, role, action)
                observed = flags['consistent_escape'] and any(
                    o['type'] == 'Predator' and o['distance'] < p['danger_radius'] for o in state['observations'])
                remembered = flags['escape_memory'] and aid in self.explorer._escape_memories
                if observed or remembered:
                    action, role = exploration[aid], 'avoiding_predator'
                    self._trace(aid, 'observed_escape' if observed else 'remembered_escape', action)
                bait_local = None
                if self.site is not None and self.bait is not None:
                    pose = self.estimator.poses.get(aid)
                    if pose is not None and pose.group_id == self.site_group:
                        bait_local = local(pose, self.site['goal'])
                action, avoiding = avoid_predators(action, state, bait_local, self.experiment['bystander'],
                                                   shared_predators=self._shared_predators(aid))
                if avoiding:
                    role = 'avoiding_predator'
                    self._trace(aid, 'bystander_avoidance', action)
                self.roles[aid] = role
                if flags['safe_steering'] or flags['shared_danger'] or flags['trap_exclusion']:
                    action = self.steer_safe(aid, state, action, self.danger_zones(aid))
                    self._trace(aid, 'experimental_safe_steering', action)
                    role = self.roles[aid]
            self._trace(aid, role, action)
            if role not in ('explorer', 'gatherer') or self.orchard.minds[aid].old or state['age'] >= 55.:
                action = action.model_copy(update={'spawn_agent': False})
            self._trace(aid, 'reproduction_role_or_age_limit', action)
            distance = min(max(0., action.move_distance), state['sprint_speed'])
            if state['energy'] < state['max_energy']/5:
                distance = min(distance, state['speed'])
            actions[aid] = action.model_copy(update={'move_distance': distance})
            self._trace(aid, 'native_movement_or_low_energy_limit', actions[aid])
        eligible = {aid for aid in states if self.roles[aid] in ('explorer', 'gatherer')
                    and not any(point_segment((0., 0.), a, b) < r for a, b, r in self.danger_zones(aid))}
        population_and_idle(actions, states, self.orchard, self.experiment, sim_time, eligible, self.extra_metrics)
        for aid, action in actions.items():
            self._trace(aid, 'emergency_birth_or_idle_conservation', action)
        # Supply the final movement to both maps before their next odometry update.
        actual = list(actions.values())
        self.explorer.planner.remember_actions(actual)
        self.explorer.population.remember_actions(actual, sim_time)
        self.explorer.harvest.remember_actions(actual, sim_time)
        remember_orchard(self.orchard, actions, states)
        for aid, action in actions.items():
            self.explorer.crowd_tracker.remember_turn(aid, action.turn_angle)
            if aid in self.explorer._escape_memories:
                self.explorer._escape_memories[aid].last_turn = action.turn_angle
        self.last_time = sim_time
        return list(actions.items())

"""Bounded, evaluator-only telemetry. Never imported by the policy worker.

Wrappers delegate to the pinned engine. Native call sites identify consumption,
rot and death causes; whole-step deltas cannot distinguish these events.
"""
from __future__ import annotations

import ast
from collections import Counter, deque
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import textwrap
import time

VERSION = 2
DEFAULTS = dict(enabled=True, sample_seconds=1., tail_seconds=30.,
                checkpoint_seconds=10., post_event_seconds=10., max_event_clips=3,
                memory_mb=64, disk_mb=128, screenshots=True)


def validate_settings(settings):
    value = {**DEFAULTS, **settings}
    if set(value) != set(DEFAULTS):
        raise ValueError('Unknown diagnostic setting')
    for name in ('enabled', 'screenshots'):
        if type(value[name]) is not bool:
            raise ValueError(f'diagnostics.{name} must be boolean')
    for name in ('sample_seconds', 'tail_seconds', 'checkpoint_seconds', 'post_event_seconds', 'memory_mb', 'disk_mb'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(f'Invalid diagnostics.{name}')
    if type(value['max_event_clips']) is not int or not 0 <= value['max_event_clips'] <= 10:
        raise ValueError('Diagnostic event clip count must be 0-10')
    return value


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode(value):
    return json.dumps(clean(value), separators=(',', ':'), allow_nan=False).encode()


def atomic(path, value):
    from scripts.research_io import atomic_bytes
    atomic_bytes(path, encode(value)+b'\n')


def call_sites(method, name, kinds):
    lines, first = inspect.getsourcelines(method)
    tree = ast.parse(textwrap.dedent(''.join(lines)))
    calls = sorted(node.lineno+first-1 for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == name)
    if len(calls) != len(kinds):
        raise RuntimeError(f'Pinned engine telemetry contract changed: {name}')
    return dict(zip(calls, kinds))


class Diagnostics:
    def __init__(self, env, folder, request, dt):
        self.env, self.folder, self.dt = env, Path(folder), dt
        self.config = validate_settings(request.get('diagnostics', {}))
        self.folder.mkdir(parents=True, exist_ok=True)
        self.request = request
        self.tick, self.now = 0, env.time
        self.roles, self.policy, self.previous_roles = {}, {}, {}
        self.totals, self.death_causes = Counter(), Counter()
        self.agents, self.fruit_birth, self.predator_ids = {}, {}, {}
        self.pending_events, self.ring, self.clips = [], deque(), []
        self.ring_bytes, self.written, self.truncated_frames = 0, 0, 0
        self.collect_seconds, self.serialize_seconds = 0., 0.
        self.next_sample, self.next_checkpoint = 0., self.config['checkpoint_seconds']
        self.pending_clip, self.last_clip_tick = None, -1
        self.originals, self.pre_maintenance, self.meal_energy = {}, {}, {}
        self.closed = False
        self.compiled = hasattr(env, 'pop_diagnostics')
        if self.compiled:
            step_source = (Path(__file__).resolve().parents[1]/'fastsim/_engine.cpp').read_bytes()
        else:
            self.native_step = type(env).non_agent_step
            self.fruit_sites = call_sites(self.native_step, 'remove_fruit', ['consumed', 'rotted'])
            self.death_sites = call_sites(self.native_step, 'kill_agent', ['energy_depletion', 'predation'])
            step_source = inspect.getsource(self.native_step).encode()
        for agent in env.agents:
            self.register(agent, None, initial=True)
        self.fruit_birth = {f.fruit_id: env.time for f in env.fruits}
        static = dict(width=env.width, height=env.height,
                      obstacles=[dict(x=o.x, y=o.y, width=o.width, height=o.height) for o in env.obstacles])
        atomic(self.folder/'static.json', static)
        atomic(self.folder/'manifest.json', dict(version=VERSION, config=self.config,
            case_id=request['case_id'], seed=request['seed'], policy_seed=request['policy_seed'],
            dt=dt, engine='fastsim' if self.compiled else 'python',
            engine_step_sha256=hashlib.sha256(step_source).hexdigest(),
            frame_contract='pre-action truth + official inputs + policy output; events occur during this tick',
            hidden_state='evaluator only; never sent to policy'))
        self.events_file = (self.folder/'events.jsonl').open('wb')
        self.series_file = (self.folder/'series.jsonl').open('wb')
        self._install()

    def register(self, agent, parent, initial=False):
        if agent.agent_id in self.agents:
            return
        self.agents[agent.agent_id] = dict(agent_id=agent.agent_id, parent=parent,
            born=self.now, initial=initial, last_meal=None, longest_meal_gap=0.,
            role_seconds=Counter(), ledger=Counter(), death=None)
        if not initial:
            self.totals['births'] += 1
            self.event('birth', agent_id=agent.agent_id, parent=parent, energy=agent.energy)

    def event(self, kind, **data):
        item = dict(tick=self.tick, sim_time=self.now, kind=kind, evidence='measured', **data)
        self.pending_events.append(item)

    def ledger(self, agent, name, amount):
        if agent.agent_id in self.agents:
            self.agents[agent.agent_id]['ledger'][name] += amount
            self.totals[name] += amount

    def _cause(self, frame, sites):
        if frame.f_code is self.native_step.__code__:
            return sites.get(frame.f_lineno, 'unknown')
        return 'unknown'

    def _maintenance(self, agent):
        before = self.pre_maintenance.pop(agent.agent_id, None)
        if before is not None:
            self.ledger(agent, 'maintenance_energy', before-agent.energy)
        self.meal_energy[agent.agent_id] = agent.energy

    def _install(self):
        env = self.env
        if self.compiled:
            env.enable_diagnostics(True)
            return
        names = ('update_entity_position', 'update_entity_direction', 'spawn_agent',
                 'agent_step', '_get_local_objects', 'non_agent_step', 'remove_fruit',
                 'spawn_fruit', 'kill_agent')
        self.originals = {name: getattr(env, name) for name in names}

        def position(entity, distance, *args, **kwargs):
            before, point = entity.energy, (entity.x, entity.y)
            result = self.originals['update_entity_position'](entity, distance, *args, **kwargs)
            if hasattr(entity, 'agent_id'):
                self.ledger(entity, 'movement_energy', before-entity.energy)
                self.ledger(entity, 'requested_distance', max(0., distance))
                moved = math.dist(point, (entity.x, entity.y))
                self.ledger(entity, 'actual_distance', moved)
                if distance > .1 and moved < .01:
                    self.ledger(entity, 'blocked_seconds', self.dt)
            return result

        def direction(entity, *args, **kwargs):
            before = entity.energy
            result = self.originals['update_entity_direction'](entity, *args, **kwargs)
            if hasattr(entity, 'agent_id'):
                self.ledger(entity, 'turning_energy', before-entity.energy)
            return result

        def spawn_agent(*args, **kwargs):
            parent = kwargs.get('parent', args[2] if len(args) > 2 else None)
            result = self.originals['spawn_agent'](*args, **kwargs)
            if result is not None:
                self.register(result, parent.agent_id if parent is not None else None)
            return result

        def agent_step(agent_id, *args, **kwargs):
            agent = env.agents_dict.get(agent_id)
            before = agent.energy if agent is not None else 0.
            costs = self.agents[agent_id]['ledger'] if agent is not None else Counter()
            prior = costs['movement_energy']+costs['turning_energy']
            result = self.originals['agent_step'](agent_id, *args, **kwargs)
            if agent is not None:
                residual = before-agent.energy-(costs['movement_energy']+costs['turning_energy']-prior)
                self.ledger(agent, 'reproduction_energy', residual)
            return result

        def objects(creature):
            if hasattr(creature, 'agent_id') and sys._getframe(1).f_code is self.native_step.__code__:
                self._maintenance(creature)
            return self.originals['_get_local_objects'](creature)

        def non_agent(dt):
            self.pre_maintenance = {a.agent_id: a.energy for a in env.agents}
            self.meal_energy = {}
            return self.originals['non_agent_step'](dt)

        def remove_fruit(fruit):
            frame = sys._getframe(1)
            cause = self._cause(frame, self.fruit_sites)
            if fruit.fruit_id in env.fruits_dict:
                if cause == 'consumed':
                    agent = frame.f_locals['agent']
                    before = self.meal_energy.get(agent.agent_id)
                    absorbed = agent.energy-before if before is not None else None
                    if absorbed is None:
                        raise RuntimeError('Missing pre-consumption energy')
                    self.meal_energy[agent.agent_id] = agent.energy
                    self.ledger(agent, 'fruits_eaten', 1)
                    self.ledger(agent, 'fruit_gross_energy', fruit.energy)
                    self.ledger(agent, 'fruit_absorbed_energy', absorbed)
                    self.ledger(agent, 'fruit_cap_waste', fruit.energy-absorbed)
                    row = self.agents[agent.agent_id]
                    row['longest_meal_gap'] = max(row['longest_meal_gap'], self.now-(row['last_meal'] if row['last_meal'] is not None else row['born']))
                    row['last_meal'] = self.now
                    self.event('fruit_consumed', agent_id=agent.agent_id, fruit_id=fruit.fruit_id,
                        role=self.roles.get(str(agent.agent_id), 'unknown'), gross=fruit.energy,
                        absorbed=absorbed, cap_waste=fruit.energy-absorbed,
                        age_seconds=self.now-self.fruit_birth.get(fruit.fruit_id, self.now))
                else:
                    self.totals['fruits_'+cause] += 1
                    self.event('fruit_'+cause, fruit_id=fruit.fruit_id, energy=fruit.energy)
                self.fruit_birth.pop(fruit.fruit_id, None)
            return self.originals['remove_fruit'](fruit)

        def spawn_fruit(*args, **kwargs):
            fruit = self.originals['spawn_fruit'](*args, **kwargs)
            if fruit is not None:
                self.fruit_birth[fruit.fruit_id] = self.now
                self.totals['fruits_spawned'] += 1
            return fruit

        def kill_agent(agent):
            frame = sys._getframe(1)
            cause = self._cause(frame, self.death_sites)
            if agent.agent_id in env.agents_dict:
                if cause == 'energy_depletion':
                    self._maintenance(agent)
                predator = frame.f_locals.get('predator') if cause == 'predation' else None
                self.death_causes[cause] += 1
                death = dict(sim_time=self.now, cause=cause, energy=agent.energy, age=agent.age,
                             role=self.roles.get(str(agent.agent_id), 'unknown'))
                self.agents[agent.agent_id]['death'] = death
                self.event('death', agent_id=agent.agent_id, predator_id=self.predator_id(predator) if predator else None, **{k:v for k,v in death.items() if k != 'sim_time'})
            return self.originals['kill_agent'](agent)

        for name, wrapper in dict(update_entity_position=position, update_entity_direction=direction,
                spawn_agent=spawn_agent, agent_step=agent_step, _get_local_objects=objects,
                non_agent_step=non_agent, remove_fruit=remove_fruit, spawn_fruit=spawn_fruit,
                kill_agent=kill_agent).items():
            setattr(env, name, wrapper)

    def predator_id(self, predator):
        if self.compiled:
            return predator.predator_id
        if predator not in self.predator_ids:
            self.predator_ids[predator] = len(self.predator_ids)
        return self.predator_ids[predator]

    def world(self):
        fields = ('x', 'y', 'direction', 'size', 'energy', 'max_energy', 'age', 'max_age',
                  'speed', 'sprint_speed', 'hearing_radius', 'vision_radius', 'cone_angle', 'resting')
        def creature(c):
            return {key:getattr(c, key) for key in fields if hasattr(c, key)}
        return dict(agents=[dict(creature(a), agent_id=a.agent_id) for a in self.env.agents],
            predators=[dict(creature(p), predator_id=self.predator_id(p)) for p in self.env.predators],
            fruits=[dict(fruit_id=f.fruit_id, x=f.x, y=f.y, radius=f.radius, energy=f.energy,
                         engine_age=f.age) for f in self.env.fruits],
            trees=[dict(x=t.x, y=t.y, radius=t.radius, age=t.age) for t in self.env.trees])

    def before(self, tick, observations):
        started = time.perf_counter()
        self.tick, self.now = tick, self.env.time
        self.frame = dict(tick=tick, sim_time=self.now, world=self.world(), observations=observations)
        self.collect_seconds += time.perf_counter()-started

    def decision(self, actions, audit):
        started = time.perf_counter()
        self.policy = audit.get('policy_debug', {})
        self.roles = {str(k):v for k,v in self.policy.get('roles', {}).items()}
        for key, role in self.roles.items():
            if self.previous_roles.get(key) != role:
                self.event('role_change', agent_id=int(key), previous=self.previous_roles.get(key), role=role)
        self.previous_roles = self.roles.copy()
        self.frame.update(actions=actions, policy=self.policy,
                          policy_metrics=audit.get('metrics', {}), trapping_metrics=audit.get('trapping_metrics', {}))
        for agent in self.env.agents:
            row = self.agents[agent.agent_id]
            row['role_seconds'][self.roles.get(str(agent.agent_id), 'unknown')] += self.dt
            self.ledger(agent, 'living_agent_seconds', self.dt)
            if agent.energy < agent.max_energy/5:
                self.ledger(agent, 'low_energy_seconds', self.dt)
        self.collect_seconds += time.perf_counter()-started

    def _line(self, stream, item):
        data = encode(item)+b'\n'
        if self.written+len(data) > self.config['disk_mb']*1024**2*.6:
            raise RuntimeError('Essential diagnostic history exceeded per-run disk allowance')
        stream.write(data)
        self.written += len(data)

    def _clip(self, name, frames):
        started = time.perf_counter()
        data = gzip.compress(b'\n'.join(row[2] for row in frames)+b'\n', compresslevel=1, mtime=0)
        existing = sum(p.stat().st_size for p in self.folder.glob('*.gz') if p.name != name)
        if existing+self.written+len(data) > self.config['disk_mb']*1024**2*.9:
            raise RuntimeError('Diagnostic clips exceeded per-run disk allowance')
        target = self.folder/name
        from scripts.research_io import atomic_bytes
        atomic_bytes(target, data)
        self.serialize_seconds += time.perf_counter()-started
        return dict(file=name, first_tick=frames[0][0] if frames else None,
                    last_tick=frames[-1][0] if frames else None,
                    first_sim_time=frames[0][1] if frames else None,
                    last_sim_time=frames[-1][1] if frames else None,
                    captured_seconds=frames[-1][1]-frames[0][1]+self.dt if frames else 0.,
                    sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    def _append_frame(self):
        raw = encode(self.frame)
        self.ring.append((self.tick, self.now, raw))
        self.ring_bytes += len(raw)
        while self.ring and self.ring[0][1] < self.env.time-self.config['tail_seconds']-self.config['post_event_seconds']:
            self.ring_bytes -= len(self.ring.popleft()[2])
        while self.ring and self.ring_bytes > self.config['memory_mb']*1024**2:
            self.ring_bytes -= len(self.ring.popleft()[2])
            self.truncated_frames += 1
        if not self.ring:
            raise RuntimeError('Single diagnostic frame exceeds memory allowance')

    def after(self):
        started = time.perf_counter()
        if self.compiled:
            from scripts.research_native_diagnostics import consume
            consume(self)
        events, self.pending_events = self.pending_events, []
        for event in events:
            self._line(self.events_file, event)
        self.frame['events'] = events
        self.frame['post_sim_time'] = self.env.time
        self._append_frame()
        deaths = [e for e in events if e['kind'] == 'death']
        trigger = any(e.get('role') in ('bait', 'replacement_bait') for e in deaths) or len(deaths) >= 2
        if trigger and self.pending_clip is None and len(self.clips) < self.config['max_event_clips'] and self.tick > self.last_clip_tick+int(self.config['tail_seconds']/self.dt):
            self.pending_clip = dict(trigger_tick=self.tick, end=self.env.time+self.config['post_event_seconds'])
        if self.pending_clip and self.env.time >= self.pending_clip['end']:
            clip = self._clip(f'event-{len(self.clips)+1}.jsonl.gz', list(self.ring))
            self.clips.append({**clip, **self.pending_clip})
            self.last_clip_tick, self.pending_clip = self.tick, None
        if self.env.time+1e-7 >= self.next_sample:
            energies = [a.energy/a.max_energy for a in self.env.agents]
            sample = dict(tick=self.tick, sim_time=self.env.time, alive=len(energies),
                roles=Counter(self.roles.get(str(a.agent_id), 'unknown') for a in self.env.agents),
                energy_fraction_min=min(energies, default=None),
                energy_fraction_median=statistics.median(energies) if energies else None,
                predators=len(self.env.predators), fruits=len(self.env.fruits), trees=len(self.env.trees),
                totals=dict(self.totals), deaths=dict(self.death_causes),
                policy_metrics=self.frame.get('policy_metrics'), trapping_metrics=self.frame.get('trapping_metrics'))
            self._line(self.series_file, sample)
            self.next_sample = self.env.time+self.config['sample_seconds']
        if self.env.time+1e-7 >= self.next_checkpoint:
            self.events_file.flush()
            self.series_file.flush()
            clip = self._clip('recent.jsonl.gz', list(self.ring))
            atomic(self.folder/'checkpoint.json', dict(last_durable_tick=self.tick, sim_time=self.env.time,
                   recent=clip, clips=self.clips, truncated_frames=self.truncated_frames))
            self.next_checkpoint = self.env.time+self.config['checkpoint_seconds']
        self.collect_seconds += time.perf_counter()-started

    def finish(self, status):
        if self.closed:
            return self.summary
        self.closed = True
        if self.compiled:
            from scripts.research_native_diagnostics import consume
            consume(self)
            self.env.enable_diagnostics(False)
        for name, method in self.originals.items():
            setattr(self.env, name, method)
        for event in self.pending_events:
            self._line(self.events_file, event)
        self.events_file.flush()
        self.series_file.flush()
        self.events_file.close()
        self.series_file.close()
        if getattr(self, 'frame', None) and (not self.ring or self.ring[-1][0] != self.frame['tick']):
            self.frame.update(events=self.pending_events, incomplete_step=True)
            self._append_frame()
        frames = [row for row in self.ring if row[1] >= self.env.time-self.config['tail_seconds']]
        tail = self._clip('terminal.jsonl.gz', frames)
        for row in self.agents.values():
            end = row['death']['sim_time'] if row['death'] else self.env.time
            row['longest_meal_gap'] = max(row['longest_meal_gap'], end-(row['last_meal'] if row['last_meal'] is not None else row['born']))
        meals = self.totals['fruits_eaten']
        self.summary = dict(version=VERSION, status=status, folder=str(self.folder),
            totals=dict(self.totals), deaths=dict(self.death_causes),
            mean_gross_energy_per_fruit=self.totals['fruit_gross_energy']/meals if meals else None,
            mean_absorbed_energy_per_fruit=self.totals['fruit_absorbed_energy']/meals if meals else None,
            absorbed_energy_per_agent_second=self.totals['fruit_absorbed_energy']/max(self.totals['living_agent_seconds'], 1e-9),
            collection_wall_seconds=self.collect_seconds, clip_serialization_wall_seconds=self.serialize_seconds,
            timing_note='Collection excludes engine hook and policy export costs; clip timing partly overlaps collection. Use paired runs to measure total overhead.',
            truncated_frames=self.truncated_frames, terminal=tail, event_clips=self.clips,
            terminal_world=self.world(), last_tick=self.tick,
            complete=status in ('horizon', 'extinct'), screenshots=[])
        atomic(self.folder/'agents.json', self.agents)
        if self.config['screenshots'] and frames:
            try:
                from scripts.research_render import render_frames
                self.summary['screenshots'] = render_frames(self.folder, frames)
            except Exception as exc:
                self.summary['screenshot_error'] = f'{type(exc).__name__}: {exc}'
        self.summary['bytes_on_disk'] = sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file())
        atomic(self.folder/'summary.json', self.summary)
        if sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file()) > self.config['disk_mb']*1024**2:
            raise RuntimeError('Diagnostic artifacts exceeded per-run allowance')
        return self.summary

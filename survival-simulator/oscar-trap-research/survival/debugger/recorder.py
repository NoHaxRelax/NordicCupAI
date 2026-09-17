"""Read-only replay capture for an upstream Survival Simulator Environment.

Attach to any local policy loop. No random draws, engine modifications or network.
Debugger state includes fields that player policies cannot observe.
"""
from __future__ import annotations
import base64
import copy
import gzip
import io
import json
import math
import os
import platform
import weakref
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_COMMIT = 'acfc31a4003a5f91bf11032a02cd98c178ddbd7e'


def number(value):
    return round(float(value), 4)


def json_value(value):
    """Convert numpy scalar values without retaining engine object references."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float):
        return number(value)
    if hasattr(value, 'item'):
        return json_value(value.item())
    return value


def action_dict(action):
    if action is None:
        return None
    if hasattr(action, 'model_dump'):
        return json_value(action.model_dump())
    return json_value(dict(action))


class ReplayRecorder:
    def __init__(self, env, *, title='Survival replay', policy='custom', seed=None,
                 dt=.1, every=1, scenario='generated', notes='', policy_sha256=None,
                 native_render=False, native_width=960):
        if every < 1:
            raise ValueError('every must be at least 1')
        self.env = env
        self.every = every
        self.tick = 0
        self.frames = []
        self.events = []
        self._ids = {'predator': weakref.WeakKeyDictionary(), 'tree': weakref.WeakKeyDictionary()}
        self._next = {'predator': 0, 'tree': 0}
        self._previous = None
        self.native = NativeRenderer(env, width=native_width) if native_render else None
        self.meta = dict(title=title, policy=policy, seed=seed, source_commit=SOURCE_COMMIT,
                         dt=dt, record_interval=number(every*dt), scenario=scenario,
                         notes=notes, policy_sha256=policy_sha256,
                         created_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                         observation_note='Engine observations precede predator movement. Last actions use the prior decision input. Hidden debug fields are not policy inputs.')
        if self.native:
            self.meta.update(renderer='upstream Environment.draw', native_resolution=list(self.native.screen.get_size()),
                             rendering_note='Captured with original simulator drawing methods. Controlled maps use a flat native biome color because their fixture has no generated texture. Native hearing/vision overlays are baked into each image.')
        self.world = dict(width=env.width, height=env.height,
                          background=self._background(),
                          obstacles=[dict(x=number(o.x), y=number(o.y), width=number(o.width), height=number(o.height)) for o in env.obstacles])

    def _id(self, kind, entity):
        ids = self._ids[kind]
        if entity not in ids:
            ids[entity] = self._next[kind]
            self._next[kind] += 1
        return ids[entity]

    def _background(self):
        import pygame
        import numpy as np
        colors = {'forest': (38, 72, 53), 'grassland': (75, 99, 64),
                  'swamp': (72, 81, 68), 'desert': (136, 117, 80), 'river': (44, 92, 119)}
        # Flat biome colours are easier to inspect than the original noisy texture.
        pixels = np.zeros((self.env.width, self.env.height, 3), dtype=np.uint8)
        unique = {id(b): b for b in self.env.biome_map.flat}
        for biome in unique.values():
            pixels[self.env.biome_map == biome] = colors.get(biome.type, (65, 73, 67))
        surface = pygame.surfarray.make_surface(pixels)
        buffer = io.BytesIO()
        pygame.image.save(surface, buffer, 'biomes.png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

    def _creature(self, creature, identity):
        return dict(id=identity, **{key:number(getattr(creature, key)) for key in
             ('x','y','direction','size','energy','max_energy','speed','sprint_speed','hearing_radius')},
             vision_range=number(creature.vision_radius), vision_angle=number(creature.cone_angle))

    def _entities(self):
        e = self.env
        agents = {a.agent_id: dict(x=number(a.x), y=number(a.y), energy=number(a.energy)) for a in e.agents}
        predators = {self._id('predator',p):dict(x=number(p.x), y=number(p.y),resting=p.resting) for p in e.predators}
        fruits = {f.fruit_id:dict(x=number(f.x),y=number(f.y),age=number(f.age)) for f in e.fruits}
        return dict(agents=agents, predators=predators, fruits=fruits)

    def _events(self):
        first_new_event = len(self.events)
        now = self._entities()
        previous = self._previous or dict(agents={}, predators={}, fruits={})
        t = number(self.env.time)
        def event(kind, entity_type, identity, text, point):
            self.events.append(dict(t=t, type=kind, entity_type=entity_type,
                                    entity_id=identity, text=text, x=point['x'], y=point['y']))
        for aid, point in now['agents'].items():
            if aid not in previous['agents']:
                text = f'Agent #{aid} present at start' if self._previous is None else f'Agent #{aid} born'
                event('birth','agent',aid,text,point)
        for aid, point in previous['agents'].items():
            if aid not in now['agents']:
                event('death','agent',aid,f'Agent #{aid} died; position is last recorded location, cause not inferred',point)
        for pid, point in now['predators'].items():
            old = previous['predators'].get(pid)
            if old is None:
                event('predator_spawn','predator',pid,f'Predator #{pid} appeared',point)
            elif old['resting'] != point['resting']:
                label = 'rest' if point['resting'] else 'wake'
                event(label,'predator',pid,f'Predator #{pid} '+('rests' if label=='rest' else 'wakes'),point)
        # Food removal is ambiguous: an old fruit may be eaten before the rot pass.
        # Do not invent an eater or cause. Filter these events in the UI if noisy.
        for fid, point in previous['fruits'].items():
            if fid not in now['fruits']:
                event('fruit_removed','fruit',fid,f'Fruit #{fid} disappeared (eaten or rotted)',point)
        self._previous = now
        return any(event['type'] != 'fruit_removed' for event in self.events[first_new_event:])

    def capture(self, actions=(), decisions=None, inputs=None, action_t=None, *, force=False):
        """Call after every step. Full frames follow `every`; events track all ticks.

        actions/decisions/inputs must describe the actions just applied, not the
        policy's next action. `inputs` are original observation DTO dictionaries.
        """
        important_event = self._events()
        should_save = force or important_event or self.tick % self.every == 0
        self.tick += 1
        if not should_save:
            return
        if self.frames and self.frames[-1]['t'] == number(self.env.time):
            return
        e = self.env
        action_map = dict(actions)
        input_map = {s['agent_id']:s for s in (inputs or [])}
        decisions = decisions or {}
        agents=[]
        for a in e.agents:
            row=self._creature(a,a.agent_id)
            ix=max(0,min(e.width-1,int(a.x)));iy=max(0,min(e.height-1,int(a.y)))
            row.update(age=number(a.age),max_age=number(a.max_age),biome=e.biome_map[ix,iy].type,
                       action=action_dict(action_map.get(a.agent_id)),decision=decisions.get(a.agent_id),
                       action_t=None if action_t is None else number(action_t),
                       observations=json_value(e.agent_observations.get(a.agent_id,[])),
                       action_observations=json_value(input_map.get(a.agent_id,{}).get('observations',[])))
            agents.append(row)
        predators=[]
        for p in e.predators:
            row=self._creature(p,self._id('predator',p));row['resting']=bool(p.resting)
            predators.append(row)
        self.frames.append(dict(t=number(e.time),score=number(e.score),agents=agents,predators=predators,
            fruits=[dict(id=f.fruit_id,**{k:number(getattr(f,k)) for k in ('x','y','radius','energy','age')}) for f in e.fruits],
            trees=[dict(id=self._id('tree',tree),**{k:number(getattr(tree,k)) for k in ('x','y','radius','age')}) for tree in e.trees]))
        if self.native:
            self.frames[-1]['native_image'] = self.native.capture()

    def save(self, path, *, reason='recording stopped', overwrite=False):
        path=Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f'Replay already exists: {path}')
        self.capture(force=True)
        summary=dict(duration=number(self.env.time),score=number(self.env.score),frames=len(self.frames),reason=reason)
        data=dict(format='survival-replay',version=1,meta=self.meta,world=self.world,
                  frames=self.frames,events=self.events,summary=summary)
        path.parent.mkdir(parents=True,exist_ok=True)
        # Publish only a fully closed recording. Concurrent discovery must never
        # mistake a file still being written for a completed replay.
        descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(temporary,'wt',encoding='utf8') as f:
                json.dump(data,f,separators=(',',':'),allow_nan=False)
            if overwrite:
                os.replace(temporary,path)
            else:
                # Atomic no-clobber publication, including concurrent writers.
                os.link(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)
        return summary


class NativeRenderer:
    """Call the upstream renderer on isolated render layers, preserving gameplay.

    Generated environments reuse their original biome/obstacle textures. Minimal
    controlled fixtures get native obstacle drawing over a flat biome palette.
    Creature.draw may initialize its vision cache; restore that cache afterward.
    """
    def __init__(self, env, width=960):
        import pygame
        import numpy as np
        if not isinstance(width, int) or not 160 <= width <= 2400:
            raise ValueError('native_width must be an integer between 160 and 2400')
        self.env = env
        self.screen = pygame.Surface((width, max(1, round(width * env.height / env.width))))
        size = (env.width, env.height)
        self.layers = {name: pygame.Surface(size, pygame.SRCALPHA) for name in
                       ('world_surface', 'vision_screen', 'leaf_screen')}
        if hasattr(env, 'static_surface'):
            self.static = env.static_surface
            self.shadow = env.shadow_surface
            self.obstacles = env.obstacle_surface
        else:
            pixels = np.zeros((env.width, env.height, 3), dtype=np.uint8)
            for biome in {id(b): b for b in env.biome_map.flat}.values():
                pixels[env.biome_map == biome] = biome.color_palette[0]
            self.static = pygame.surfarray.make_surface(pixels)
            self.shadow = pygame.Surface(size, pygame.SRCALPHA)
            self.obstacles = pygame.Surface(size, pygame.SRCALPHA)
            for obstacle in env.obstacles:
                obstacle.draw_shadow(self.shadow, tuple(max(0, c-50) for c in obstacle.color))
                obstacle.draw(self.obstacles)

    def capture(self):
        import pygame
        render_env = copy.copy(self.env)
        for name, surface in self.layers.items():
            setattr(render_env, name, surface)
        render_env.static_surface = self.static
        render_env.shadow_surface = self.shadow
        render_env.obstacle_surface = self.obstacles
        caches = [(creature, creature._vision_poly) for creature in self.env.agents + self.env.predators]
        try:
            render_env.draw(self.screen)
        finally:
            for creature, cache in caches:
                creature._vision_poly = cache
        buffer = io.BytesIO()
        pygame.image.save(self.screen, buffer, 'native.png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

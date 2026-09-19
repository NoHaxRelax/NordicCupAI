"""
Interactive viewer for a local survival-simulator game.

Step through the game one tick at a time, play it at different speeds, and go back in
time. Click an agent or a predator to see its observations and its next action.

Usage (from the survival-simulator folder):
    python scripts/game_viewer.py [--seed 123] [--policy module.path:function] [--history-mb 1024]

The policy must have the same signature as
src.utils.controllers.dummy_agent_policy.action_decision: (observation_dict, rng) -> ActionRequest.

Controls:
    Space               play / pause
    Right or .          one tick forward
    Left or ,           one tick back (pauses playback)
    Shift + Left/Right  10 ticks back / forward
    PageUp / PageDown   100 ticks back / forward
    Home / End          first tick / latest simulated tick
    Timeline            click or drag the bar at the bottom of the panel
    Up / Down           faster / slower playback
    Left click          select the agent or predator under the cursor
    Tab                 select the next agent
    Esc                 clear the selection
    Mouse wheel         scroll the info panel
    R                   restart with the same seed
    Q                   quit

Going back in time:
    The viewer saves a snapshot of the game after every tick. Moving forward from a
    past tick loads the saved future, so the game plays out exactly as before, and new
    ticks are only simulated past the latest one. When the snapshots exceed
    --history-mb, older ticks are thinned to one every 50. Jumping to a thinned tick
    loads the nearest earlier snapshot and simulates forward from it. If your policy
    keeps its own internal state, that state is not rewound.

Predators decide inside the simulation step, after the agents have moved. The
panel therefore shows two things for a predator:
  - the observation and action from the tick that just ran (what it actually did)
  - a preview of its next action, computed from the current state before the agents
    move. The real decision can differ because the agents move first.
"""
import argparse
import copy
import importlib
import io
import math
import pickle
import random
import sys
import zlib
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import SimulationCore  # noqa: E402
from src.elements.predator import Predator  # noqa: E402

PANEL_WIDTH = 540
FPS = 60
SPEEDS = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200]  # ticks per second
MAX_TICKS_PER_FRAME = 40
MAX_TIME = 3000
KEYFRAME_EVERY = 50  # ticks kept when old history is thinned
KEEP_RECENT = 500  # never thin the most recent ticks
TIMELINE_H = 14

BG = (24, 26, 30)
TEXT = (220, 222, 228)
DIM = (140, 144, 152)
HEAD = (255, 205, 90)
SELECT = (0, 230, 255)
BAD = (255, 90, 90)
OBS_COLORS = {
    "Fruit": (255, 140, 200),
    "Agent": (120, 200, 255),
    "Predator": (255, 80, 80),
    "Tree": (120, 255, 120),
    "Edge": (255, 255, 0),
}


# ----------------------------------------------------------------------------
# Record what each predator saw and did during the real simulation step.
# ----------------------------------------------------------------------------
_original_predator_step = Predator.step
_current_tick = [0]


def _recording_predator_step(self, observation=None):
    signals = _original_predator_step(self, observation)
    self._viz_obs = observation
    self._viz_signals = dict(signals)
    self._viz_tick = _current_tick[0]
    return signals


Predator.step = _recording_predator_step


# ----------------------------------------------------------------------------
# Snapshots. Parts of the environment that never change after creation (render
# surfaces, biome map, obstacles) are shared between snapshots instead of copied.
# ----------------------------------------------------------------------------
class _SnapshotPickler(pickle.Pickler):
    def __init__(self, file, shared):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.shared = shared

    def persistent_id(self, obj):
        return id(obj) if id(obj) in self.shared else None


class _SnapshotUnpickler(pickle.Unpickler):
    def __init__(self, file, shared):
        super().__init__(file)
        self.shared = shared

    def persistent_load(self, pid):
        return self.shared[pid]


def shared_env_objects(env):
    objs = [env.biome_map, env.obstacles, env.edges, env.grid_obstacles, env.grid_edges, *env.obstacles]
    objs += [v for v in vars(env).values() if isinstance(v, pygame.Surface)]
    return {id(o): o for o in objs}


def load_policy(spec):
    module_name, func_name = spec.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def fmt_angle(a):
    return f"{a:+.3f} rad ({math.degrees(a):+.0f}°)"


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def fmt_obs(o):
    """Format one observation dict as a single line."""
    t = o.get("type")
    if t == "Edge":
        (sx, sy), (ex, ey) = o["coords"]
        return f"Edge   ({sx:7.1f},{sy:7.1f}) -> ({ex:7.1f},{ey:7.1f})"
    s = f"{t:<8} d={o['distance']:6.1f} a={math.degrees(o['angle']):+5.0f}°"
    if "rel_dir" in o:
        s += f" rel_dir={math.degrees(o['rel_dir']):+5.0f}°"
    if "id" in o:
        s += f" id={o['id']}"
    return s


def obs_world_pos(creature, o):
    """World position of an observed point entity."""
    ang = creature.direction + o["angle"]
    return creature.x + o["distance"] * math.cos(ang), creature.y + o["distance"] * math.sin(ang)


def edge_world(creature, coords):
    """Convert edge coordinates from the creature's local frame back to world coordinates."""
    c, s = math.cos(creature.direction), math.sin(creature.direction)
    return [(creature.x + rx * c - ry * s, creature.y + rx * s + ry * c) for rx, ry in coords]


class Viewer:
    def __init__(self, seed, policy, history_mb=1024):
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.policy = policy
        self.history_budget = history_mb * 1024 * 1024
        pygame.init()
        pygame.key.set_repeat(300, 40)
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 14)
        self.font_big = pygame.font.SysFont("dejavusansmono,monospace", 16, bold=True)
        self.reset()

        info = pygame.display.Info()
        max_h = int(info.current_h * 0.88)
        max_w = int(info.current_w * 0.95) - PANEL_WIDTH
        self.zoom = min(max_h / self.sim.env_height, max_w / self.sim.env_width)
        self.world_w = int(self.sim.env_width * self.zoom)
        self.world_h = int(self.sim.env_height * self.zoom)
        self.screen = pygame.display.set_mode((self.world_w + PANEL_WIDTH, self.world_h))
        pygame.display.set_caption(f"Survival simulator viewer (seed {self.seed})")
        self.world_view = self.screen.subsurface((0, 0, self.world_w, self.world_h))
        self.timeline_rect = pygame.Rect(self.world_w + 10, self.world_h - TIMELINE_H - 26, PANEL_WIDTH - 20, TIMELINE_H)
        self.clock = pygame.time.Clock()

        self.playing = False
        self.speed_idx = SPEEDS.index(10)
        self.accum = 0.0
        self.scroll = 0
        self.dragging = False

    # ------------------------------------------------------------ simulation
    def reset(self):
        self.sim = SimulationCore(seed=self.seed)
        self.shared = shared_env_objects(self.sim.env)
        self.action_rng = random.Random(self.seed)
        self.tick = 0
        self.actions = []
        self.next_actions = {}  # agent_id -> ActionRequest
        self.state = None
        self.predator_ids = {}  # Predator -> label (part of the snapshot so identities survive restores)
        self.game_over = False

        self.history = {}  # tick -> compressed snapshot, in increasing tick order
        self.history_bytes = 0
        self.latest = 0
        self.born_at = {}  # selection key -> first tick it existed
        self.died_at = {}  # selection key -> first tick it no longer existed

        self.selected_key = None  # ("agent", agent_id) or ("predator", label)
        self.selected = None
        self._simulate()  # first step without actions gives the initial observations
        self._record()

    def predator_label(self, p):
        if p not in self.predator_ids:
            self.predator_ids[p] = len(self.predator_ids)
        return self.predator_ids[p]

    def _alive_keys(self):
        keys = {("agent", a.agent_id) for a in self.sim.env.agents}
        keys |= {("predator", self.predator_label(p)) for p in self.sim.env.predators}
        return keys

    def _simulate(self):
        """Run one real simulation tick from the current state."""
        before = self._alive_keys()
        _current_tick[0] = self.tick
        self.state = self.sim.step(self.actions)
        self.tick += 1

        after = self._alive_keys()
        for k in after - before:
            self.born_at.setdefault(k, self.tick)
        for k in before - after:
            self.died_at.setdefault(k, self.tick)

        self.actions = []
        self.next_actions = {}
        for agent, agent_state in zip(self.sim.env.agents, self.state["observations"]):
            action = self.policy(agent_state, self.action_rng)
            self.actions.append((agent.agent_id, action))
            self.next_actions[agent.agent_id] = action

        if self.state["num_agents"] == 0 or self.sim.env.time > MAX_TIME:
            self.game_over = True

    def _record(self):
        buf = io.BytesIO()
        _SnapshotPickler(buf, self.shared).dump((
            self.sim, self.action_rng, self.tick, self.actions, self.next_actions,
            self.state, self.game_over, self.predator_ids,
        ))
        data = zlib.compress(buf.getvalue(), 1)
        self.history[self.tick] = data
        self.history_bytes += len(data)
        self.latest = max(self.latest, self.tick)
        if self.history_bytes > self.history_budget:
            self._thin_history()

    def _thin_history(self):
        target = self.history_budget * 0.75
        for t in list(self.history):
            if self.history_bytes <= target:
                break
            if t % KEYFRAME_EVERY and t != 1 and t < self.latest - KEEP_RECENT:
                self.history_bytes -= len(self.history.pop(t))

    def _restore(self, t):
        (self.sim, self.action_rng, self.tick, self.actions, self.next_actions,
         self.state, self.game_over, self.predator_ids) = _SnapshotUnpickler(
            io.BytesIO(zlib.decompress(self.history[t])), self.shared).load()

    def forward(self):
        if self.tick >= self.latest and self.game_over:
            self.playing = False
            return
        nxt = self.tick + 1
        if nxt in self.history:
            self._restore(nxt)
        elif nxt <= self.latest:
            self._simulate()  # inside a thinned stretch of history
        else:
            self._simulate()
            self._record()
        self.resolve_selection()

    def goto(self, t):
        t = max(1, min(int(t), self.latest))
        if t == self.tick:
            return
        if t in self.history:
            self._restore(t)
        else:
            if not (self.tick < t and t - self.tick < KEYFRAME_EVERY):
                self._restore(max(k for k in self.history if k <= t))
            while self.tick < t:
                self._simulate()
        self.resolve_selection()

    def predator_preview(self, p):
        """Predict the predator's next action from the current state without changing the simulation."""
        env = self.sim.env
        if p.resting:
            return None, None
        clone = copy.copy(p)
        rng = random.Random()
        rng.setstate(env.rng.getstate())  # the predator shares the env rng; never advance the real one
        clone.rng = rng
        obs = clone.observe(agents=env._get_local_agents(p), edges=env._get_local_edges(p))
        return obs, _original_predator_step(clone, obs)

    # ------------------------------------------------------------ selection
    def resolve_selection(self):
        """Find the selected creature in the current state (objects are replaced on restore)."""
        self.selected = None
        if self.selected_key is None:
            return
        kind, ident = self.selected_key
        if kind == "agent":
            self.selected = self.sim.env.agents_dict.get(ident)
        else:
            self.selected = next((p for p in self.sim.env.predators if self.predator_ids.get(p) == ident), None)

    def select(self, key):
        self.selected_key = key
        self.scroll = 0
        self.resolve_selection()

    def select_at(self, sx, sy):
        wx, wy = sx / self.zoom, sy / self.zoom
        best, best_d = None, float("inf")
        for c in list(self.sim.env.agents) + list(self.sim.env.predators):
            d = math.hypot(c.x - wx, c.y - wy)
            if d < max(c.size + 4, 10 / self.zoom) and d < best_d:
                best, best_d = c, d
        if best is None:
            self.select(None)
        elif isinstance(best, Predator):
            self.select(("predator", self.predator_label(best)))
        else:
            self.select(("agent", best.agent_id))

    def select_next_agent(self):
        ids = sorted(self.sim.env.agents_dict)
        if not ids:
            return
        cur = self.selected_key[1] if self.selected_key and self.selected_key[0] == "agent" else -1
        self.select(("agent", next((i for i in ids if i > cur), ids[0])))

    # ------------------------------------------------------------ input
    def timeline_tick(self, x):
        r = self.timeline_rect
        frac = min(max((x - r.x) / r.w, 0.0), 1.0)
        return 1 + round(frac * max(self.latest - 1, 0))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                k = event.key
                shift = event.mod & pygame.KMOD_SHIFT
                if k == pygame.K_q:
                    return False
                elif k == pygame.K_SPACE:
                    self.playing = not self.playing and not (self.game_over and self.tick >= self.latest)
                elif k in (pygame.K_RIGHT, pygame.K_PERIOD):
                    if shift:
                        self.goto_or_simulate(self.tick + 10)
                    elif not self.playing:
                        self.forward()
                elif k in (pygame.K_LEFT, pygame.K_COMMA):
                    self.playing = False
                    self.goto(self.tick - (10 if shift else 1))
                elif k == pygame.K_PAGEUP:
                    self.playing = False
                    self.goto(self.tick - 100)
                elif k == pygame.K_PAGEDOWN:
                    self.goto_or_simulate(self.tick + 100)
                elif k == pygame.K_HOME:
                    self.playing = False
                    self.goto(1)
                elif k == pygame.K_END:
                    self.goto(self.latest)
                elif k == pygame.K_UP:
                    self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
                elif k == pygame.K_DOWN:
                    self.speed_idx = max(self.speed_idx - 1, 0)
                elif k == pygame.K_ESCAPE:
                    self.select(None)
                elif k == pygame.K_TAB:
                    self.select_next_agent()
                elif k == pygame.K_r:
                    self.reset()
                    self.playing = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.timeline_rect.inflate(0, 12).collidepoint(event.pos):
                    self.dragging = True
                    self.playing = False
                    self.goto(self.timeline_tick(event.pos[0]))
                elif event.pos[0] < self.world_w:
                    self.select_at(*event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                self.goto(self.timeline_tick(event.pos[0]))
            elif event.type == pygame.MOUSEWHEEL:
                self.scroll = max(0, self.scroll - event.y * 3)
        return True

    def goto_or_simulate(self, t):
        """Jump forward to tick t, simulating new ticks if t is past the latest one."""
        self.goto(min(t, self.latest))
        while self.tick < t and not (self.game_over and self.tick >= self.latest):
            self.forward()

    # ------------------------------------------------------------ drawing
    def w2s(self, x, y):
        return int(x * self.zoom), int(y * self.zoom)

    def draw_selection_overlay(self, obs, move_dist, move_dir):
        c = self.selected
        surf = self.world_view
        pos = self.w2s(c.x, c.y)

        for o in obs or []:
            color = OBS_COLORS.get(o.get("type"), TEXT)
            if o.get("type") == "Edge":
                a, b = edge_world(c, o["coords"])
                pygame.draw.line(surf, color, self.w2s(*a), self.w2s(*b), 3)
            else:
                target = self.w2s(*obs_world_pos(c, o))
                pygame.draw.line(surf, color, pos, target, 1)
                pygame.draw.circle(surf, color, target, 5, 1)

        # Facing direction
        face = (c.x + 25 * math.cos(c.direction), c.y + 25 * math.sin(c.direction))
        pygame.draw.line(surf, (255, 255, 255), pos, self.w2s(*face), 1)

        # Planned move (move_direction is relative to the facing direction)
        if move_dist is not None and move_dist > 0:
            ang = c.direction + (move_dir or 0.0)
            length = max(move_dist, 1.0) * 3  # exaggerate so it is visible
            end = (c.x + length * math.cos(ang), c.y + length * math.sin(ang))
            pygame.draw.line(surf, SELECT, pos, self.w2s(*end), 3)
            pygame.draw.circle(surf, SELECT, self.w2s(*end), 3)

        pygame.draw.circle(surf, SELECT, pos, int(c.size * self.zoom) + 6, 2)

    def panel_lines(self):
        """Build the info panel as a list of (text, color) lines, plus the overlay data."""
        env = self.sim.env
        c = self.selected
        lines = []
        overlay = (None, None, None)

        if self.selected_key is None:
            lines.append(("Nothing selected", HEAD))
            lines.append(("Click an agent or predator, or press Tab.", DIM))
            lines.append(("", TEXT))
            lines.append((f"Agents ({len(env.agents)}):", HEAD))
            for a in sorted(env.agents, key=lambda a: a.agent_id):
                lines.append((f"  #{a.agent_id:<4} energy {a.energy:6.1f}/{a.max_energy:.0f}  age {a.age:6.1f}", TEXT))
            lines.append(("", TEXT))
            lines.append((f"Predators ({len(env.predators)}):", HEAD))
            for p in env.predators:
                st = "resting" if p.resting else "awake"
                lines.append((f"  P{self.predator_label(p):<4} energy {p.energy:6.1f}  {st}", TEXT))
            return lines, overlay

        if c is None:
            kind, ident = self.selected_key
            name = f"Agent #{ident}" if kind == "agent" else f"Predator P{ident}"
            born, died = self.born_at.get(self.selected_key), self.died_at.get(self.selected_key)
            if born is not None and self.tick < born:
                lines.append((f"{name} does not exist yet (appears at tick {born})", BAD))
            elif died is not None:
                lines.append((f"{name} died at tick {died}", BAD))
            else:
                lines.append((f"{name} is not alive at this tick", BAD))
            return lines, overlay

        if isinstance(c, Predator):
            lines.append((f"Predator P{self.predator_label(c)}", HEAD))
            lines.append((f"  pos ({c.x:.1f}, {c.y:.1f})  facing {fmt_angle(wrap(c.direction))}", TEXT))
            lines.append((f"  energy {c.energy:.1f}/{c.max_energy:.0f}  {'RESTING' if c.resting else 'awake'}", TEXT))
            lines.append((f"  speed {c.speed}  sprint {c.sprint_speed}  hearing {c.hearing_radius}  vision {c.vision_radius}", DIM))
            lines.append(("", TEXT))

            prev_obs, prev_sig = self.predator_preview(c)
            lines.append(("NEXT ACTION (preview, before agents move)", HEAD))
            if prev_sig is None:
                lines.append((f"  resting, wakes above {c.max_energy * 0.5:.0f} energy (+3/tick)", TEXT))
            else:
                lines += self.signal_lines(prev_sig)
                overlay = (prev_obs, prev_sig.get("move"), prev_sig.get("direction"))
            lines.append(("", TEXT))

            acted_last_tick = getattr(c, "_viz_tick", None) == self.tick - 1
            if acted_last_tick:
                lines.append((f"LAST ACTION (tick {self.tick - 1}, actually taken)", HEAD))
                lines += self.signal_lines(c._viz_signals)
            else:
                lines.append(("LAST ACTION: none (resting last tick)", HEAD))
            lines.append(("", TEXT))

            if prev_obs is not None:
                lines += self.obs_lines(prev_obs, "OBSERVATIONS (preview)")
            elif acted_last_tick:
                lines += self.obs_lines(c._viz_obs, "OBSERVATIONS (last tick)")
            return lines, overlay

        # Agent
        st = next((s for s in self.state["observations"] if s and s["agent_id"] == c.agent_id), None)
        act = self.next_actions.get(c.agent_id)
        lines.append((f"Agent #{c.agent_id}", HEAD))
        lines.append((f"  pos ({c.x:.1f}, {c.y:.1f})  facing {fmt_angle(wrap(c.direction))}", TEXT))
        if st is not None:
            for k, v in st.items():
                if k in ("observations", "agent_id"):
                    continue
                val = f"{v:.3f}" if isinstance(v, float) else str(v)
                lines.append((f"  {k:<15} {val}", TEXT))
        lines.append((f"  {'max_age':<15} {c.max_age:.1f}", DIM))
        lines.append(("", TEXT))

        lines.append(("NEXT ACTION", HEAD))
        move_dist = move_dir = None
        if act is None:
            lines.append(("  (no action)", DIM))
        else:
            sprint = act.move_distance > c.speed
            lines.append((f"  move_distance  {act.move_distance:.3f}{'  (sprint)' if sprint else ''}", TEXT))
            lines.append((f"  move_direction {fmt_angle(act.move_direction)}", TEXT))
            lines.append((f"  turn_angle     {fmt_angle(act.turn_angle)}", TEXT))
            lines.append((f"  spawn_agent    {act.spawn_agent}{'' if c.energy > 100 else '  (energy <= 100, ignored)'}", TEXT))
            move_dist, move_dir = act.move_distance, act.move_direction
        lines.append(("", TEXT))

        obs = st["observations"] if st else []
        lines += self.obs_lines(obs, "OBSERVATIONS")
        return lines, (obs, move_dist, move_dir)

    def signal_lines(self, sig):
        out = []
        if not sig:
            return [("  (no signals)", DIM)]
        if "move" in sig:
            d = sig["direction"]
            out.append((f"  move      {sig['move']:.3f}", TEXT))
            out.append((f"  direction {'forward (None)' if d is None else fmt_angle(float(d))}", TEXT))
        if "turn" in sig:
            out.append((f"  turn      {fmt_angle(float(sig['turn']))}", TEXT))
        return out

    def obs_lines(self, obs, title):
        obs = obs or []
        counts = {}
        for o in obs:
            counts[o["type"]] = counts.get(o["type"], 0) + 1
        summary = ", ".join(f"{k} {v}" for k, v in counts.items()) or "nothing"
        out = [(f"{title}: {summary}", HEAD)]
        order = {"Predator": 0, "Agent": 1, "Fruit": 2, "Tree": 3, "Edge": 4}
        for o in sorted(obs, key=lambda o: (order.get(o["type"], 9), o.get("distance", 0))):
            out.append(("  " + fmt_obs(o), OBS_COLORS.get(o["type"], TEXT)))
        return out

    def draw_timeline(self):
        r = self.timeline_rect
        pygame.draw.rect(self.screen, (50, 54, 62), r, border_radius=4)
        if self.latest > 1:
            x = r.x + round((self.tick - 1) / (self.latest - 1) * r.w)
            pygame.draw.rect(self.screen, (80, 110, 150), (r.x, r.y, x - r.x, r.h), border_radius=4)
            pygame.draw.line(self.screen, SELECT, (x, r.y - 4), (x, r.bottom + 3), 3)
        label = f"tick {self.tick} / {self.latest}   history {self.history_bytes / 2**20:.0f} MB"
        if self.tick < self.latest:
            label += "   (viewing the past)"
        self.screen.blit(self.font.render(label, True, DIM), (r.x, r.bottom + 6))

    def draw_panel(self, lines):
        x0 = self.world_w
        pygame.draw.rect(self.screen, BG, (x0, 0, PANEL_WIDTH, self.world_h))
        env = self.sim.env
        speed = SPEEDS[self.speed_idx]
        status = "GAME OVER" if self.game_over else ("PLAYING" if self.playing else "PAUSED")
        header = [
            (f"{status}   tick {self.tick}   t={env.time:.1f}s", HEAD),
            (f"score {self.state['score']:.2f}   agents {len(env.agents)}   predators {len(env.predators)}", TEXT),
            (f"speed {speed} ticks/s ({speed / 10:g}x real time)   seed {self.seed}", TEXT),
            ("Space play  ←/→ step  PgUp/PgDn ±100  ↑↓ speed", DIM),
            ("Tab next agent  Esc clear  R restart  drag timeline", DIM),
        ]
        y = 8
        for text, color in header:
            self.screen.blit((self.font_big if color is HEAD else self.font).render(text, True, color), (x0 + 10, y))
            y += 20
        pygame.draw.line(self.screen, DIM, (x0 + 8, y + 2), (x0 + PANEL_WIDTH - 8, y + 2))
        y += 10

        line_h = 17
        bottom = self.timeline_rect.y - 10
        visible = (bottom - y) // line_h
        self.scroll = min(self.scroll, max(0, len(lines) - visible))
        for text, color in lines[self.scroll:self.scroll + visible]:
            self.screen.blit(self.font.render(text, True, color), (x0 + 10, y))
            y += line_h
        if len(lines) > visible:
            self.screen.blit(self.font.render(f"[{self.scroll + 1}-{min(len(lines), self.scroll + visible)} of {len(lines)}]", True, DIM),
                             (x0 + PANEL_WIDTH - 150, 8))
        self.draw_timeline()

    def draw(self):
        self.sim.env.draw(self.world_view)
        lines, (obs, move_dist, move_dir) = self.panel_lines()
        if self.selected is not None:
            self.draw_selection_overlay(obs, move_dist, move_dir)
        self.draw_panel(lines)
        pygame.display.flip()

    # ------------------------------------------------------------ main loop
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            running = self.handle_events()
            if self.playing:
                self.accum += SPEEDS[self.speed_idx] * dt
                n = int(self.accum)
                self.accum -= n
                for _ in range(min(n, MAX_TICKS_PER_FRAME)):
                    self.forward()
                    if not self.playing:
                        break
                if n > MAX_TICKS_PER_FRAME:
                    self.accum = 0.0  # the simulation can't keep up; drop the backlog
            else:
                self.accum = 0.0
            self.draw()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="simulation seed (random if omitted)")
    parser.add_argument("--policy", default="src.utils.controllers.dummy_agent_policy:action_decision",
                        help="agent policy as module.path:function")
    parser.add_argument("--history-mb", type=int, default=1024,
                        help="memory for saved snapshots before older ticks are thinned")
    args = parser.parse_args()
    Viewer(args.seed, load_policy(args.policy), args.history_mb).run()


if __name__ == "__main__":
    main()

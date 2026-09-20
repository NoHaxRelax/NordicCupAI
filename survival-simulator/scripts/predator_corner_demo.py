"""Watch native predators try to leave a bare arena corner.

From the repository root:
    python survival-simulator/scripts/predator_corner_demo.py
    python survival-simulator/scripts/predator_corner_demo.py --start touching
    python survival-simulator/scripts/predator_corner_demo.py --start touching --heading diagonal
    python survival-simulator/scripts/predator_corner_demo.py --start touching --heading diagonal --gap 5
    python survival-simulator/scripts/predator_corner_demo.py --headless --output corner.json

Only the scene is arranged: uniform terrain, no prey, and no ambient spawns.
Native sensing, decisions, collisions, energy, and sleep cycles are unchanged.
An escape means the center first crosses the marked radius from the inside
corner. This is a finite experiment, not a guarantee for every corner or chase.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame

from src.elements.biome import (
    Desert_biome, Grassland_biome, Map_generator, River_biome, Swamp_biome,
)
from src.elements.environment import Environment
from src.elements.predator import Predator

DT = 0.1
WIDTH, HEIGHT, PANEL = 800, 600, 320
WALL = 30
ESCAPE_RADIUS = 180
BIOMES = {"grassland": Grassland_biome, "desert": Desert_biome,
          "swamp": Swamp_biome, "river": River_biome}
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
BG, TEXT, MUTED = (17, 24, 33), (234, 241, 248), (161, 179, 195)
RED, GREEN, AMBER = (255, 108, 116), (94, 222, 167), (255, 202, 112)


class CornerEnvironment(Environment):
    """Use a controlled scene while retaining Environment.non_agent_step."""

    def _render_biome_surface(self):
        self.biome_surface.fill((31, 49, 48))

    def spawn_tree(self, *args, **kwargs):
        return None

    def spawn_fruit(self, *args, **kwargs):
        return None

    def spawn_predator(self, *args, **kwargs):
        return None  # The fixed test population is installed below.


class Trial:
    def __init__(self, args):
        self.args = args
        self.heading = getattr(args, "heading", "varied")
        self.gap = getattr(args, "gap", 0.0)
        biome = BIOMES[args.biome]()

        def flat_map(generator):
            result = np.empty((generator.width, generator.height), dtype=object)
            result.fill(biome)
            return result

        # Patch only map generation during construction, never movement code.
        with patch.object(Map_generator, "generate", flat_map):
            self.env = CornerEnvironment(WIDTH, HEIGHT, 400, random.Random(args.seed))
        right, bottom = "right" in args.corner, "bottom" in args.corner
        self.corner = (WIDTH - WALL if right else WALL,
                       HEIGHT - WALL if bottom else WALL)
        self.sign = (-1 if right else 1, -1 if bottom else 1)
        setup_rng = random.Random(args.seed)
        side = math.ceil(math.sqrt(args.count))
        self.starts = []
        for i in range(args.count):
            if args.start == "touching":
                dx = dy = 10.0 + self.gap  # Radius plus clearance from each wall.
                heading = math.pi + (i + 0.5) / args.count * math.pi / 2
            else:
                dx = 10 + self.gap + 50 * (i % side) / max(1, side - 1)
                dy = 10 + self.gap + 50 * (i // side) / max(1, side - 1)
                heading = math.atan2(-dy, -dx) + setup_rng.uniform(-0.25, 0.25)
            x = self.corner[0] + self.sign[0] * dx
            y = self.corner[1] + self.sign[1] * dy
            predator = Predator(x, y, rng=self.env.rng)
            predator.direction = math.atan2(self.sign[1] * math.sin(heading),
                                             self.sign[0] * math.cos(heading))
            if self.heading == "diagonal":
                # Exact inward bisector: no heading spread or random jitter.
                predator.direction = math.atan2(-self.sign[1], -self.sign[0])
            if self.env._in_obstacle((x, y), predator.size, self.env.obstacles):
                raise RuntimeError(f"Predator {i + 1} starts inside a wall")
            self.env.predators.append(predator)
            self.starts.append([x, y, predator.direction])
        self.env._update_predator_grid()
        self.trails = [deque([(p.x, p.y)], maxlen=450) for p in self.env.predators]
        self.escape_times = [None] * args.count
        self.ticks = 0
        self.limit = math.ceil(args.seconds / DT)

    @property
    def seconds(self):
        return self.ticks * DT

    @property
    def finished(self):
        return self.ticks >= self.limit

    def outside(self, predator):
        return math.hypot(predator.x - self.corner[0],
                          predator.y - self.corner[1]) >= ESCAPE_RADIUS

    def step(self):
        if self.finished:
            return
        self.env.non_agent_step(DT)
        self.ticks += 1
        for i, predator in enumerate(self.env.predators):
            self.trails[i].append((predator.x, predator.y))
            if self.escape_times[i] is None and self.outside(predator):
                self.escape_times[i] = round(self.seconds, 6)

    def summary(self):
        times = [t for t in self.escape_times if t is not None]
        return {
            "seed": self.args.seed, "corner": self.args.corner,
            "start": self.args.start, "biome": self.args.biome,
            "heading": self.heading,
            "gap": self.gap,
            "count": len(self.env.predators), "seconds": round(self.seconds, 6),
            "dt": DT, "finished": self.finished,
            "escape_rule": f"Center first reaches {ESCAPE_RADIUS} units from inside corner",
            "escaped_ever": len(times),
            "outside_now": sum(self.outside(p) for p in self.env.predators),
            "first_escape_seconds": min(times) if times else None,
            "last_escape_seconds": max(times) if times else None,
            "assumptions": ["No prey or ambient spawns; uniform terrain",
                            "Native predator sensing, movement, collision, energy and rest",
                            "Native initial energy 0; predators start resting",
                            "Predators do not collide with one another in the engine",
                            "Leaving this region once does not imply permanent absence"],
            "predators": [
                {"id": i + 1, "start": self.starts[i],
                 "escape_seconds": self.escape_times[i], "end": [p.x, p.y],
                 "energy": p.energy, "resting": p.resting}
                for i, p in enumerate(self.env.predators)
            ],
        }


class Renderer:
    def __init__(self):
        pygame.font.init()
        self.canvas = pygame.Surface((WIDTH + PANEL, HEIGHT))
        self.font = pygame.font.SysFont("segoeui", 17)
        self.small = pygame.font.SysFont("segoeui", 15)
        self.title = pygame.font.SysFont("segoeui", 25, bold=True)
        self.big = pygame.font.SysFont("segoeui", 40, bold=True)

    def draw(self, surface, trial, paused=False, speed=1.0, trails=True):
        # Windows/SDL may return a smaller display surface than requested.
        # Render at a fixed size, then fit the complete scene and sidebar to it.
        self._draw_frame(self.canvas, trial, paused, speed, trails)
        width, height = surface.get_size()
        if width <= 0 or height <= 0:
            return
        scale = min(width / self.canvas.get_width(), height / self.canvas.get_height())
        size = (max(1, round(self.canvas.get_width() * scale)),
                max(1, round(self.canvas.get_height() * scale)))
        frame = (self.canvas if size == self.canvas.get_size()
                 else pygame.transform.smoothscale(self.canvas, size))
        surface.fill(BG)
        surface.blit(frame, ((width - size[0]) // 2, (height - size[1]) // 2))

    def _draw_frame(self, surface, trial, paused, speed, trails):
        surface.fill(BG)
        world = surface.subsurface((0, 0, WIDTH, HEIGHT))
        world.blit(trial.env.static_surface, (0, 0))
        cx, cy = trial.corner
        sx, sy = trial.sign
        arc = [(cx + sx * ESCAPE_RADIUS * math.cos(a),
                cy + sy * ESCAPE_RADIUS * math.sin(a))
               for a in np.linspace(0, math.pi / 2, 90)]
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(shade, (255, 192, 95, 22), [(cx, cy)] + arc)
        world.blit(shade, (0, 0))
        pygame.draw.lines(world, AMBER, False, arc, 2)
        if trails:
            for i, path in enumerate(trial.trails):
                if len(path) > 1:
                    color = (52, 107, 88) if trial.escape_times[i] is not None else (122, 68, 70)
                    pygame.draw.lines(world, color, False, list(path), 1)
        for obstacle in trial.env.obstacles:
            pygame.draw.rect(world, (83, 96, 109),
                             (obstacle.x, obstacle.y, obstacle.width, obstacle.height))
        for i, p in enumerate(trial.env.predators):
            color = GREEN if trial.escape_times[i] is not None else RED
            center = (round(p.x), round(p.y))
            pygame.draw.circle(world, BG, center, round(p.size) + 1)
            pygame.draw.circle(world, color, center, round(p.size), 2)
            tip = (round(p.x + math.cos(p.direction) * p.size),
                   round(p.y + math.sin(p.direction) * p.size))
            pygame.draw.line(world, color, center, tip, 2)
            if p.resting:
                pygame.draw.circle(world, AMBER, center, 3)

        x, y = WIDTH + 22, 20

        def label(text, color=TEXT, font=None, gap=26):
            nonlocal y
            rendered = (font or self.font).render(text, True, color)
            surface.blit(rendered, (x, y))
            y += gap

        escaped = sum(t is not None for t in trial.escape_times)
        label("CORNER ESCAPE", font=self.title, gap=39)
        label(f"{escaped} / {len(trial.escape_times)}", GREEN, self.big, 53)
        label("have crossed the amber arc", MUTED)
        label(f"Time  {trial.seconds:.1f} / {trial.args.seconds:g} s")
        state = "FINISHED" if trial.finished else "PAUSED" if paused else "RUNNING"
        label(f"{state}    {speed:g}x playback", AMBER)
        label(f"Outside now: {sum(trial.outside(p) for p in trial.env.predators)}", gap=32)
        times = [t for t in trial.escape_times if t is not None]
        label(f"First departure: {min(times):.1f} s" if times else "First departure: waiting")
        label(f"Last departure: {max(times):.1f} s" if times else "Last departure: waiting", gap=32)
        label(f"{trial.args.corner} | {trial.args.biome}", MUTED)
        start_label = "near corner" if trial.args.start == "touching" and trial.gap else trial.args.start
        label(f"Start: {start_label} | Seed: {trial.args.seed}", MUTED)
        label(f"Heading: {trial.heading} | Gap: {trial.gap:g}", MUTED, self.small, 25)
        label("Red: has not left yet", RED, self.small, 22)
        label("Green: has left at least once", GREEN, self.small, 22)
        label("Amber dot: resting", AMBER, self.small, 30)
        label("Arc = 180 units from corner.", MUTED, self.small, 22)
        label("No prey. Normal energy and rest.", MUTED, self.small, 22)
        label("Predators can overlap in this engine.", MUTED, self.small, 33)
        label("SPACE pause    RIGHT single step", font=self.small, gap=23)
        label("+ / - speed    T trails    R replay", font=self.small, gap=23)
        label("N next seed    ESC quit", font=self.small)


def save_results(args, trial, renderer, surface):
    summary = trial.summary()
    print(f"{summary['escaped_ever']}/{summary['count']} left the corner at least once "
          f"in {summary['seconds']:g}s; {summary['outside_now']} outside now.")
    if summary["first_escape_seconds"] is not None:
        print(f"Departure times: {summary['first_escape_seconds']:g} to "
              f"{summary['last_escape_seconds']:g}s.")
    for path in (args.output, args.screenshot):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Results: {args.output}")
    if args.screenshot:
        screenshot = pygame.Surface((WIDTH + PANEL, HEIGHT))
        renderer.draw(screenshot, trial)
        pygame.image.save(screenshot, str(args.screenshot))
        print(f"Screenshot: {args.screenshot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--corner", choices=CORNERS, default="top-left")
    parser.add_argument("--start", choices=("spread", "touching"), default="spread")
    parser.add_argument("--gap", type=float, default=0,
                        help="Extra starting clearance from each wall, 0 to 60 units")
    parser.add_argument("--heading", choices=("varied", "diagonal"), default="varied",
                        help="Use varied headings or the exact 45-degree inward corner diagonal")
    parser.add_argument("--biome", choices=BIOMES, default="grassland")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--headless", action="store_true", help="Run to completion without opening a window")
    parser.add_argument("--output", type=Path, help="Write per-predator escape times as JSON on exit")
    parser.add_argument("--screenshot", type=Path, help="Save the final rendered frame as a PNG on exit")
    args = parser.parse_args()
    if not 1 <= args.count <= 500:
        parser.error("--count must be between 1 and 500")
    if not math.isfinite(args.gap) or not 0 <= args.gap <= 60:
        parser.error("--gap must be between 0 and 60, keeping all starting positions inside the escape arc")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be a positive finite number")
    if not math.isfinite(args.speed) or not 0.125 <= args.speed <= 16:
        parser.error("--speed must be between 0.125 and 16")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.display.init()
    surface = (pygame.Surface((WIDTH + PANEL, HEIGHT)) if args.headless
               else pygame.display.set_mode((WIDTH + PANEL, HEIGHT), pygame.RESIZABLE))
    pygame.display.set_caption("Predator corner escape - native simulation")
    renderer = Renderer()
    trial = Trial(args)
    print(f"Testing {args.count} predators, {args.corner}, {args.start}, "
          f"heading {args.heading}, gap {args.gap:g}, {args.biome}, seed {args.seed}.")
    print("Predators start resting with native energy 0; their first movement is around 3.5s.")
    try:
        if args.headless:
            while not trial.finished:
                trial.step()
        else:
            clock = pygame.time.Clock()
            paused, running, trails = False, True, True
            accumulator, speed = 0.0, args.speed
            while running:
                elapsed = min(clock.tick(60) / 1000, 0.25)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            accumulator = 0
                        elif event.key == pygame.K_RIGHT:
                            paused = True
                            accumulator = 0
                            trial.step()
                        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                            speed = min(16, speed * 2)
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            speed = max(0.125, speed / 2)
                        elif event.key == pygame.K_t:
                            trails = not trails
                        elif event.key in (pygame.K_r, pygame.K_n):
                            if event.key == pygame.K_n:
                                args.seed += 1
                            trial = Trial(args)
                            accumulator = 0

                if not paused and not trial.finished:
                    accumulator += elapsed * speed
                    while accumulator >= DT and not trial.finished:
                        trial.step()
                        accumulator -= DT
                surface = pygame.display.get_surface()
                renderer.draw(surface, trial, paused, speed, trails)
                pygame.display.flip()
        save_results(args, trial, renderer, surface)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

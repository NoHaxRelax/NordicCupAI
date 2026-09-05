"""Headless, fast-forwarded driver for the DM i AI 2025 race-car simulator.

This is a *rehearsal* artifact. Its value is not race-car -- that competition
is over -- but the four-step recipe it demonstrates, which is what we will
run against the 2026 simulator on day one:

  1. Find the seam. The organisers' `game_loop` mixes simulation, rendering,
     input and termination. `update_game(action)` is the pure-simulation
     part; drive that directly.
  2. Kill the realtime clock. `game_loop` calls `clock.tick(60)`, so a
     60-second episode costs 60 wall-clock seconds. Bypassing it is worth
     ~36x, which is the difference between RL being possible in a four-day
     event and not.
  3. Port the termination check. Collision detection lives in the render
     loop, NOT in `update_game`. Drive `update_game` alone and the ego car
     is immortal -- every reward signal is silently wrong.
  4. Force a headless video driver. SDL_VIDEODRIVER=dummy plus a throwaway
     display mode; sprites need a video surface to load even when nothing
     is drawn.
  5. Fix the working directory. Assets load through paths relative to cwd,
     so the sim only initialises from inside its own directory.

Step 3 is the one that bites quietly: it does not raise, it just makes the
ego car immortal and every reward signal wrong. Budget an hour for it on
day one.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Must be set before pygame initialises its video backend.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ACTIONS = ("ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT", "NOTHING")

def _default_sim_path() -> Path:
    """Locate the 2025 race-car sim. Override with RACECAR_SIM_PATH."""
    override = os.environ.get("RACECAR_SIM_PATH")
    if override:
        return Path(override)
    candidates = [
        Path.home() / "bforbanks" / "dm-i-ai-2025" / "race-car",
        Path("/home/user/bforbanks/dm-i-ai-2025/race-car"),
        Path.cwd().parents[1] / "bforbanks" / "dm-i-ai-2025" / "race-car",
        Path.cwd() / "DM-i-AI-2025" / "race-car",
    ]
    for candidate in candidates:
        if (candidate / "src" / "game" / "core.py").exists():
            return candidate
    return candidates[0]


DEFAULT_SIM_PATH = _default_sim_path()


@dataclass
class StepResult:
    observation: dict
    reward: float
    terminated: bool
    info: dict


class RaceCarEnv:
    """Minimal Gym-style wrapper. No gymnasium dependency on purpose -- the
    point is to show the seam, not to marry an RL library before we know
    what the 2026 task even is."""

    max_ticks = 60 * 60  # organisers' MAX_TICKS: 60 seconds at 60 fps

    def __init__(self, sim_path: Path | str = DEFAULT_SIM_PATH, seed: int = 0) -> None:
        self.sim_path = Path(sim_path)
        if not (self.sim_path / "src" / "game" / "core.py").exists():
            raise FileNotFoundError(
                f"race-car simulator not found at {self.sim_path}. Clone "
                "https://github.com/amboltio/DM-i-AI-2025 and point sim_path at its race-car/ dir."
            )
        if str(self.sim_path) not in sys.path:
            sys.path.insert(0, str(self.sim_path))

        import pygame

        self.pygame = pygame
        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((64, 64))

        import src.game.core as core

        self.core = core
        self.seed = seed
        self.reset(seed)

    @contextlib.contextmanager
    def _in_sim_dir(self):
        """The sim loads sprites by relative path (step 5)."""
        previous = Path.cwd()
        os.chdir(self.sim_path)
        try:
            yield
        finally:
            os.chdir(previous)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed
        with self._in_sim_dir():
            self.core.initialize_game_state("http://localhost/predict", self.seed)
        self.state = self.core.STATE
        self._last_distance = 0.0
        return self.observe()

    def observe(self) -> dict:
        """Mirror the shape the competition server actually sends."""
        state = self.state
        return {
            "did_crash": state.crashed,
            "elapsed_time_ms": state.elapsed_game_time,
            "distance": int(state.distance),
            "velocity": {"x": int(state.ego.velocity.x), "y": int(state.ego.velocity.y)},
            "coordinates": {"x": int(state.ego.x), "y": int(state.ego.y)},
            "sensors": {s.name: s.reading for s in state.sensors},
        }

    def step(self, action: str) -> StepResult:
        core, state = self.core, self.state
        state.ticks += 1
        state.elapsed_game_time += 1000 // 60

        core.update_game(action)

        # --- step 3: termination, lifted out of the organisers' render loop
        for car in state.cars:
            if car is not state.ego and core.intersects(state.ego.rect, car.rect):
                state.crashed = True
        for wall in state.road.walls:
            if core.intersects(state.ego.rect, wall.rect):
                state.crashed = True

        reward = state.distance - self._last_distance
        self._last_distance = state.distance
        terminated = bool(state.crashed) or state.ticks > self.max_ticks
        return StepResult(
            observation=self.observe(),
            reward=float(reward),
            terminated=terminated,
            info={"ticks": state.ticks, "distance": state.distance, "crashed": state.crashed},
        )

    def run_episode(self, policy, max_ticks: int | None = None) -> dict:
        """`policy` maps observation -> action string."""
        self.reset(self.seed)
        limit = max_ticks or self.max_ticks
        for _ in range(limit):
            result = self.step(policy(self.observe()))
            if result.terminated:
                break
        return {
            "distance": round(self.state.distance, 1),
            "ticks": self.state.ticks,
            "crashed": bool(self.state.crashed),
        }

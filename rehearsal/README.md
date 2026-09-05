# rehearsal

Day-one readiness, practised against the DM i AI 2025 race-car simulator --
real organiser code, not our own fiction.

The point is not race-car. It is that the 2026 simulator will arrive with the
same shape of problems, and we will have solved them once already.

## The recipe

The organisers ship a runnable local simulator (confirmed: `race-car/src/`
contains the full game, collision math and assets). It is built to be *played*
at 60 fps, not to be *trained against*. Converting one into a training
environment took five steps:

1. **Find the seam.** `game_loop` mixes simulation, rendering, input and
   termination. `update_game(action)` is the pure-simulation core.
2. **Kill the realtime clock.** `game_loop` calls `clock.tick(60)`, so a
   60-second episode costs 60 wall-clock seconds. Worth ~28x.
3. **Port the termination check.** Collision detection lives in the render
   loop, not in `update_game`. This one is silent -- see below.
4. **Force a headless video driver.** `SDL_VIDEODRIVER=dummy` plus a
   throwaway `set_mode`; sprites need a video surface even when nothing draws.
5. **Fix the working directory.** Assets load by relative path, so the sim
   only initialises from inside its own directory.

## Why step 3 is the dangerous one

Driving `update_game` alone, the ego car never crashes:

| | distance | crashed |
| --- | --- | --- |
| before porting the collision check | 42,911 | never |
| after | 2,844 | tick 302 |

No exception, no warning -- just a reward signal inflated 15x. A day of
training against that produces a policy that looks superb locally and dies
immediately on the competition server.

**Day-one rule: before trusting any local environment, verify that episodes
actually terminate.** A policy that never dies is a bug, not a breakthrough.

## Measured throughput

    headless           1,671 ticks/sec/core
    organisers' loop      60 ticks/sec
    speedup                28x

10M environment steps:

    4 cores          0.42 h
    32 cores         0.05 h
    realtime loop      46 h    <- longer than the entire competition

That is the whole argument for doing step 2 before anything else. RL is
viable in a four-day event only on the near side of that gap.

## Usage

    pip install pygame
    export RACECAR_SIM_PATH=/path/to/DM-i-AI-2025/race-car
    python bench.py

`racecar_env.py` exposes a minimal Gym-style API (`reset`, `step`,
`run_episode`). It deliberately does not depend on gymnasium -- committing to
an RL library before seeing the 2026 task would be premature.

## What carries to 2026

Nothing in `racecar_env.py` survives contact except the recipe. Expect to
spend the first hour of the competition doing steps 1-5 against whatever
simulator ships, and expect step 3 to be lurking in it.

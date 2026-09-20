# Predator corner escape demo

Run from the repository root with the simulator's Python environment:

```powershell
python survival-simulator/scripts/predator_corner_demo.py
```

If your active Python reports `No module named 'pygame'`, use the simulator's
existing virtual environment. For this checkout, the verified environment is
in the sibling `NordicCupAI` repository. From the repository root in PowerShell:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" "survival-simulator/scripts/predator_corner_demo.py" --start touching
```

The window shows 64 predators near the top-left arena corner, initially facing
into it. They start with native zero energy and rest before moving at about
3.5 simulated seconds. The simulation stops at 60 seconds and leaves the window
open for inspection.

The window is resizable. The complete arena and sidebar scale to fit the actual
window size, including smaller windows on displays with Windows scaling.
Saved screenshots retain the full 1120-by-600 layout.

To start all 64 predators in exact contact with both walls:

```powershell
python survival-simulator/scripts/predator_corner_demo.py --start touching
```

Predators may overlap because the native engine does not make them collide with
each other. In `touching` mode they share the same initial center but have
different headings into the corner. The default `spread` mode varies both their
positions and headings.

For the exact 45-degree inward diagonal, with every predator touching both walls:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" "survival-simulator/scripts/predator_corner_demo.py" --start touching --heading diagonal
```

There is no heading jitter in this mode. The initial centers and headings are
identical, so all 64 predators can look like a single predator while overlapping.

Add `--gap 5` to start 5 units clear of **each wall**, keeping the diagonal
heading. With `--start touching`, this offsets the shared starting position;
the predators are only actually touching when the gap is zero.

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" "survival-simulator/scripts/predator_corner_demo.py" --start touching --heading diagonal --gap 5
```

The gap can range from 0 to 60 units, keeping every starting position inside
the escape arc. In the spread layout, it moves the entire group away from both
walls. The displayed gap is clearance beyond the predators' radius, not the
distance from their centers to the walls.

An escape is counted when a predator's center first reaches 180 units from the
inside corner, crossing the amber arc. Red means it has not crossed; green means
it has crossed at least once. An amber center dot means resting. `Outside now`
counts the current positions separately: predators can wander back into the
corner after escaping.

Controls: **Space** pauses, **Right arrow** advances one tick, **+ / -** changes
playback speed, **T** toggles trails, **R** replays the same seed, **N** starts the
next seed, and **Escape** exits. Playback speed does not change the simulation's
fixed 0.1-second timestep.

Other options include `--count`, `--seed`, `--seconds`, `--corner` (`top-left`,
`top-right`, `bottom-left`, `bottom-right`), and `--biome` (`grassland`, `desert`,
`swamp`, `river`). For a run without a window, with results and a rendered image:

```powershell
python survival-simulator/scripts/predator_corner_demo.py --headless --start touching --seconds 20 --output corner-results.json --screenshot corner.png
```

## What was verified

Using seed 1, 64 predators, and the `touching` start, all 64 escaped in every
combination of the four arena corners and four terrain types. Every predator's
position remained outside obstacles on every tick of the 20-second tests.

| Terrain | First-to-last departure across all four corners |
| --- | --- |
| Grassland | 5.0–5.1 seconds |
| Desert | 5.4–5.5 seconds |
| Swamp | 6.5–6.7 seconds |
| River | 8.5–8.9 seconds |

The default spread start also escaped 64/64 on top-left grassland, at 4.8–5.7
seconds. The window rendering and pause, step, replay, next-seed, speed, trails,
and quit controls were smoke-tested using SDL's dummy display.

With `--start touching --heading diagonal`, seed 1, and grassland, all 64
predators departed at 5.0 seconds in each of the four corners. At initial contact,
the wall avoidance rule selects one closest wall and commands a 90-degree turn.
It does not add two opposing wall avoidance turns that could cancel. Native
collision adjustment then finds a legal movement direction.

The diagonal was also tested with one predator in each corner, on grassland,
seed 1, for gaps of 0.001, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30,
50, 75, and 100 units. All 64 single-predator scenarios escaped within 6.4
seconds, including the initial rest. Selected results across the four corners:

| Gap from each wall | Departure time |
| --- | --- |
| 0.001–2 units (sampled values above) | 5.0 seconds |
| 5 units | 5.2 seconds |
| 10 units | 5.1 seconds |
| 30 units | 5.5 seconds |
| 50 units | 5.6–5.7 seconds |

The 75- and 100-unit probes used direct scene setup; the viewer's 60-unit limit
also keeps its spread layout inside the arc.

The smaller-window crash was reproduced at 640-by-360 and fixed by fitting an
offscreen canvas to the display. Regression tests cover small windows, different
aspect ratios, resizing, and zero-size surfaces. The interactive loop was also
checked with SDL returning 640-by-360 for the requested 1120-by-600 window;
the scene and sidebar remained visible and PNG export retained full resolution.

This uses the real `Predator`, `Creature.observe`, and
`Environment.non_agent_step`, including native collision adjustment, energy
costs, and rest cycles. Only scene construction changes: uniform terrain, no
prey, and disabled ambient spawns. The results demonstrate escape from bare
90-degree arena corners. They do not establish escape from narrow obstacle
pockets or while chasing inaccessible prey, nor do they guarantee that an
escaped predator will stay away.

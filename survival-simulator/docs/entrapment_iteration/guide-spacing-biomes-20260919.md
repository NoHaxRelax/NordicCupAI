# Guide spacing and biome transitions, 2026-09-19

The guide now prefers a configurable 100–120-unit following distance. This is
a soft objective: it may get closer for a safe pass or delivery. The forecast
uses native bounded turns and watched sideways movement instead of instant
homing. Only the native 15-unit endpoint contact radius is treated as capture;
the old fixed 48-unit predator clearance applies only with lookahead disabled.

The native engine applies terrain at the **start** of each movement. Sprint
speed 20 produces only 6 units in river while a predator on forest can move 15.
Sprint availability alone therefore does not prove that an escape exists.
Premature capture while sprint is available is nevertheless a failure metric
and a useful case to inspect. Deliberate `hold_at_delivery` sacrifices are
separate. Availability is measured at the start of the fatal native tick.

The policy receives current biome labels and shared visited biome samples,
never the engine biome map. Nominal forecasts interpolate very locally from
these samples; final candidates also replay the same movement commands with
river slowdown starting at either of the next two ticks. Predator speed cap
and terrain are sampled independently of guide terrain. Unknown exact borders,
rest state, targets, wandering, wall deflection, observation lag and motion
beyond three ticks still limit the forecast. Sampled safety is not a guarantee.

CLI parameters: `--guide-distance-min 100 --guide-distance-max 120`.
The full-game recorder saves premature sprint-available guide death details;
the replay lists these as `SPRINT FAILURE` and jumps one second before capture.
Every frame of the new native run is in
`logs/entrapment-iteration/guide-spacing-biomes-v2-20260919`.

## Completed v2 native replay

Seed 1883894846, 3000-second maximum, normal extinction at **750.4 seconds**.
Score **784.88**, runtime **255.81 seconds**, all **7,504 frames** retained.
Replay: `http://localhost:9078/`. Source change: `c5c7f50` (the relief helper
saved separately is inactive). Exact manifest and compact summary are adjacent.

- **0** premature guide predator deaths with sprint available at the fatal
  tick, versus **6** in the v1 replay on the same seed. The six v1 cases were
  at 263.4, 440.7, 478.8, 481.0 and 497.1 seconds (two at 497.1).
- **2** deliberate delivery sacrifices; **7** other guide captures happened
  below the sprint threshold. Another **7** guides died from energy loss.
- **3** non-guide agents died to predators while sprint was available.
  The broader all-agent benchmark is therefore **not passed**.
- Only **2 of 19 guide assignments reached delivery**. This counts arrival,
  not confirmed predator capture. Four predators were simultaneously within
  40 units of bait for at least 30 seconds; this is only a retention proxy.
- **81.7 seconds** without bait after first arrival; bait continuity is still
  unsolved. Score improved over v1's 592.23 but remains below the earlier
  872.19 default. One seed cannot establish reliability.

The preferred gap is not an enforced band: among 388 en-route, sprint-capable
guide ticks with a reported distance and handoff over 80 units away, median
observed distance was 124.755, and 28.35% lay within 100–120. Recovery, delivery,
terrain and route priorities can take it outside the target. Do not report
these results as reliably maintaining the band.

## Earlier comparison results

The preceding v1 ablations on seed 1883894846 all ended in extinction:

| v1 options | Score | Lifetime (s) |
| --- | ---: | ---: |
| Search + shared corridors + earlier food dispatch | 592.23 | 579.9 |
| No search; shared corridors + earlier food dispatch | 472.33 | 446.8 |
| Search + earlier food dispatch; no corridors | 808.92 | 766.3 |
| Earlier food dispatch only | 691.72 | 653.6 |

The older corresponding default scored 872.19. These single-seed comparisons
do not establish a population-level improvement or isolate all interactions.
The exact summaries/settings are in `guide-v1-ablations-20260919.json`.

The frozen nursery v3 Runpod batch completed 12 paired seeds (24 full games).
Mean score was 927.17 with nursery off and 973.30 with two nursery parents;
mean lifetime 903.22 vs 923.49 seconds. No run reached 3000 seconds. One seed
in each arm never established bait, so its zero post-arrival gap is **not** a
continuity success. Nursery produced 63 observed children, of which 19 became
bait. Keep this experimental feature off by default. All artifacts were copied
locally before deleting the pod; billing details are in `runpod-budget.json`.

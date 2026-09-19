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

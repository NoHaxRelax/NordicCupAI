# Recovery and colony supply, 19 September 2026

## Evidence from the spacing replay

Many guide episodes entered recovery within two sightings because the predator
was looking elsewhere. Recovery asked to approach, while the forecast continued
to prefer 100–120 units. Guide 91 started with 431 energy but spent almost its
entire episode returning/waiting; guide 100 started at 500 and used over 400
energy before reaching delivery. Several other guides were assigned below the
20% sprint threshold. This is not a population reliably supplied with fit guides.

At the first bait gap (616.7 s), surviving food collectors were mostly far to
the southeast, while bait occupied a river crevice near the western boundary.
The last incoming bait arrived with 82 energy at age 14.6. By the next handoff,
no candidate passed the conservative travel/lifetime test. Earlier reservation
and a better-fed population are relevant; merely waiting later cannot solve it.

## Recovery spacing experiment (v3)

When the existing not-following latch is true, temporarily target hearing
contact (at most 55 units), then restore the 100–120 following preference.
The native three-tick capture checks still apply. This is an experiment, not
a demonstrated safety improvement.

Seed 1883894846: **score 548.95**, extinction **536.8 s**, runtime recorded in
`guide-reacquire-v3-summary.json`; **2/31** assignments reached delivery,
**5 premature guide captures with sprint available**, 16 overlapping bait
replacements. Worse than v2 (784.88, 0 premature sprint-capable guide captures).
The fatal forecasts did report danger, so a safe three-tick endpoint is not
enough to ensure that recovery leaves an escape available later. Some mutated
agents also have sprint speed below predator sprint speed. Keep this variant
out of the default until it is improved.

The default remains v2. Use `--guide-reacquire-close` to reproduce this
experimental behavior with the current runner. The remote frozen source was
taken before the flag existed, when v3 was unconditional; the checked-in
settings specification now enables it explicitly for reproducibility.

## Paired settings comparison on free PC CPUs

Oscar's `survival-simulator/oscar-overnight-cpp` now contains newer settings at
`a62e04e97a3b571861ae56a69f1ea36b7d9a0a25`, in
`survival-simulator/oscar-overnight-cpp/configs/best-configs.json`.
The reported no-predator mean score is 2599 on 768 fresh seeds in his native
policy. That is a different controller and condition, not a result of ours.

`colony-settings-20260919.json` copies only his survival parameter set. It
compares current settings, newer Oscar settings, earlier bait reservation, and
a higher breeding-energy reserve with earlier bait reservation. No changes
to Oscar's survival implementation are required for these parameter tests.

Frozen remote source: `/home/lucas/colony-settings-20260919`, six workers,
seeds 204871, 605319, 730951, 1883894846, four variants = 16 full games.
All arms use the same experimental v3 recovery controller; do not compare
their raw scores to v2 as if recovery were unchanged. Any useful settings
must subsequently be checked with the chosen guide controller.
Results: remote `logs/paired4`, runner `scripts/compare_colony_settings.py`.
Interpreter: `/home/lucas/entrapment-research-20260918/.venv/bin/python`.
No paid pod is running for this comparison. Each process saves source hashes,
all native metrics and its exact settings. Infrastructure failures remain
separate from game outcomes.

## Subsequent controlled probes

- Contact forecasting v4 (`--guide-contact-forecast`, code `1e6453a`) samples
  resting as well as moving predators and prefers paths retaining sight across
  those possibilities. Dev seed: **440.34 score, 428.4 s, 1/16 delivery
  arrivals, 1 premature sprint-available guide capture**. It regressed and
  remains disabled by default. All frames and compact summary/manifest saved.
- Frontal orbit v5 (`--guide-orbit-recovery`, code `44aef71`) approaches the
  front of a visible predator at the configured preferred radius while the
  not-following latch is true. It retains the same capture checks and leaves
  blind recovery unchanged. Local run folder:
  `logs/entrapment-iteration/guide-orbit-v5-20260919`. Completed: **558.56
  score, 546.8 s, 2/23 arrivals, 3 premature sprint-available captures**.
  Also regressed; default remains off. Native replay at `http://localhost:9079/`.
- A separate local run uses newer Oscar settings with the default v2 guide:
  `logs/entrapment-iteration/oscar-latest-v2-guide-20260919`. This isolates
  settings from recovery changes. Completed: **771.25 score, 737.3 s,
  3/49 arrivals, 2 premature sprint-available guide captures**, versus v2's
  784.88, 750.4 s, 2/19 arrivals and zero such captures. This one seed does
  not establish which survival settings are better; await the paired screen.

The five v3 premature captures involved sprint traits 11.39, 13.34, 11.39,
11.39 and 11.39 in swamp. They had sprint energy, but sprint was slower than
the predator's native 15-unit direct chase. One even had walking trait 12.60
above its sprint trait. Native movement clamps to sprint speed first, even
when walking. Commit `d8daafb` corrects guide, bystander, relief and bait-travel
forecasts for this case. A focused probe verified both guide and avoidance
commands respect 11.39 with walking trait 12.60 and low energy. The in-flight
runs above predate this correction and retain their original source hashes.

No new Runpod pod was created. Six PC jobs remain the free paired screen;
their frozen files must not be edited while it runs. Retrieve the final
`results.json` and per-game manifests/summaries before drawing conclusions.

## Next coordinator issue to address

`_track_predators` retains any living assigned guide before considering fresh
observers as replacements. Shared sightings can keep the track alive even
when its assigned guide has lost sight and spends its energy in recovery.
The replay also assigned guides with energy 43, 59 and 63, already below the
sprint threshold. The Sol relief helper is not yet connected to this role
logic. Handover and candidate viability now deserve priority over more local
steering variants: none of the three steering probes improved the dev game.

## Coordination probe and partial PC results

The fit-guide coordination probe (`68b3b25`) also failed to improve the dev
game: 754.39 vs 784.88 score, still two deliveries, and more ordinary-agent
predator deaths. Keep `--guide-coordination` optional. Details and both exact
manifests are in `guide-coordination-20260919.md` and adjacent JSON files.

The free-PC screen has returned one complete game (current settings, seed
204871: score 1043.25, lifetime 1005.8 seconds) and five infrastructure timeouts
at its 1800-second wall limit. These do not provide a paired settings result.
`colony-settings-partial-20260919.json` records the partial snapshot; full
downloaded artifacts are under `logs/entrapment-iteration/colony-settings-pc-20260919`.
The other ten jobs continue in the unchanged remote source. Future reruns of
timeouts need a longer wall limit; do not classify them as extinctions or draw
a mean from only the runs which happened to finish quickly.

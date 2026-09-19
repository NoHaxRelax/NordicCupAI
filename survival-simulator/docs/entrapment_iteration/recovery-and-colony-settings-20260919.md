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

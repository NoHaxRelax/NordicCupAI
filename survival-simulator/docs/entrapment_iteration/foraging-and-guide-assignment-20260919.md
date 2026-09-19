# Foraging destinations and guide assignment

The completed free-PC survival-settings comparison used three paired seeds
and frozen source `0f348ad`, before the latest handoff/lag corrections:

| Settings | Mean score | Mean life (s) | Mean energy / capacity | Mean bait gap (s) | Reached 3000 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 851.50 | 799.6 | 0.347 | 13.3 | 0/3 |
| Oscar's newer parameters | 1257.54 | 1168.7 | 0.237 | 142.4 | 0/3 |

Newer parameters won on two seeds and lost on one. They improved this small
sample's mean score, but did not establish reliable bait continuity or high
energy, and no game reached the requested horizon. The full results are in
`colony-settings-complete-20260919.json`; all original manifests/configs/logs
are retained in `logs/entrapment-iteration/colony-settings-escape-complete-20260919`.
No default survival parameters were changed based on three games.

## Destination filtering probe

`--safe-foraging` filters fruits, tree posts, watch points and exploration
destinations inside observed predator sensing, occupied-trap exclusion or
predicted guide traffic. It releases existing unsafe claims too. All threat
information is ordinary sightings and the shared observed map. The survival
module gets a small optional destination-predicate hook; default behavior is
unchanged when no callback is supplied. The entrapment module implements the
predicate. Final movement still uses normal avoidance.

On seed 1883894846 this **regressed to 370.04 score / 345.5 seconds**, compared
with 805.35 / 752.0 for the current default. It remains off. Filtering many
resources is not itself evidence of improved survival. Summary and manifest
are saved as `safe-foraging-*.json`, with every frame in the corresponding
`safe-foraging-20260919` replay folder.

## Initial guide selection probe

A native-state evaluator audit of the default replay found that only **7 of
24 unambiguous single-sighting guide assignments** selected the predator's
closest detectable prey before the next agents moved. Several selected
guides were not detectable at that point, and some had only 15–45 energy.
This is evaluator evidence of poor assignment alignment, not access to a
hidden target ID or proof of its choice after subsequent agent actions.
Exact cases: `guide-assignment-target-audit-20260919.json`.

`--guide-chased-only` tests a simpler policy-side rule: among unassigned
observers, select the nearest one within the predator's public hearing/vision
geometry. Hearing works through walls; vision requires facing and clear LOS.
If our own sighting could have come through hearing, known walls are checked.
No strict route-energy viability gate is added. Existing guides retain their
ownership; this probe changes initial assignment only. Normal following
detection and capture avoidance still apply. The sighting itself is delayed,
so this remains an estimate of who is being chased.

The standalone dev-seed game is running with destination filtering off and
all frames retained in `guide-chased-only-20260919`. The new option remains
off until outcomes support adoption. No paid compute was used.

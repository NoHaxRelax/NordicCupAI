# Simplified adjacent-start guiding

Current work is in [SESSION3_RESULTS.md](SESSION3_RESULTS.md); the preceding snapshot is [SESSION2_RESULTS.md](SESSION2_RESULTS.md). The text below records the original pilot and its old budget; it is historical, not the latest result.

## Original pilot

User requested an already-engaged guide near a predator, continuous gaze during
travel, and a simple final approach. This experiment isolates that task: bait
is predeployed at depth 5; guide starts 55 units from the predator, both facing
each other; predator starts awake with full energy. These are disclosed setup
interventions, not random independent starts. There is no child. Native
predator energy/rest/movement continue unmodified after setup. Agents have
infinite energy. Runtime controller receives only native DTOs, static map,
role IDs and public time, never setup positions or hidden predator state.

`policy.py` imports the frozen v15 static map/localization/movement helpers,
but replaces the guide state machine with continuous gaze and reachable
movement candidate scoring. Walking is preferred when separation allows it;
sprinting is allowed for spacing. The axial final lead and sacrifice follow
the earlier arranged strategy. Native rules chase directly below 90 units even
when watched; hearing is 60 units. The present preferred separation is 65,
which is still outside hearing during rest and needs reconsideration.

First pilot: map 5101, fixture 9203, requested 60 seconds, stopped by the user's
60%-remaining quota at 23.2 seconds. Receipt/replay suffix `76cfee4a`.
233/233 native frames saved, including the initial frame. The evaluator
confirmed pursuit at 0.1 seconds. Both bait and guide survived; zero physical
entries and zero captures before interruption. Do not count this partial run
as a full-horizon delivery failure or success.

Observed checkpoints (evaluator only):

| Time | Guide position | Predator position | Separation |
| --- | --- | --- | --- |
| 0 | 1425.5, 685.0 | 1445.5, 736.2 | 55.0 |
| 4 | 1080.8, 519.5 | 1147.8, 531.3 | 68.0 |
| 8 | 797.4, 516.1 | 858.8, 530.7 | 63.1 |
| 12 | 553.0, 449.8 | 618.7, 467.0 | 68.0 |
| 20 | 390.7, 417.3 | 442.6, 424.9 | 52.5 |
| 23.2 | 335.7, 477.0 | 311.5, 465.4 | 26.8 |

This is substantial cross-map transport but not capture evidence. Inspect the
late turn/alignment and stale-observation safety before claiming reliability.
No energy saving was measured: agent energy is refilled. Starting awake is a
setup exception, and no actual predator target is forced; pursuit is verified
by the evaluator. The initial attempt rejected the bait spawn before any
simulation tick because native spawn has a larger box restriction than the
agent collision radius; the final harness explicitly positions the fixture
bait at a radius-valid goal before recording.

Run from the handoff root with the simulator requirements installed:

```sh
python survival/research/simple_chase/run.py --policy simple_chase.policy:SimpleChase \
  --seconds 60 --map-seed 5101 --fixture-seed 9203 --station-bait
```

Do not remove the quota sentinel without a new authorized budget. The user authorized all remaining usage; the shared stop threshold is now 0%. The every-frame viewer now indexes these pilots:
http://127.0.0.1:9055/research/real_map_visualization/index.html .
The old 9053 full-library server is stopped to avoid loading large recordings
into RAM. Use debugger/catalog.py --verify-stream for safe discovery checks.
Original recordings remain local under the existing gzip/replays ignore policy.

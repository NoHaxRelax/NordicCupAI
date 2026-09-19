# Edge pockets and passive predator migration

Experimental continuation of the expanded-local-food confinement census. The
original orchard configuration stays unchanged; use `gentle-config.json` to enable
the candidate. All decisions remain in the native C++ observation-only policy.

## Geometry evidence

Regenerating geometry for all 1,000 frozen census seeds (19001–20000):

- 4,810 / 4,902 confinement episodes (98.12%) have a bounding-box centre within
  35 units of **two distinct obstacle rectangles**.
- For episodes lasting at least 300 seconds, this is 2,871 / 2,944 (97.52%).
- Median centre distance to the nearest two obstacles: **12.51 and 14.37 units**.
- Random free-space controls: 31,982 / 352,211 (9.08%) satisfy the same test.
  Controls exclude centres less than 10 units from an obstacle.
- 855 / 1,000 maps have an observed confinement episode satisfying this geometry.
  This is not a count of maps on which deliberate delivery has been verified.

This supports searching for narrow two-surface pockets, frequently a rock near the
map boundary. All ten previously selected clips have this pattern. Native wandering
reacts to the nearest visible edge; competing surfaces can sustain small loops or
stalls. Corners and entry heading matter. Geometry is a candidate filter, **not a
98% capture-success claim**. Episode centres can differ from actual instantaneous
positions; the random controls and episode centres are not equivalent samples.
The study does not establish that every pocket is accessible from outside.
The online detector has edge segments rather than obstacle identities, so its
two-surface proxy can also admit some single-rock corners. It is not identical to
the offline two-rectangle measurement.

## Small paired full-game pilot

Same 16 fresh seeds 21001–21016 in each arm, policy seed 0, ordinary energy, naturally
spawning predators, 3,000-second horizon. PC CPU only; up to 12 total research
workers including geometry/recording. No Runpod. All variants start from the frozen
expanded-local-food config used in the census.

| Variant | Mean score | Paired gain | 95% interval for gain | Confined predator time |
|---|---:|---:|---:|---:|
| Baseline | 1568.9 | 0.0 | — | 15.06% |
| Edge bias | 1667.5 | +98.6 | −108.6 to +305.9 | 15.43% |
| Edge bias + protection | 1657.5 | +88.7 | −124.5 to +301.9 | 15.41% |
| **Gentle, close encounters** | **1739.8** | **+170.9** | **−48.4 to +390.3** | **22.05%** |
| Strong, close encounters | 1549.0 | −19.8 | −130.3 to +90.7 | 19.25% |

The gentle variant gains an estimated 10.9% score and 6.99 percentage points of
confinement. These are exploratory estimates: variants were selected on these maps,
there is no separate holdout, and the ordinary paired Student-t intervals do not
correct for trying several variants. No reliable score improvement is established.
Confinement is still the spectator's >60-second, 40×40-box criterion, sampled each
second; the table averages each game's fraction of predator-time in these episodes.
It does not count how many successful intentional deliveries occurred. Changes in
survival, exploration and predator spawns can affect this metric too.

## What the candidate does

1. During encounters within 70 units, find a nearby pocket from observed edge
   segments. A proposed point sits 13 units off one surface and within 35 units of
   another non-collinear surface. Reject points crossing an observed wall or closer
   than 10 units to any observed edge. Search reach is 180 units.
2. Compare 24 movement headings at the action's existing movement distance. Prefer
   preserving the orchard action, separation from predators, and a predicted
   predator heading toward the pocket. The gentle steering weight is 10.
3. Maintain spatial occupancy patches in each group's learned coordinate frame.
   Repeated sightings within a 20-unit-radius patch for 60 seconds, with no gap
   over two seconds, mark it as observed confinement. This is **patch occupancy**,
   not a stable predator identity; the DTO supplies no predator IDs.
4. Penalize movement toward confirmed patches. Fresh observed predator headings add
   a cone-avoidance penalty; otherwise use an 85-unit exclusion disc. Forget a patch
   after 120 seconds without a sighting. Protection weight is 2.

No bait, forced spawns, free energy, hidden positions, preloaded full-map geometry,
or offline census coordinates enter the policy. Information is shared within the
existing mapped agent groups. The orchard and reproduction parameters are unchanged.

This is a low-cost **bias**, not a delivery controller or certified safety planner.
It does not guarantee departure within ±10 degrees, preserve every trap, or protect
every agent. Safety uses a conservative distance penalty, not a full rollout;
walls, unseen observation-delay movement, terrain, and multiple prey remain limits.
When the existing action is stationary, this layer leaves it stationary. Long
observation gaps reset confirmation, so many real trapped predators never qualify.

The “Turn predators” controller supplied the correct public heading reconstruction
(`angle + pi - rel_dir`) and the important distinction between last heading and
later wandering. Its full stateful maneuver is not ported here: it was qualified
for one prepared guide in open space, not these multi-agent wall encounters. The
current close-encounter forecast uses the native clamped pursuit-turn rule.

## Files and reproduction

- `_evasion.hpp`: `edge_migration`, called inside `act` before action-memory updates.
- `_orchard_policy.cpp`: four opt-in parameters. Zero weights preserve baseline.
- `gentle-config.json`: complete candidate config and pilot estimate.
- `geometry.json`, `pilot16/`, `narrow16/`, `summary.json`: raw measurements/configs.
- `scripts/analyze_stuck_geometry.py`: offline census geometry/control analysis.
- `scripts/bench_edge_migration.py`: paired games and confinement measurement.
- `scripts/report_edge_migration.py`: table and paired intervals.
- `scripts/record_edge_migration.py`: every-tick comparison replay, first pilot seed.

From `survival-simulator`, after building `fastsim/build.py` and
`fastsim/build_policy.py` with the project Python environment:

```bash
python fastsim/check_boundary.py
python scripts/bench_edge_migration.py --out /tmp/edge-pilot --workers 12 \
  --count 16 --seed 21001 --arms baseline,edge_bias,edge_protect
python scripts/bench_edge_migration.py --out /tmp/edge-narrow --workers 12 \
  --count 16 --seed 21001 --arms gentle,urgent
python scripts/record_edge_migration.py --folder /tmp/edge-replays \
  --seed 21001 --arms baseline,gentle
python scripts/stuck_replay.py serve --folder /tmp/edge-replays --port 9102
```

Native compilation and the observation-boundary check passed. Replay recording
independently checks equality with an uninterrupted full-game run. The initial
pilot's build provenance is stored separately; subsequent manifests embed it.
No main BO evaluation or existing predator-video service was changed.

Local viewer: **http://localhost:9102/**. Select `baseline`, `edge_protect`, or
`gentle` for seed 21001. All three replay scores exactly match the PC pilot; every
0.1-second frame is retained. Recordings live in `/tmp/edge-migration-replays` and
on the PC under `/home/lucas/edge-migration/survival-simulator/results/replays`;
large recordings are not committed. The local service is
`edge-migration-replay.service`. The first pilot seed was chosen independently of
its score, not as the strongest demonstration of trapping.

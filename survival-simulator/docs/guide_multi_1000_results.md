# 30 preloaded predators + one new delivery: 1,000-map evaluation

Batch: `logs/guide_batch/runpod-20260918-30plus1-1000/`.
Versioned results: [aggregate statistics](guide_multi_1000/aggregate.json),
[all 1,000 cases](guide_multi_1000/cases.json),
[seeds and source hashes](guide_multi_1000/manifest.json), and
[trace verification](guide_multi_1000/integrity.json).
The checked-in policy, simulator, and evaluation runners match all 21 recorded
source hashes. Large raw tick traces, generated replay images, and the local
duplicate source archive remain in the ignored batch directory, outside Git.

The frozen policy and simulator are in its `source/` directory; their exact
hashes are recorded in `manifest.json`. This is the newer 55-unit delivery
strategy, not the policy from the earlier single-predator evaluation.

## Protocol

- Same 1,000 unique map seeds and encounter seeds as the earlier batch:
  batch seed `9182026`, native 1,600 × 1,200 maps.
- Place 30 predators in the validated front approach lane, then settle for
  10 simulated seconds. Require all 30 held and the rear entrance clear.
  Delivering these first 30 is not tested.
- Add one randomly placed visible predator/guide encounter, at least 250
  units from bait. The guide starts full and then uses native energy/aging.
- Bait stays full-energy and unaged but can be eaten. Ambient predator
  spawning is disabled. Predators retain native movement, sensing, energy,
  resting, and overlap behavior.
- Guiding uses ordinary observations, known static edges, and relative bait
  localization. Omniscient measurements are evaluation only.
- Allow up to 60 seconds for delivery, stopping the delivery phase after a
  10-second hold or 20 seconds after guide death. Then observe another
  30 seconds with any surviving guide standing still.
- A held predator is within 40 units of bait and senses it. A pass requires
  all 31 held throughout the final 30 seconds, all original 30 continuously
  retained after settling, bait alive, and the replacement side clear.
- The replacement side is the half-plane past the gap midpoint toward its
  rear mouth, within 60 units of bait or the replacement entry. Occupation
  by **any** predator during the final 30 seconds counts as failure. This is
  conservative relative to checking only the final frame.
- Missing sites, setup failures, and computational timeouts remain in the
  overall 1,000-map denominator. Conditional rates use stated denominators.

Rear clearance is a geometric/hearing-distance proxy; this does not simulate
an actual replacement agent. The final 30-second hold is not whole-game
retention. Guide death itself is allowed. A transient departure beyond the
40-unit hold zone by an original predator counts as a retention failure.

## Reproduce

Interactive overview: <http://127.0.0.1:9058/>. Filter by final outcome or rear
visits, select a case, and load its native game replay. Rendering uses the frozen
batch source locally; results and final states are checked against the original
recording. Two local workers handle rendering, with at most four cases queued.

Restart the overview with:

```sh
.venv/bin/python survival-simulator/scripts/guide_batch_viewer.py survival-simulator/logs/guide_batch/runpod-20260918-30plus1-1000 --port 9058
```

Use the batch's archived `source/` and `runtime_versions.json`, then run:

```sh
python source/scripts/guide_batch.py --multi --maps 1000 --workers 32 --seed 9182026 --timeout 300 --output results
```

For a native every-frame replay of one case, use its saved seeds:

```sh
python source/scripts/guide_multi.py --deliveries 1 --seed MAP_SEED --encounter-seed ENCOUNTER_SEED --output REPLAY_DIRECTORY
```

The bulk batch saves every tick to `ticks.jsonl.gz` without rendering PNGs.
`guide_multi_report.py` consolidates results and validates source hashes;
`verify_guide_multi_batch.py` checks gzip integrity, tick counts, and final-state
agreement. The trace field named `all_33_continuous_seconds` is a legacy name:
for this one-delivery configuration it measures all **31** predators held.

## Results

**682/1,000 passed (68.2%)**, or **682/953 maps with valid sites/preloads
(71.6%)**. Wilson 95% intervals: 65.2–71.0% across all maps and 68.6–74.3%
among eligible maps. All 953 eligible maps successfully settled their initial
30 predators. There were no worker errors or computational timeouts.

Using the requested **wrong side at the final frame** rule, mutually exclusive
primary results are:

| Result | Cases |
| --- | ---: |
| Delivery and retention passed | 682 |
| New delivery or final retention failed | 141 |
| Original predator(s) left the hold zone, without a rear-at-end failure | 124 |
| Predator(s) on the replacement side at the end | 6 |
| No usable bait site | 47 |
| Total | 1,000 |

The stricter recorded runner also rejects any rear occupation during the final
30 seconds: 17 cases meet that condition, including the six above. Its primary
failure categories are 17 rear failures, 116 original-retention failures,
138 other delivery/retention failures, and 47 missing sites. **The pass count
is identical under either rear-side rule**: the additional transient rear
visits also fail delivery or original retention independently.

In total, 127 runs had at least one original predator leave the hold zone;
some also had a rear-side failure. 722 runs held all 31 during the final
30 seconds before applying the rear-side and earlier-original-retention checks.

The batch ran on 32 Runpod CPU workers in **849.1 seconds (14 minutes 9 seconds)**.
All case results, raw traces, source, seeds, and runtime versions are saved in
the batch directory. Case zero reproduced locally with identical timing,
delivery result, final predator states, and retention measurements.

Machine-readable results: `aggregate.json` and `cases.json` in the batch
directory. `outcome_rear_at_end_only` gives the final-frame classification;
`outcome` preserves the stricter original run classification.

All **953 complete gzip traces / 652,674 ticks** passed integrity, sequential
tick, frame-count, end-time, and final-state checks. The 47 missing-site cases
have setup outcomes rather than simulation traces.

The dedicated pod `x2xr6km7ful09f` was created at 12:57:50.981 UTC and its
deletion verified by HTTP 404 at 13:17:25 UTC. Estimated compute cost is
**$0.313**, plus the small 20 GB ephemeral-disk charge, within the $1 budget.
This is a rate-times-lifetime estimate, not a settled invoice. No other team
pods were changed.

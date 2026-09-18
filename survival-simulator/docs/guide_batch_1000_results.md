# Guiding on 1,000 random maps — 18 September 2026

The unchanged survival/contact/destination policy delivered in **794/1,000
maps (79.4%)**, or **794/953 maps with usable bait sites (83.3%)**. The
guide remained alive at the end of **690** successful runs. This does not
meet the 95% delivery target.

| Result | Cases |
| --- | ---: |
| Delivered, guide alive | 690 |
| Delivered, guide dead | 104 |
| Guide died without delivery | 112 |
| Still alive but no delivery by 60 seconds | 47 |
| No usable bait site | 47 |
| Total | 1,000 |

**Bait-site availability:** 953/1,000 = 95.3%, with a 95% Wilson interval of
93.8–96.4%. A geometric site includes a second access point for replacing bait;
this test does not verify replacement under live predator pressure.

**Delivery reliability:** the 95% Wilson interval is 76.8–81.8% across all
maps, and 80.8–85.5% among maps with a bait site. Delivery with the guide alive
occurred on 69.0% of all maps or 72.4% of eligible maps.

**Maintaining contact:** the tracked predator could sense the live guide on
195,412/235,836 measured ticks (82.9%). There were 133 runs with detection on
every alive tick. These are correlated ticks, not independent trials, so no
binomial confidence interval is applied to this measure.

## Protocol and limitations

- Batch random seed `9182026`; 1,000 distinct uniformly sampled map seeds and
  one generated encounter seed per map. All seeds are saved in `manifest.json`.
- Native 1,600 × 1,200 random maps, biomes, collision rules, food, energy, aging,
  and predator behavior. One selected geometric trap site per eligible map.
- The guide starts with full energy and sees an awake predator 80–160 units
  away. The encounter is at least 250 units from bait. Finding a predator is
  outside this test.
- Static edges and exact relative localization are known. The guide gets no
  hidden predator coordinates, target, resting state, or energy.
- Bait is stationary, held at full energy, and does not age. It can be eaten.
- A delivery passes after the tracked predator continuously spends 10 seconds
  within 40 units of bait while sensing it. This is a local delivery proxy,
  not a full-game retention or causation test.
- Runs stop at success, bait death, 60 seconds, or 20 seconds after guide death.
  Guide survival is survival to that stopping time, not necessarily 60 seconds.

## Runtime and cost

One dedicated Runpod `cpu3c` pod with 32 vCPUs and 64 GB RAM ran 32 workers.
The main batch finished in **328.7 seconds (5 minutes 29 seconds)**. Cases
276 and 409 (zero-based indices 275 and 408) exceeded the initial 180-second
worker limit. Both were rerun with the same seeds and unchanged policy; both
eventually ended with the guide dead and no delivery. Their original partial
traces are preserved, and the complete retries supply the final result.

The slow cases repeatedly attempted an unsuccessful predator-width route
search; this computational cost remains a limitation of the current planner.

Pod `rzr2n0by6iu7tm` was created at 11:20:45.292 UTC and deleted by 11:40:07 UTC.
Deletion was verified by the API returning 404. Estimated compute plus 20 GB
ephemeral disk cost is **about $0.31**, below the $1 cap. Runpod had not yet
posted a billing record at cleanup, so this is a rate × lifetime estimate,
not a settled invoice. Other team pods were not changed.

## Saved data and viewing

The versioned statistics are in [guide_batch_1000/](guide_batch_1000/):
[aggregate totals](guide_batch_1000/aggregate.json),
[all 1,000 case results](guide_batch_1000/cases.json),
[seeds and source hashes](guide_batch_1000/manifest.json), and
[slow-case setup audit](guide_batch_1000/setup_audit.json).
The policy and simulator files in this commit match all 17 source hashes in
that manifest. The lab runner, map-site selector, batch runner, replay HTML,
and requirements also match the archived batch source byte for byte.
Later multi-predator and sacrificial-handoff experiments are not part of this
tested policy. Large per-tick traces and rendered frames remain local; they
are not included in Git.

The complete artifact directory is `logs/guide_batch/runpod-20260918-1000/`:

- `aggregate.json`: final counts, denominators, confidence intervals, limitations.
- `cases.json`: consolidated cases with completed retries reconciled.
- `manifest.json`: all seeds, configuration, and source hashes.
- `case-*/`: original worker log, result, frozen policy, and compressed tick trace.
- `retries/`: completed retries for the two slow cases.
- `setup_audit.json`: independent confirmation that both slow cases had bait sites.
- `source/`: frozen simulator and policy used for on-demand replay rendering.
- `replays/`: locally generated native frames and per-tick inspection files.

All **953 complete simulation traces and 268,981 ticks** passed gzip integrity
and frame-count checks. The 47 maps without a site have setup results instead
of simulation traces. Two example replays reproduced cloud actions and states
at every tick; native observation-list ordering differed, but the contents matched.

Open <http://127.0.0.1:9057/> for the overview. Case #0002 is a delivery with
guide survival; case #0004 is an early capture without delivery. Other cases
can generate a replay locally on demand. The slow cases may exceed the local
replay rendering limit; their complete recorded traces remain available.

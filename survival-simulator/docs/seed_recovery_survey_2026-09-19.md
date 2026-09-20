# Thirty-second shared-map survey before seed search

19 September 2026. Follow-up to the [data budget and joint-constraint experiments](seed_recovery_data_budget_2026-09-19.md).

**Implemented and tested: walking and looking around for 30 simulated seconds substantially reduced expensive candidate verification.** Across three in-range targets, the shared-map biome filter left exactly the correct seed among 100,000 candidates. Each recovered seed then reproduced all 300 public frames under the recorded actions. A fourth target outside the searched range correctly produced no match in that range.

This demonstrates bounded recovery without a prebuilt map catalog. It does not establish uniqueness or practical search time over all 2^32 possible seeds.

## What the survey does

[seed_survey_probe.py](../scripts/seed_survey_probe.py) uses the existing `ExpertPolicy`, `GlobalPlanner`, `WorldEstimator` and exploration coordinator. Only public observations and the actual chosen actions enter these components. The simulator is unmodified.

The first six actions are stationary 60-degree turns, preserving the earlier 0.7-second collection baseline. Agents then walk and scan until simulated time 30.0. The local experiment configuration keeps scouting and looking active after alignment, uses a walking fraction of 0.6, disables optional scouting/food sprints and reproduction, and retains ordinary survival behavior. All effective policy/planner settings are stored in the evidence. Predator and tree dynamics remain native.

The shared model makes the observations useful together by:

- Aligning agent groups through boundary geometry, identified agent sightings and shared landmarks.
- Correcting position estimates from repeated rock/tree observations instead of assuming commanded movement always succeeds.
- Transforming retained geometry and observed biome samples when groups join the same world frame.
- Deduplicating visited biome cells and retaining observation uncertainty.

All five founders survived every tested survey. Each run ended with all five agents localized in one shared world frame. The lowest final agent energy across the four runs was approximately 68.48; no births were requested. This is not a long-game score comparison against the ordinary policy.

## Which map information enters the search

The filter uses actual non-river biome observations retained by anchored map groups. It does **not** turn the biome estimator's predictions into observed facts. Optional biome fitting, trap inference and predator-belief propagation are disabled in this experiment's policy configuration; the shared geometric estimator and exploration coordinator remain active.

Samples with declared position uncertainty above 6 world units are excluded. The rest receive a radius of `6 + uncertainty` around their estimated position. The extra margin accommodates localization and integer-pixel uncertainty. These radii are conservative heuristics, not calibrated probability bounds.

Up to 64 samples are selected by geographic separation and observed label diversity. Selection does not inspect candidate seeds or hidden simulator state. At 30 seconds, the four maps retained 115–147 biome cells; 70–132 non-river samples were eligible before selecting 64. The sampled cells and their labels are correlated, so their count should not be interpreted as independent random outputs.

For each candidate, the filter generates only the ten early Voronoi sites and land types. If the true sample point is within radius `r` of the estimate, a site of the observed type that could own that point must satisfy:

```text
distance_to_nearest_site_of_observed_type
    <= distance_to_nearest_site_of_any_type + 2*r
```

This is a necessary, deliberately permissive condition. It can retain impossible candidates but avoids requiring the candidate to match exactly at a potentially noisy estimated pixel. River labels are excluded because the river overwrites the base land map.

The true seed passed every checkpoint's filter in all four runs, including the out-of-range target when checked individually for validation. Hidden position audits found no low-uncertainty pose outside its assigned radius. Maximum audited absolute-position error across the runs was about `1.14e-11` world units. These favorable measurements apply to these runs; they do not guarantee that future map alignment or collision recovery will be equally accurate. Hidden coordinates were never used to select samples, modify radii or steer agents.

## How much the extra time helped

All checkpoint searches were performed **after the full 30-second survey**, using stored map snapshots. The search range was 0–99,999, inclusive.

| Target seed | Biome survivors at 0.7 s | At 10 s | At 20 s | At 30 s |
| --- | ---: | ---: | ---: | ---: |
| 3 | 156 | 6 | **1** | **1** |
| 11 | 1,825 | 3 | **1** | **1** |
| 78431, additional held-out target | 100,000 | 173 | **1** | **1** |
| 20260919, outside range | 26,093 | **0** | **0** | **0** |

The 0.7-second counts use the same shared-model uncertainty treatment as the later checkpoints. They therefore differ from the earlier nearly exact boundary-point filter's counts.

Seed 78431 is a useful case: at 0.7 seconds the shared model had no anchored agent, so the filter correctly rejected nothing. At 10 seconds only one agent was anchored. By 20 seconds all five were in one anchored shared frame, and the combined biome samples isolated the correct candidate. No prebuilt map catalog for this seed was used.

| Target | Distinct rock dimensions at 0.7 s | At 30 s | Anchored agents at 0.7 s | At 30 s |
| --- | ---: | ---: | ---: | ---: |
| 3 | 28 | 105 | 4 | 5 |
| 11 | 54 | 115 | 4 | 5 |
| 78431 | 33 | 96 | 0 | 5 |
| 20260919 | 38 | 98 | 2 | 5 |

The surviving candidates were checked against all recovered founder headings and distinct rock dimensions. The verifier then initialized a fresh native world and replayed the complete recorded action sequence. Seeds 3, 11 and 78431 each matched **all 300 public frames**, with observation list ordering normalized. No candidate remained for the out-of-range target. Later trees were handled through replay rather than incorrectly comparing 30-second tree observations against an initial-tree catalog.

Twenty seconds already sufficed to isolate the tested in-range seeds within this range. Thirty seconds added map coverage and geometric evidence. The result supports the proposed collection period, while suggesting that eventual stopping criteria should consider alignment and evidence quality rather than a timer alone.

## Timings and limits

The subsequent [Rust implementation](seed_search_rust.md) completed an actual **10-million-seed scan in 6.96 seconds**, or **15.14 seconds including native verification**, using the held-out survey and 20 CPU threads. The Python timings below remain the earlier baseline.

- Each 30-simulated-second survey took approximately **4.32–4.71 seconds of local wall time**, including world initialization and policy/model work. A live game may advance at a different pace.
- Evaluating all four checkpoints for the original three targets against 100,000 seed prefixes took **8.70 seconds**. This measures twelve checkpoint signatures, not a single final-frame query.
- The additional held-out target's four-checkpoint search took **3.79 seconds**.
- Only **one native candidate world per in-range target** needed verification, taking about **1.67–1.89 seconds to initialize**, followed by action replay. The old short-survey test needed 99 candidate worlds for seed 3.

A subsequent benchmark used only seed 78431's final 30-second signature (64 observed biome samples) and scanned **1,000,000 candidate prefixes in 22.49 seconds**, about **44,469 seeds/second**. Only seed 78431 passed. Linear extrapolation gives **224.9 seconds, or about 3 minutes 45 seconds, for 10 million candidates** with this single-process Python implementation. This excludes observation collection and native survivor verification. It is an extrapolation from one target, not a completed ten-million scan or a guarantee that only one candidate will survive the larger range. Measurements and input/source hashes are in [seed_survey_million_benchmark_2026-09-19.json](seed_survey_million_benchmark_2026-09-19.json).

Search still enumerates candidate integers. A full-range scan, robust behavior on difficult maps, and live search latency remain untested. Four targets are insufficient to estimate population-wide failure or false-match rates. The filter must retain its uncertainty handling and a final replay check; predicted terrain or incorrectly merged maps could otherwise eliminate the true seed.

## Artifacts and validation

- [Collector, filter and replay verifier](../scripts/seed_survey_probe.py).
- [Three-target experiment](seed_survey_probe_2026-09-19.json).
- [Additional held-out target](seed_survey_heldout_2026-09-19.json).

Both records include effective settings, source hashes, actual actions, public-frame hashes, selected biome samples, checkpoint survivors and replay outcomes. All final survivors were verified. Four focused checks passed for uncertainty near a biome boundary, absent biome types, empty evidence and spatial sample selection.

Source manifests differ between the two runs in `predator_belief.py` and `map_renderer.py` due to concurrent workspace changes. Predator-belief propagation was disabled, and the map-renderer helper was not used. The collector, native generation code and shared-world estimator were unchanged between the runs. No production policy, simulator or configuration file was edited for this experiment.

Reproduce from `survival-simulator`, choosing new output filenames:

```text
python scripts/seed_survey_probe.py --output docs/seed_survey_repeat.json
python scripts/seed_survey_probe.py --targets 78431 --output docs/seed_survey_heldout_repeat.json
```

The default native-verification cap is 20 worlds per target. If the biome filter leaves more survivors, or the true seed fails the offline validation check, the result explicitly records skipped verification and `verification_complete=false` rather than claiming recovery.

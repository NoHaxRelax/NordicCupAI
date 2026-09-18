# Frozen 9059 benchmark results

The user stopped the experiment after **991 of 1,000 games completed**. Nine outcomes remain unknown. All 12 Runpod pods are confirmed deleted. Estimated compute cost is **$7.97**; billing had not fully posted at shutdown. Small disk charges are additional and fit within the $10 cap.

The exact strategy is commit **9b0c3e8**, with all 39 original source hashes verified. The executed batch wrapper was commit **7eaef09**. Seeds 0–999 were assigned without map filtering; each completed native game ran until extinction or a 3,000-second horizon. No infinite energy, known-map input, forced predator encounters, or privileged controller inputs were used.

## Completed-game results

| Metric | Result |
|---|---:|
| Completed / unfinished / simulation errors | 991 / 9 / 0 |
| Survived to 3,000 seconds | 0 / 991 completed games |
| Mean / median final score | 637.53 / 639.41 |
| Score range | 132.90–1,491.62 |
| Mean / median survival | 621.38 / 622.10 seconds |
| Maps with geometrically valid sites | 949 / 991 |
| Maps where policy reported a trap | 966 / 991 |
| Physical bait established | 708 / 991 |
| Guide delivery arrivals / assignments | 2,676 / 14,556 (18.38%) |
| Baited games with rear occupation | 335 / 708 |
| Mean / maximum simultaneous Held30 predators | 2.07 / 11 |

These are descriptive statistics for the **completed subset**, not an unbiased estimate over all 1,000 requested games: slower runs were more likely to remain unfinished. Unfinished seeds: **287, 620, 735, 752, 824, 860, 874, 920, 932**. Their last saved progress appears in `summary.json`; partial scores are not included in the final-score histogram.

A guide arrival is not necessarily a capture. Held30 means within 40 units of physically verified bait for 30 seconds, not permanent retention. Bait gaps include the final loss of bait before colony extinction. Reported trap discoveries can exceed geometric validity because the frozen 9059 version retains its original map-boundary reconstruction bug. It also retains the age-55 reproduction cutoff without the later emergency fallback.

## Interruption and reproducibility

The pilot pod disappeared before its second batch finished; its cause was not established. All 53 downloaded completed cases in that shard were preserved, and the remaining 28 seeds were restarted on an existing pod using unchanged source. Twenty-two of those restarts completed before shutdown; six remained unfinished. Three other unfinished cases remained on their original pods. `interruption.json` records the affected seeds. No result was discarded because its outcome was undesirable.

Policy timing and native ordering effects mean seed alone does not guarantee a bit-identical trajectory on another process or CPU. Treat this as an incomplete benchmark with a documented infrastructure interruption, not a claim of 1,000 fully reproducible completed games.

## Visualization and records

Open **http://localhost:9061** for the score histogram, score-range filtering, case table and recorded tick playback. The replay uses native creature drawing and saved positions, energy and roles. Fruit/tree states and controller inputs/actions were not recorded in this compact batch format.

`cases.json` contains all 991 completed results; `summary.json` contains aggregate metrics, source hashes and the nine incomplete cases. `infra-final.json` records pod lifetimes and conservative estimated compute costs. Raw trajectories remain under `survival-simulator/logs/entrapment_benchmark/runpod-20260918-9059-1000/` locally, outside Git. See `../entrapment_benchmark_viewer.md` for launch instructions.

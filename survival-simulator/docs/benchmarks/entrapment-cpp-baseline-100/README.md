# Auxiliary orchard baseline on the free PC

100 complete games, seeds 2026091900–2026091999, maximum duration 3000 seconds,
with predators, normal energy and aging. Frozen controller commit: `04173f8`.
Configuration: `native_policy/configs/with-predators-best.json` (trapping off).

Mean score **1563.783**, median **1557.273**, mean bootstrap 95% interval
**1503.749–1624.992**. All 100 populations became extinct before 3000 seconds.
Two workers took **1231.24 seconds** on the free Ryzen 5 3600 PC.

These are auxiliary results. The final paired comparison uses the separate
Runpod baseline and entrapment batches, compiled and run in the same environment.
Do not mix these PC rows into the Runpod batch.

`manifest.json` records source, binary, runtime and result hashes;
`games.jsonl` retains every result. See `score-histogram.svg` for the distribution.

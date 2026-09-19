# Completed frozen evaluation

All 12,000 games completed on 2026-09-19. Jobs launched approximately 15:16:31 UTC;
the slowest shard took 545.67 seconds (**9.1 minutes**). Summed job time was
5051.11 pod-seconds, approximately **$1.35 active compute** at $0.96/pod-hour.
This excludes setup, storage and idle pod uptime, and is not a billing invoice.
All ten Oscar CPU pods were left running as requested.

Verified 1200 rows per shard, exactly 1000 unique paired map seeds 10001–11000
per model, identical configurations and source hashes across shards, local source
hash matches, and positive tick/interface/policy/engine/CPU/wall timing fields.
The reporter verifies map ordering before paired bootstrap resampling.

Runner source afdecaf, allocation 42d32f6. Profiling was enabled using the native
engine's existing timing path; no policy, simulator rules, configurations, or
objective were changed during evaluation. Seed 9999 was used only for a local
timing-field smoke check, excluded from all reported results.

The score leader is scheduled_breeding, mean 1560.0 (95% CI 1534.1–1585.3).
Its paired advantage is +68.3 over the previous winner (CI 41.4–95.2), and
+93.0 over the original baseline (CI 65.3–120.8). These are pointwise intervals;
the report does not claim multiplicity-adjusted significance for all comparisons.
All 66 pairwise differences and raw timing breakdowns are saved.

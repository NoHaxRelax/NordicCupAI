# Frozen final evaluation

Evaluate all ten families20 winners, the previous population winner and the
original baseline on the same 1000 fresh maps: seeds 10001–11000. These 12
configurations are frozen before this evaluation; no further tuning occurs.
Total 12,000 full games, normal energy/predators, 3000-second horizon or extinction.
Policy random seed is fixed at zero independently of world seed.

Ten existing CPU pods each receive 100 maps and ALL twelve configurations.
Randomize job order within each shard, 32 workers. This balances hardware across
models rather than assigning a different machine to each model. Leave pods running.

Enable existing native per-tick profiling. Record public-interface, policy and
engine elapsed nanoseconds, total native-loop elapsed time, per-process CPU time,
and tick count for every game. Exclude generation and initialization. Report
tick-weighted means: sum time / sum ticks. A tick updates the entire population.
Wall measurements include scheduling delays under 32-worker contention; they
are not isolated policy CPU times. Profiling does not supply extra policy inputs.

Report each score mean with 95% bootstrap CI (10,000 map resamples), differences
against both controls with paired CIs, and all pairwise differences as an artifact.
Do not treat pointwise CIs as multiple-comparison-adjusted evidence. Preserve raw
per-map results, config/source hashes, and environment metadata. No competition
validation attempt. Expected runtime around 10–15 minutes based on prior runs.

# Burst score pilot

68 shared fresh seeds (91001–91068), eight variants, 544 complete local games. 95% intervals use 10,000 paired map bootstrap resamples. These are exploratory selection results, not an independent final test.

| Variant | Mean score | 95% mean CI | Successful transfers | Games above 20k | Survival seconds | Runtime seconds |
|---|---:|---|---:|---:|---:|---:|
| fast1m | 45,276.9 | 42,754.9–47,903.0 | 597/601 | 68/68 | 1282.5 | 16.2 |
| lowpop400k | 19,665.1 | 18,504.4–20,900.4 | 622/628 | 28/68 | 1272.2 | 13.6 |
| birth400k | 19,540.8 | 18,346.5–20,738.4 | 615/625 | 29/68 | 1350.3 | 16.5 |
| population400k | 19,105.3 | 17,800.3–20,439.3 | 600/609 | 28/68 | 1352.9 | 16.7 |
| fast400k | 18,938.7 | 17,904.3–20,009.4 | 597/601 | 23/68 | 1282.5 | 13.7 |
| burst400k | 18,182.2 | 17,317.7–19,046.9 | 569/580 | 22/68 | 1359.9 | 14.6 |
| biome400k | 18,108.4 | 17,217.6–19,024.7 | 567/579 | 20/68 | 1332.9 | 14.3 |
| oscar200k | 9,814.6 | 9,366.1–10,266.2 | 569/580 | 0/68 | 1359.9 | 14.0 |

## Equal-payload comparisons

All changes below are compared with `burst400k` (400,000 turns and a 50-second cooldown).

| Variant | Paired score gain | 95% paired CI |
|---|---:|---|
| fast400k | +756.4 | -400.6 to +1,945.0 |
| lowpop400k | +1,482.8 | +178.3 to +2,825.0 |
| birth400k | +1,358.6 | +242.8 to +2,530.4 |
| population400k | +923.1 | -488.4 to +2,344.9 |
| biome400k | -73.8 | -1,044.8 to +909.2 |

## Interpretation

The best local score is `fast1m`: 1,000,000 stationary turns, zero cooldown, otherwise Oscar’s existing observation-only contact predictor and population policy. Its 597 transfers are exactly the same count as fast400k. The extra 600,000 turns add 3,000 score per transfer; this explains the entire paired score difference. Thus the major gain is payload scaling, not better survival.

The best 400k mean is lowpop400k, which allows transfers with four live agents (min_free=2) instead of eight. birth400k lowers reproduction energy reserves. Neither guarantees 20k. The pilot has only 68 maps; confirm the chosen policy on fresh maps before promoting it.

One-million-action response size and hosted viability remain unverified. Oscar documented a clean hosted 400k no-op response, not a successful hosted 1M harvest. All experiments here were local simulations; no hosted validation/evaluation was submitted. Simulation runtime excludes HTTP serialization and organizer-side processing.

Source baseline: 43e3d52. Sweep implementation: a42d067. Raw results and exact variant manifests are in pilot/. Four pods created for this pilot were deleted after all 544 unique rows were collected and verified. SSH PC jobs completed.

# Frozen 1000-map evaluation

Twelve frozen configurations, identical fresh world seeds 10001–11000. 95% percentile bootstrap intervals use 10,000 shared map resamples. No tuning on this panel.

| Model | Mean score | 95% CI | Policy µs/tick | Whole loop CPU µs/tick | Whole loop wall µs/tick |
|---|---:|---|---:|---:|---:|
| scheduled_breeding | 1560.0 | 1534.1–1585.3 | 558.5 | 973.3 | 996.2 |
| trait_succession | 1537.7 | 1513.4–1561.9 | 464.0 | 799.7 | 818.8 |
| population_harvest | 1529.3 | 1505.1–1553.6 | 502.3 | 842.1 | 866.2 |
| population_explore | 1519.5 | 1495.2–1543.2 | 503.8 | 882.8 | 901.9 |
| distributed_orchards | 1517.8 | 1494.1–1540.8 | 489.6 | 853.2 | 874.8 |
| mobile_harvest | 1517.5 | 1494.3–1540.4 | 478.5 | 856.7 | 881.5 |
| clustered_orchards | 1509.9 | 1484.8–1535.2 | 456.0 | 783.7 | 807.3 |
| expanded_population | 1505.1 | 1481.1–1529.4 | 491.8 | 843.3 | 867.5 |
| economy_gaze | 1503.9 | 1481.0–1527.3 | 507.9 | 877.1 | 899.9 |
| ripeness_reserve | 1500.4 | 1476.7–1523.8 | 492.8 | 834.0 | 855.3 |
| previous_winner | 1491.7 | 1467.4–1515.8 | 503.7 | 859.6 | 886.4 |
| original_baseline | 1466.9 | 1441.0–1493.3 | 340.2 | 581.1 | 598.2 |

## Paired differences versus previous_winner

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| scheduled_breeding | +68.3 | +41.4–+95.2 |
| trait_succession | +46.0 | +20.3–+71.7 |
| population_harvest | +37.6 | +12.0–+63.5 |
| population_explore | +27.7 | +3.6–+52.8 |
| distributed_orchards | +26.1 | +1.2–+50.0 |
| mobile_harvest | +25.8 | +1.2–+51.3 |
| clustered_orchards | +18.2 | -8.2–+45.0 |
| expanded_population | +13.4 | -11.8–+39.0 |
| economy_gaze | +12.1 | -13.6–+37.6 |
| ripeness_reserve | +8.7 | -17.7–+35.4 |
| original_baseline | -24.8 | -51.0–+1.5 |

## Paired differences versus original_baseline

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| scheduled_breeding | +93.0 | +65.3–+120.8 |
| trait_succession | +70.7 | +44.5–+96.6 |
| population_harvest | +62.4 | +35.2–+89.5 |
| population_explore | +52.5 | +25.6–+79.2 |
| distributed_orchards | +50.9 | +25.8–+76.1 |
| mobile_harvest | +50.6 | +24.6–+76.6 |
| clustered_orchards | +43.0 | +14.5–+71.6 |
| expanded_population | +38.2 | +11.4–+65.6 |
| economy_gaze | +36.9 | +11.6–+62.8 |
| ripeness_reserve | +33.5 | +7.1–+60.0 |
| previous_winner | +24.8 | -1.5–+51.0 |

A tick means one simulation update for the whole population, not one agent action. Timing is total measured time divided by total ticks across 1000 games; initialization is excluded. Policy timing uses native elapsed clocks and includes scheduling delays; loop CPU uses per-process CPU time. All models share each pod and its 100-map shard, in randomized job order. Measurements reflect 32 concurrent workers, hardware differences and profiling overhead, not isolated production latency. Interface and engine breakdowns are in summary.json.

Intervals are pointwise, not corrected for multiple comparisons. A top test rank alone does not establish superiority. All 66 pairwise comparisons are in paired.json.

Attributed active compute: $1.35, excluding setup/storage/idle pod uptime.

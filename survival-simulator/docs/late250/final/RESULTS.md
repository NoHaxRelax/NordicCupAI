# Frozen 1000-map evaluation

22 frozen configurations, identical fresh world seeds 12001–13000. 95% percentile bootstrap intervals use 10,000 shared map resamples. No tuning on this panel.

| Model | Mean score | 95% CI | Policy µs/tick | Whole loop CPU µs/tick | Whole loop wall µs/tick |
|---|---:|---|---:|---:|---:|
| local_food | 1598.2 | 1573.3–1623.1 | 560.3 | 1017.3 | 1028.3 |
| cluster_food | 1562.5 | 1537.7–1587.7 | 556.2 | 997.2 | 1015.7 |
| economy_mix | 1559.8 | 1535.1–1584.4 | 610.0 | 1038.0 | 1057.0 |
| relocate | 1558.7 | 1533.6–1584.0 | 568.8 | 1000.9 | 1022.1 |
| elite_heirs | 1555.6 | 1532.6–1578.6 | 435.7 | 760.3 | 769.2 |
| spread_food | 1549.1 | 1524.8–1573.1 | 559.0 | 990.5 | 1000.2 |
| late_reserves | 1548.8 | 1522.3–1575.5 | 549.6 | 962.0 | 975.5 |
| wide_food | 1548.7 | 1523.6–1573.5 | 538.1 | 958.0 | 970.4 |
| scan | 1548.4 | 1523.8–1573.0 | 533.7 | 957.9 | 968.5 |
| nursery | 1547.0 | 1522.5–1571.9 | 550.4 | 963.1 | 974.2 |
| birth_pacing | 1543.5 | 1518.8–1568.0 | 547.1 | 957.8 | 972.0 |
| late_gaze | 1537.9 | 1513.9–1562.3 | 568.1 | 1001.5 | 1015.5 |
| conserve_explore | 1531.4 | 1506.9–1556.6 | 568.2 | 995.1 | 1011.1 |
| early_heirs | 1529.3 | 1504.7–1554.3 | 561.7 | 983.7 | 997.5 |
| old_priority | 1528.6 | 1504.2–1553.1 | 562.8 | 985.0 | 997.5 |
| retirement | 1521.7 | 1496.6–1547.2 | 564.3 | 983.9 | 999.6 |
| breed_feeding | 1515.0 | 1488.3–1540.9 | 548.8 | 964.7 | 977.6 |
| scheduled_breeding_baseline | 1512.5 | 1487.0–1538.7 | 567.7 | 991.7 | 1005.1 |
| ripe_food | 1494.2 | 1468.5–1519.8 | 561.9 | 983.7 | 999.3 |
| original_baseline | 1452.6 | 1429.6–1475.7 | 342.3 | 588.5 | 598.3 |
| tree_capacity | 1450.9 | 1426.4–1475.9 | 288.8 | 509.7 | 517.4 |
| small_population | 1405.5 | 1380.7–1430.1 | 254.1 | 445.5 | 453.2 |

## Paired differences versus scheduled_breeding_baseline

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| local_food | +85.7 | +59.1–+112.4 |
| cluster_food | +50.0 | +22.5–+77.4 |
| economy_mix | +47.3 | +19.9–+74.5 |
| relocate | +46.2 | +20.3–+72.5 |
| elite_heirs | +43.1 | +16.6–+69.2 |
| spread_food | +36.6 | +8.9–+64.6 |
| late_reserves | +36.3 | +12.9–+59.1 |
| wide_food | +36.2 | +8.9–+63.7 |
| scan | +35.9 | +9.1–+62.7 |
| nursery | +34.5 | +7.3–+61.5 |
| birth_pacing | +31.1 | +4.7–+57.1 |
| late_gaze | +25.4 | -0.2–+51.9 |
| conserve_explore | +18.9 | -7.7–+45.6 |
| early_heirs | +16.8 | -10.7–+44.9 |
| old_priority | +16.1 | -10.1–+42.5 |
| retirement | +9.2 | -17.5–+36.0 |
| breed_feeding | +2.5 | -26.2–+29.7 |
| ripe_food | -18.3 | -45.3–+8.9 |
| original_baseline | -59.9 | -85.2–-33.9 |
| tree_capacity | -61.6 | -90.1–-33.2 |
| small_population | -107.0 | -135.6–-78.3 |

## Paired differences versus original_baseline

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| local_food | +145.5 | +118.5–+172.3 |
| cluster_food | +109.9 | +83.2–+135.3 |
| economy_mix | +107.1 | +80.0–+133.9 |
| relocate | +106.0 | +79.5–+132.2 |
| elite_heirs | +102.9 | +77.1–+128.0 |
| spread_food | +96.4 | +69.2–+123.2 |
| late_reserves | +96.2 | +69.3–+122.5 |
| wide_food | +96.0 | +70.3–+121.2 |
| scan | +95.7 | +69.5–+121.6 |
| nursery | +94.3 | +68.4–+119.9 |
| birth_pacing | +90.9 | +65.1–+117.1 |
| late_gaze | +85.3 | +59.8–+110.8 |
| conserve_explore | +78.8 | +52.5–+105.1 |
| early_heirs | +76.7 | +49.9–+102.8 |
| old_priority | +75.9 | +50.6–+101.1 |
| retirement | +69.1 | +42.2–+95.8 |
| breed_feeding | +62.3 | +34.7–+88.7 |
| scheduled_breeding_baseline | +59.9 | +33.9–+85.2 |
| ripe_food | +41.5 | +14.7–+68.4 |
| tree_capacity | -1.8 | -27.3–+23.5 |
| small_population | -47.2 | -74.5–-20.0 |

A tick means one simulation update for the whole population, not one agent action. Timing is total measured time divided by total ticks across 1000 games; initialization is excluded. Policy timing uses native elapsed clocks and includes scheduling delays; loop CPU uses per-process CPU time. All models share each pod and its 100-map shard, in randomized job order. Measurements reflect 32 concurrent workers, hardware differences and profiling overhead, not isolated production latency. Interface and engine breakdowns are in summary.json.

Intervals are pointwise, not corrected for multiple comparisons. A top test rank alone does not establish superiority. All 231 pairwise comparisons are in paired.json.

Attributed active compute: $2.66, excluding setup/storage/idle pod uptime.

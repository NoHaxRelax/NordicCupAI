# Frozen 1000-map evaluation

23 frozen configurations, identical fresh world seeds 15001–16000. 95% percentile bootstrap intervals use 10,000 shared map resamples. No tuning on this panel.

| Model | Mean score | 95% CI | Policy µs/tick | Whole loop CPU µs/tick | Whole loop wall µs/tick |
|---|---:|---|---:|---:|---:|
| expanded_local_food | 1706.0 | 1682.1–1729.6 | 522.1 | 964.3 | 977.4 |
| risk_one_tick | 1673.5 | 1650.3–1696.0 | 452.3 | 799.8 | 812.5 |
| late_food_radius | 1662.5 | 1637.2–1688.0 | 544.4 | 986.9 | 997.0 |
| fast_travel | 1659.2 | 1635.3–1683.0 | 535.4 | 981.2 | 990.0 |
| late_breed_reserve | 1656.1 | 1630.4–1681.1 | 498.9 | 903.9 | 914.2 |
| trait_selection | 1650.8 | 1624.9–1676.8 | 559.5 | 1020.0 | 1030.9 |
| local_food_baseline | 1644.9 | 1619.7–1669.7 | 563.6 | 1018.9 | 1032.5 |
| late_risk | 1644.5 | 1618.2–1669.9 | 594.0 | 1037.6 | 1053.6 |
| cluster_local_food | 1641.9 | 1618.2–1665.1 | 548.0 | 1009.4 | 1021.8 |
| late_small_colony | 1639.6 | 1614.7–1665.2 | 529.6 | 956.1 | 969.2 |
| idle_pulse | 1636.5 | 1610.7–1661.5 | 547.1 | 992.3 | 1004.9 |
| small_colony_rescue | 1601.6 | 1576.5–1626.9 | 549.2 | 999.4 | 1009.4 |
| sparse_pulse | 1576.2 | 1551.1–1601.4 | 563.9 | 1014.5 | 1023.4 |
| late_retirement | 1573.7 | 1549.2–1597.8 | 467.6 | 844.5 | 854.0 |
| scheduled_breeding_baseline | 1550.7 | 1524.8–1575.9 | 569.5 | 996.4 | 1008.9 |
| late_pulse | 1550.4 | 1527.2–1573.1 | 495.9 | 886.8 | 895.2 |
| risk_two_ticks | 1543.8 | 1515.6–1570.9 | 609.5 | 1057.2 | 1070.9 |
| pulse_scan | 1487.8 | 1461.0–1515.1 | 610.2 | 1009.3 | 1021.7 |
| fast_scan_risk | 1483.6 | 1455.6–1511.6 | 663.9 | 1074.2 | 1088.3 |
| original_baseline | 1483.3 | 1458.7–1508.7 | 334.0 | 578.5 | 585.2 |
| wall_risk | 1447.5 | 1425.8–1467.9 | 543.4 | 903.4 | 912.2 |
| behind_two_tick | 1359.0 | 1337.9–1379.6 | 814.5 | 1258.6 | 1268.9 |
| behind_predator | 1337.7 | 1316.9–1358.6 | 666.9 | 1104.2 | 1115.7 |

## Paired differences versus local_food_baseline

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| expanded_local_food | +61.1 | +34.7–+86.8 |
| risk_one_tick | +28.6 | +2.5–+54.2 |
| late_food_radius | +17.6 | -7.9–+43.0 |
| fast_travel | +14.3 | -11.0–+39.7 |
| late_breed_reserve | +11.2 | -13.6–+37.2 |
| trait_selection | +5.9 | -20.9–+32.6 |
| late_risk | -0.4 | -27.6–+26.0 |
| cluster_local_food | -3.0 | -28.4–+22.9 |
| late_small_colony | -5.3 | -31.5–+21.3 |
| idle_pulse | -8.4 | -34.2–+17.5 |
| small_colony_rescue | -43.3 | -69.3–-16.3 |
| sparse_pulse | -68.7 | -94.9–-42.8 |
| late_retirement | -71.2 | -98.5–-44.0 |
| scheduled_breeding_baseline | -94.1 | -120.2–-68.6 |
| late_pulse | -94.5 | -119.6–-70.0 |
| risk_two_ticks | -101.1 | -129.7–-73.7 |
| pulse_scan | -157.1 | -186.0–-128.3 |
| fast_scan_risk | -161.3 | -191.6–-131.6 |
| original_baseline | -161.6 | -188.5–-134.3 |
| wall_risk | -197.4 | -223.2–-171.9 |
| behind_two_tick | -285.9 | -311.6–-260.0 |
| behind_predator | -307.2 | -333.0–-281.0 |

## Paired differences versus original_baseline

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| expanded_local_food | +222.7 | +195.5–+249.5 |
| risk_one_tick | +190.1 | +162.0–+217.6 |
| late_food_radius | +179.2 | +150.6–+206.9 |
| fast_travel | +175.9 | +149.6–+202.7 |
| late_breed_reserve | +172.7 | +144.8–+200.5 |
| trait_selection | +167.4 | +139.0–+195.6 |
| local_food_baseline | +161.6 | +134.3–+188.5 |
| late_risk | +161.2 | +131.8–+190.3 |
| cluster_local_food | +158.6 | +131.0–+184.8 |
| late_small_colony | +156.3 | +128.9–+183.0 |
| idle_pulse | +153.1 | +124.9–+180.5 |
| small_colony_rescue | +118.3 | +89.4–+146.8 |
| sparse_pulse | +92.8 | +64.9–+121.2 |
| late_retirement | +90.4 | +61.8–+118.4 |
| scheduled_breeding_baseline | +67.4 | +38.8–+95.0 |
| late_pulse | +67.0 | +39.9–+94.1 |
| risk_two_ticks | +60.4 | +29.3–+91.0 |
| pulse_scan | +4.5 | -25.4–+34.4 |
| fast_scan_risk | +0.2 | -30.9–+32.0 |
| wall_risk | -35.8 | -63.1–-9.0 |
| behind_two_tick | -124.4 | -151.5–-97.4 |
| behind_predator | -145.6 | -173.8–-118.2 |

A tick means one simulation update for the whole population, not one agent action. Timing is total measured time divided by total ticks across 1000 games; initialization is excluded. Policy timing uses native elapsed clocks and includes scheduling delays; loop CPU uses per-process CPU time. All models share each pod and its 100-map shard, in randomized job order. Measurements reflect 32 concurrent workers, hardware differences and profiling overhead, not isolated production latency. Interface and engine breakdowns are in summary.json.

Intervals are pointwise, not corrected for multiple comparisons. A top test rank alone does not establish superiority. All 253 pairwise comparisons are in paired.json.

Attributed active compute: $3.03, excluding setup/storage/idle pod uptime.

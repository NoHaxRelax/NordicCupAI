# Frozen 1000-map evaluation

24 frozen configurations, identical fresh world seeds 19001–20000. 95% percentile bootstrap intervals use 10,000 shared map resamples. No tuning on this panel.

| Model | Mean score | 95% CI | Policy µs/tick | Whole loop CPU µs/tick | Whole loop wall µs/tick |
|---|---:|---|---:|---:|---:|
| expanded_food_baseline | 1676.4 | 1652.5–1701.0 | 485.2 | 888.7 | 895.5 |
| congestion_pricing | 1648.6 | 1622.6–1675.2 | 8714.1 | 9159.5 | 9191.7 |
| net_energy_fruit | 1641.7 | 1615.3–1668.1 | 8134.3 | 8560.1 | 8591.5 |
| cooperative_harvest | 1640.9 | 1616.8–1665.6 | 8826.2 | 9284.9 | 9313.4 |
| late_auction | 1639.4 | 1612.9–1666.2 | 7594.8 | 8009.0 | 8035.9 |
| shared_food_control | 1637.4 | 1611.6–1663.1 | 8280.8 | 8719.5 | 8740.3 |
| leave_empty_area | 1612.4 | 1586.5–1638.0 | 5442.6 | 5786.8 | 5807.1 |
| local_food_baseline | 1608.6 | 1582.0–1634.7 | 526.5 | 946.3 | 950.9 |
| mobile_colony | 1605.4 | 1579.2–1630.7 | 6060.7 | 6418.8 | 6448.0 |
| fruit_auction | 1602.2 | 1571.9–1631.1 | 7445.6 | 7845.4 | 7875.0 |
| renewal_sites | 1597.5 | 1572.1–1623.0 | 5041.5 | 5369.9 | 5392.4 |
| late_food_capacity | 1596.9 | 1570.3–1623.1 | 8014.9 | 8440.1 | 8471.4 |
| late_positioning | 1584.0 | 1560.9–1607.3 | 8529.9 | 8974.9 | 9002.9 |
| shared_one_tick | 1579.7 | 1551.6–1606.8 | 8704.3 | 9146.7 | 9181.2 |
| budget_breeding | 1578.4 | 1551.8–1604.9 | 7195.7 | 7593.0 | 7621.9 |
| safe_food | 1574.9 | 1547.6–1601.5 | 5156.9 | 5497.8 | 5519.4 |
| safe_harvest | 1574.3 | 1546.2–1601.8 | 7486.6 | 7904.4 | 7928.2 |
| shared_alerts | 1570.0 | 1544.3–1595.2 | 5282.1 | 5620.5 | 5639.5 |
| scarcity_economy | 1556.8 | 1527.9–1585.4 | 7183.9 | 7590.8 | 7608.4 |
| scheduled_breeding_baseline | 1556.3 | 1530.4–1582.2 | 522.7 | 910.3 | 915.8 |
| aging_efficiency | 1550.1 | 1522.8–1576.8 | 8444.8 | 8886.5 | 8923.2 |
| food_capacity | 1548.0 | 1521.4–1574.8 | 5167.4 | 5491.0 | 5508.9 |
| rock_aware_posts | 1464.2 | 1439.3–1489.2 | 8022.6 | 8418.3 | 8440.2 |
| fruit_centroid_posts | 1438.3 | 1412.0–1464.2 | 5707.4 | 6013.9 | 6032.5 |

## Paired differences versus expanded_food_baseline

| Model | Mean difference | 95% paired CI |
|---|---:|---|
| congestion_pricing | -27.8 | -54.2–-0.7 |
| net_energy_fruit | -34.6 | -63.4–-6.5 |
| cooperative_harvest | -35.5 | -63.2–-8.1 |
| late_auction | -37.0 | -64.1–-9.8 |
| shared_food_control | -38.9 | -67.1–-11.0 |
| leave_empty_area | -64.0 | -92.7–-35.7 |
| local_food_baseline | -67.7 | -94.8–-41.1 |
| mobile_colony | -71.0 | -99.7–-42.5 |
| fruit_auction | -74.2 | -104.7–-44.4 |
| renewal_sites | -78.9 | -107.7–-50.5 |
| late_food_capacity | -79.4 | -107.3–-51.1 |
| late_positioning | -92.4 | -118.8–-66.1 |
| shared_one_tick | -96.7 | -126.9–-67.8 |
| budget_breeding | -97.9 | -126.5–-70.4 |
| safe_food | -101.5 | -129.8–-73.5 |
| safe_harvest | -102.1 | -131.4–-73.1 |
| shared_alerts | -106.3 | -134.5–-77.8 |
| scarcity_economy | -119.5 | -149.3–-89.9 |
| scheduled_breeding_baseline | -120.1 | -146.8–-93.3 |
| aging_efficiency | -126.2 | -155.7–-97.5 |
| food_capacity | -128.3 | -157.2–-99.9 |
| rock_aware_posts | -212.2 | -237.6–-186.1 |
| fruit_centroid_posts | -238.1 | -265.9–-210.7 |

A tick means one simulation update for the whole population, not one agent action. Timing is total measured time divided by total ticks across 1000 games; initialization is excluded. Policy timing uses native elapsed clocks and includes scheduling delays; loop CPU uses per-process CPU time. All models share each pod and its 100-map shard, in randomized job order. Measurements reflect 32 concurrent workers, hardware differences and profiling overhead, not isolated production latency. Interface and engine breakdowns are in summary.json.

Intervals are pointwise, not corrected for multiple comparisons. A top test rank alone does not establish superiority. All 276 pairwise comparisons are in paired.json.

Attributed active compute: $21.73, excluding setup/storage/idle pod uptime.

# Completion and reproducibility

All ten jobs completed on 2026-09-19, launched approximately 14:18:26 UTC.
The slowest took 2571.27 seconds (**42.9 minutes**). Summed job time was
22,955.18 pod-seconds, equivalent to **$6.12 attributed active compute** at
$0.96 per pod-hour. This excludes setup, storage and the existing team's idle
pod uptime; it is not an invoiced billing total. CPU pods were left running.

Verified **40,800 games**: 600 trials × 64 training maps, ten × 200 test maps,
and two × 200 control maps. Every training iteration has exactly seeds
8001–8064; every final evaluation has exactly seeds 9001–9200. Winners match
the maximum training mean. All recorded source hashes match the local frozen
source; configurations, raw games and native build metadata are retained.

Source commit: db63edc. Dispatch mapping: dispatch.json, committed at 6d99f57.
All simulation and policy ticks ran in native C++; Python scheduled and fit GPs.
No official competition validation was attempted.

The best test mean was clustered_orchards at 1551.4, compared with the previous
population winner at 1506.3 and original baseline at 1509.9 on these same maps.
Its paired gain over the previous winner was +45.1 with an unadjusted 95%
bootstrap interval of [-11.3, 101.8]. None of the ten paired intervals excludes
zero; there is no established improvement over the previous winner in this batch.
Do not directly compare these means to the earlier, different 100-map test set.

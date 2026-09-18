# Single-active leader reserves fitted result

`multi_site.leader_reserves:LeaderReserves` passed both requested fitted cases
under frozen causal scorer v4. No fresh maps were run.

| Map / fixture | v4 delivery | Stable tail | Births | Guide-lineage deaths | Survivors at 300s |
| --- | ---: | ---: | ---: | ---: | --- |
| 10138 / 20138 | 71.4s | 228.7s | 8 | 6 | 4, 6, 9 |
| 10224 / 20224 | 138.6s | 161.5s | 8 | 3 | 2, 5, 6, 7, 8, 9 |

Both 300-second recordings contain and verify 3,001/3,001 native frames. In
both cases guide 1 survived the early encounter that killed the prior single
guide, remained the causal handoff guide, and died at the bait handoff. The
predator then stayed physically contained through the horizon. Reserve election
did change leaders while routing, but only the elected controller executed its
v30 wide-route planner; inactive reserves used stateless DTO-relative evade or
hold actions. Their planner poses therefore were not advanced for replaced
actions.

The policy spends the full eight-birth budget immediately and retains several
survivors. This is fitted evidence only. It does not show fresh-map reliability,
low-cost behavior, post-handoff guide survival, or successful release. Direct
DTO-relative reserve evasion is not obstacle planned, and repeated elections
can still interrupt an individual route. These limitations should be addressed
before any promotion or fresh batch.

Receipts and replays use stems `ded791f0` and `fbf64ae9` under
`results/simple_chase`; v4 sidecars are under
`results/multi_site/leader_reserves`.

# Frozen 9059 benchmark baseline

The historical `benchmark` command preserves the exact policy and simulator
source used for the original native seed-0 replay. The normal `trapping`,
`serve` and `tune` commands now use the newer modular integration; see
[the branch review](policy_branch_integration.md).

Every one of the 39 source files in `docs/entrapment_native_seed0/manifest.json` was restored from the run's source archive and checked against its SHA-256 hash. This includes Nikolaj's observation-only explorer/map estimator, Oscar's orchard policy, **our** trap detector, and the initial colony coordinator.

The later experimental boundary-reconstruction fix and emergency reproduction fallback are **not included** in this trapping baseline. Its limitations are retained for comparison. The separate no-predator Orchard runner uses its audited observed-bounds adapter; see the [current quickstart](../README.md).

## Behavior

- Normal native `SimulationCore`, default population and natural predator spawns.
- No infinite energy, forced encounters, known map, hidden predator identities or hidden targeting inputs.
- Nikolaj discovers the map; our detector alone accepts traps (depth 5, gap 10.1–19.1, overlap >=10.3, opposite-end bait access).
- Oscar provides gathering/population decisions after a trap is found.
- Old agents are preferred for bait and guides. Replacements overlap before the preceding bait dies when sufficient candidates exist.
- This baseline suppresses reproduction after age 55 and has no emergency exception. Preserve that behavior for the benchmark.

## Seed-0 reference result

| Metric | Value |
|---|---:|
| Trap discovered / first bait | 26.5 / 29.0 seconds |
| Guide assignments / delivery arrivals | 32 / 15 |
| Overlapping bait replacements | 23 |
| Predators simultaneously near bait for at least 30 seconds | 11 |
| Colony extinction | 1,073.6 seconds |
| Bait gap after first arrival | 8.0 seconds, at the end |
| Saved frames | 10,737 |

Delivery arrival and near-bait retention are proxies, not permanent-capture success rates. Predator associations are inferred from observations and can be ambiguous. Observed geometry can be incomplete. Guides snapshot their known edges when assigned. Full-game survival is not established.

The seed-0 post-run audit uses separate evaluator geometry and does not feed hidden information into the policy. The source manifest, summary, and audit are retained in `entrapment_native_seed0/`. Full recordings are local artifacts and are not required to run the policy.

## Local reference run

```bash
python survival-simulator/scripts/entrapment_game.py --seed 0 --seconds 3000 --out /tmp/entrapment-reference
python survival-simulator/run.py view /tmp/entrapment-reference --port 9059
```

The [benchmark protocol](entrapment_9059_benchmark_protocol.md) is pinned to this source baseline. All maps count, including maps where no trap is discovered or geometrically valid.

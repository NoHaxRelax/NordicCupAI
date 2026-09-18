# Frozen 9059 benchmark baseline

This branch preserves the exact policy and simulator source used for the native seed-0 replay at `http://localhost:9059`.

Every one of the 39 source files in `docs/entrapment_native_seed0/manifest.json` was restored from the run's source archive and checked against its SHA-256 hash. This includes Nikolaj's observation-only explorer/map estimator, Oscar's orchard policy, **our** trap detector, and the initial colony coordinator.

The later boundary-reconstruction fix and emergency reproduction fallback are deliberately **not included**. The newer development workspace remains separate. This is an immutable experimental baseline, including its limitations, not an improved implementation disguised as the earlier version.

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

The seed-0 post-run audit uses separate evaluator geometry and does not feed hidden information into the policy. The original replay and source archive are kept in the original development workspace under `survival-simulator/logs/entrapment_game/native-seed-0-v1/`.

## Local reference run

```bash
python survival-simulator/scripts/entrapment_game.py --seed 0 --seconds 3000 --out /tmp/entrapment-reference
python survival-simulator/scripts/entrapment_viewer.py /tmp/entrapment-reference --port 9059
```

The 1,000-game benchmark protocol and results will be added separately, pinned to this source baseline. All maps will count, including maps where no trap is discovered or geometrically valid.

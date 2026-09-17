# Investigator 13: observability and contract audit

17 September 2026. Astra reviewer. Personal hackathon research using the locally vendored simulator at `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`.

## Reproduce

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/13_astra_observability.py
```

[Executable](13_astra_observability.py), [evidence JSON](../../results/mechanics_hunt/13_astra_observability.json). The JSON includes SHA256 hashes of vendor Python files, checked unchanged before/after execution. Upstream identity follows `survival/SOURCE.md`; this test does not independently download or verify Git blobs.

## Ranked usable conclusions

1. **Detect old-age drain from ordinary consecutive payloads.** An agent's own `age`, `energy`, `speed`, `sprint_speed`, `max_energy`, and biome are exposed. Its random `max_age` threshold is hidden. After an idle tick, the normal loss is 0.1; old-age loss is `0.1 + 0.01 * new_age`. At age70 this is about 8 energy/second rather than 1. This allows an ordinary controller to detect onset and prioritize replacement or food. It cannot predict the threshold beforehand.
2. **Calculate reproduction eligibility after movement and turning.** All required cost inputs are exposed. The check is strictly `energy - movement_cost - turn_cost > 100`; fruit gathered later in the tick cannot fund that birth. Energy100.5 supports an idle birth but not a walk10 birth. Energy101 fails after walk10 plus a half-turn; energy101.01 succeeds, but that parent then dies during passive drain. Count offspring IDs, not net population change, when verifying marginal births. Whether deliberate parent replacement raises game score remains untested here.
3. **Treat fruit ripeness and predator sleep as uncertain unless history establishes them.** Fresh and mature fruit at the same position produce identical complete agent payloads, but eating differs by40 energy. A resting predator at energy0 and one at102 likewise produce identical payloads; in the arranged contact-range test, waiting survives the first and dies in the second. Fruit maturation and sleep mechanics were already known; these paired tests explicitly establish the limits of single-observation inference.

## Generated-map evidence

The detector receives only the previous and current ordinary agent payloads. Every action is one finite idle request per living observed agent. Complete default `SimulationCore` maps, starting energies, obstacles, fruit, trees, predators and spawning are retained. True state is read only afterward to validate classifications. The stop condition is extinction or110 seconds. No artificial energy, rearranged positions, disabled spawning or replacement policy is used in these runs.

| Seed | Duration | Young classifications | Old classifications | Skipped/unknown | Wrong classifications |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 104.5 s | 4,150 | 383 | 4 | 0 |
| 29 | 102.3 s | 4,207 | 350 | 3 | 0 |
| 47 | 92.8 s | 3,402 | 364 | 1 | 0 |

All 12,856 classified surviving-agent updates agreed with diagnostic truth. Fourteen agents had observable aging onsets, with the reported age less than 0.1 seconds beyond the hidden threshold. Eight updates with unchanged age were rejected, protecting the inference from the known death/list-iteration skip. These idle trials test observability, not a competitive policy or a score advantage. Sample limits and local platform behavior still apply.

## Negative controls and rejected hypotheses

- **An exact hidden aging threshold is visible before onset:** rejected by identical payloads at age70 with thresholds60 and120. Their next-tick energy differs by0.701.
- **Energy changes always identify aging:** rejected when an old agent collects fruit. Separate fixtures at energy150 and at its500 cap are marked `confounded`, not forced into young/old. Require an uncontaminated transition or account for known intake. The generated idle runs happened to contain no such surviving-transition intake confound; fixture controls cover it.
- **Every global tick updates every surviving agent:** rejected by eight generated-map skips. Checking the agent's age delta prevents false confidence from a cached/skipped update.
- **A fruit observation tells ripeness:** rejected by payload equality and a40-energy collection gap. The fixtures explicitly set fruit growth; they do not demonstrate tracking its natural birth. Continuous observation of a new fruit may support a timer, but lack of IDs, occlusion, collection-before-return and association ambiguity must be handled separately.
- **A nearby stationary predator is safely asleep:** rejected as a single-frame decision. The energy0/102 pair has identical visible geometry and opposite survival outcomes. These states are explicitly arranged. The test does not prove that every multi-frame history is ambiguous or that carefully inferred sleep is impossible.
- **A successful birth necessarily increases population:** rejected by the energy101.01 movement/turn fixture. The parent dies that tick and the child remains.

## Compliance, provenance and limits

All submitted test requests obey one action per observed agent. No newborn action, duplicate-ID sequence or malformed input is used. Policy-facing logic reads no true map coordinates, hidden predator state, fruit age, or hidden max-age threshold. Fixture construction and diagnostic assertions do read/set those fields to establish ground truth, and are not presented as achievable policy setup.

The independent tests extend the earlier fruit/sleep/aging report; those underlying rules are not new discoveries. The strongest new practical output is the ordinary-payload aging detector and its negative controls. It can combine with an existing food-aware capped population controller, while reproduction eligibility can prevent silent failed births. Actual score benefit from that combination requires a separate policy benchmark; this reviewer does not claim one.

Source anchors: `environment.py:575` own-state payload, `environment.py:497` movement costs, `environment.py:564` turning costs, `environment.py:602` action/birth order, `environment.py:635` aging/collection, `environment.py:675` predator wake-up; `creature.py:130` observation fields. All anchors are under `survival/vendor/survival-simulator/src/elements/`.

No remote interaction or engine edits occurred. Water baiting, wall acquisition, dedicated sprint decoys, predator banishment and replay-viewer work were left to their assigned investigations.

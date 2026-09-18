# Mechanics hunt 01: action order, duplicate IDs, and newborn timing

Tested 17 September 2026 against the unmodified vendored simulator at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. This is bounded local research, not a hosted validation or evaluation. It is not a claim of exhaustive proof.

Run the complete asserted probe from the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/01_actions.py
```

The machine-readable evidence is written to [`survival/results/mechanics_hunt/01_actions.json`](../../results/mechanics_hunt/01_actions.json). Use `--skip-generated-maps` for the fast fixture-only suite.

## Ranked findings

### 1. Compliant: budget action cost, birth cost, and passive drain in that order

An action moves first, turns second, checks `energy > 100` for birth third, and only then runs the world's passive/old-age/death and food phases. This yields two useful rules for a young agent in an ordinary biome:

```text
birth succeeds iff energy - movement_cost - turn_cost > 100
parent survives iff energy - movement_cost - turn_cost - 100 - passive_drain > 0
```

The controller can apply these rules from ordinary status fields and its own requested action. The random private `max_age` means an older agent needs an additional conservative margin once age exceeds 60.

Minimal evidence:

- At energy `100.40`, an idle birth succeeded and the parent survived at `0.30`.
- At the same energy, either walking 10 (`0.50`) or turning π (`0.50`) cancelled the birth because the check occurred after that cost.
- At `100.05`, an idle birth succeeded but passive drain killed the parent; the child survived, so population stayed at one with living ID 1.
- At `100.11`, both parent and child survived.

Practical magnitude: the difference between replacement and population growth is only the tick's drain margin. A low-energy emergency birth can preserve the lineage, but a simultaneous walk or turn can silently prevent it. This fully respects one action per observed agent.

### 2. Compliant: food cannot fund this tick's birth, and a dying mover cannot eat

Fruit collection happens after birth and after the energy death check.

- A 99-energy parent placed on a ripe 60-energy fruit requested birth. No child appeared on tick one; the fruit then raised it to `158.9`. The same request on tick two produced the child.
- An agent with `0.59` energy walked 10 onto a ripe fruit. The `0.50` move plus `0.10` passive drain killed it before collection, and the fruit remained.
- The `0.61` negative control survived at `0.01`, collected the fruit, and ended at `60.01`.

The fruit positions and ripeness are controlled fixture advantages. The policy rule itself is observation-compatible: reserve action plus passive energy before committing to a food endpoint, and never assume visible fruit will retroactively finance reproduction. Fruit energy/ripeness and the elder threshold are hidden, so a real policy needs margin.

### 3. Known rule-breaking gap, extended: duplicate IDs make per-tick speed effectively action-count limited

The previously documented duplicate-ID issue reproduces exactly. The action loop performs every entry before one world update, with no uniqueness or count check.

| Request | Distance in one 0.1 s tick | Energy spent including passive |
| --- | ---: | ---: |
| One walk 10 | 10 | 0.60 |
| One oversized 1000 | 20 | 5.60 |
| Ten walks 10 | 100 | 5.10 |
| One hundred walks 10 | 1000 | 50.10 |

The single oversized request is correctly capped at sprint speed. Repeating walk-speed actions bypasses the per-action cap at the ordinary walking energy rate per unit.

On default generated maps with ordinary starting energy, obstacles, terrain, fruit, and trees, ten duplicates produced mean one-tick displacement `67.79` versus `7.20` for seed 1 and `68.00` versus `6.80` for seed 42. Per-agent ratios were 8.73–10×; terrain and collision deflection explain the shortfall from a perfect 10×. Positions were read only for diagnostics.

This is not a compliant strategy. The upstream instructions require one action for each observed agent. Local DTO acceptance does not establish hosted acceptance or competition permission.

### 4. Known rule-breaking newborn action, extended with three hard limits

A birth immediately inserts the child into `agents_dict`, so a later entry for its predictable sequential ID executes in the same request list. Reversing the order is a negative control: the future-ID action is skipped, then the child is born idle.

The extension narrows its practical value:

1. The child starts at energy 75, below 20% of its 500 max, so a requested sprint 20 is reduced to a 10-unit walk.
2. Its initial heading is random and not in the preceding observations, so the blind walk cannot normally be aimed.
3. Movement occurs before `turn_angle`, so turning in the same extra action does not steer that movement.

The acted child ended at energy `74.240845`; the idle control ended at `74.9`. Both were already age `0.1` in the returned response. The parent saw the new child's ID and the child saw the parent in ordinary same-tick relative observations. Those age/drain/observation facts are compliant consequences of birth; acting before the child's first response violates the one-action-per-observed-agent contract.

### 5. Compliant but not prospectively exploitable: request order assigns birth RNG and IDs

With two observed parents each taking one birth action, request-list order decides which parent consumes each segment of the shared RNG stream and which lineage receives the next child ID. In the controlled two-lineage case, child ID 2 inherited speed 10 when parent 0 acted first and speed 18 when parent 1 acted first.

This affects deterministic replay and controller bookkeeping. It is not a demonstrated scoring advantage because the RNG state and mutation result are hidden until after the choice. Sorting actions by ID is the cleanest reproducibility convention.

### 6. Malformed validation gaps are state corruption, not useful tactics

- Unknown IDs are silently skipped, but the world still ages and drains living agents.
- `ActionRequest` accepts a NaN movement value in the local direct-DTO path. It poisoned energy to NaN and moved the fixture agent to the boundary through Python comparison/clamping behavior.

These are malformed inputs, not compliant tactics. NaN may also fail at an HTTP/JSON serialization boundary; that path was intentionally not tested. No survival or score benefit was established.

## Source path explaining the results

- [`src/utils/simulation.py:43-78`](../../vendor/survival-simulator/src/utils/simulation.py) iterates the entire action list before a single `non_agent_step` and does not enforce unique IDs.
- [`src/elements/environment.py:602-623`](../../vendor/survival-simulator/src/elements/environment.py) applies movement, then turning, then birth and inserts the child immediately.
- [`src/elements/environment.py:634-672`](../../vendor/survival-simulator/src/elements/environment.py) applies age/passive drain, checks death, computes observations, and only then collects fruit.
- [`src/elements/environment.py:282-392`](../../vendor/survival-simulator/src/elements/environment.py) uses shared RNG during birth and allocates sequential IDs.
- [`src/elements/environment.py:500-530`](../../vendor/survival-simulator/src/elements/environment.py) caps each action independently and limits a low-energy newborn to walking speed.
- [`src/utils/DTOs.py:4-12`](../../vendor/survival-simulator/src/utils/DTOs.py) defines floats but no finite-value or list-level validation.

## Coverage and rejected hypotheses

| ID | Hypothesis | Result |
| --- | --- | --- |
| A1 | Duplicate IDs execute sequentially before one world tick | Confirmed known issue; extended to 100 actions and two default maps |
| A2 | One oversized action bypasses sprint cap | Rejected: capped at 20 |
| A3 | Omission or unknown ID freezes an agent | Rejected: age/passive drain continue |
| A4 | A future child ID can act after birth in the same list | Confirmed known issue; reverse-order control passes |
| A5 | Newborn avoids the birth tick's aging/drain/observation | Rejected: age 0.1, idle energy 74.9, mutual observations present |
| A6 | Same-action turn steers same-action movement | Rejected: movement precedes turn |
| A7 | Energy just over 100 guarantees birth and parent survival | Rejected: action cost can cancel birth; passive drain can kill parent |
| A8 | Fruit collected this tick can finance this tick's birth | Rejected: birth waits until next tick |
| A9 | Reaching fruit rescues an agent after movement crosses zero | Rejected: death precedes collection |
| A10 | Two compliant births are order-independent | Rejected for deterministic replay; no prospective advantage |
| A11 | DTO validation rejects non-finite movement | Rejected locally; NaN corrupts state and is not useful |

## Boundaries and remaining uncertainty

- Proven compliant policy guidance: reproduction reserve accounting, food-arrival reserve accounting, newborn same-tick drain/observations, and deterministic action sorting.
- Proven only with rule-breaking input: repeated IDs and a same-request action for an unobserved child.
- Fixture-only: exact threshold energies, positioned ripe fruit, fixed forest biome, suppressed aging through large `max_age`, and diagnostic coordinates.
- Real generated-map evidence: duplicate movement magnitude only, seeds 1 and 42, one tick each. It confirms engine behavior, not permission.
- Not tested by design: hosted parsing/enforcement, competition APIs, score submissions, non-local targets, long-run malformed-input effects, and prediction of hidden RNG.

No remaining untested lead in this action-order area looked both high-value and contract-compliant. The main operational follow-up is straightforward: make the real controller subtract planned movement/turn and a conservative drain margin before requesting birth or food-reaching movement.

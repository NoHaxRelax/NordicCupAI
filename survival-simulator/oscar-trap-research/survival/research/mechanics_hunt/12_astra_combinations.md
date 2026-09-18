# Astra reviewer 12: interaction combinations

Local research against vendored commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. The engine is unchanged. [Executable](12_astra_combinations.py), [JSON evidence](../../results/mechanics_hunt/12_astra_combinations.json). Run from the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/12_astra_combinations.py --horizon 600 --seeds 1 7 42
```

## Controlled findings

These extend already documented aging, reproduction and collection timing rather than discovering those individual rules anew. Small fixtures replace terrain with forest and arrange positions, energy and ages; they use original engine actions and physics, natural founder energy/traits, and naturally reachable ripe fruit energy. They do not disable subsequent spawning. Their one/two-step outcomes are not full-game strategy evidence.

- **Birth before collection enables renewal but not same-tick funding.** A 90-energy parent attempting birth while touching a 60-energy fruit stays alone at 149.9 energy. Repeating birth next tick creates a child with 74.9 energy and leaves the parent at 49.8. Fruit must fund the next tick's birth.
- **Old-parent renewal moves energy into a cheaper body.** At assigned age 100 and aging threshold 60, an idle parent loses 1.101 energy in a tick. With birth it retains 48.899 from 150 and creates a 74.9-energy child. With a ripe fruit at its endpoint it instead retains 108.899. This spends 25 total energy to create the young reserve; it creates no energy. Population maintenance and food allocation determine whether it helps.
- **Emergency birth and predator targeting interact.** Arranged four-seed encounters compare idle, birth, sprint and sprint plus birth at founder energy 150, with an active predator 25 units away. All actions obey one-action-per-observed-agent. Idle loses the species in 4/4 cases; birth preserves one body in 3/4, with the parent itself surviving only 1/4. Birth can alter the nearest target immediately. Ordinary sprinting preserves the parent in 4/4 with 144.4 energy and score 0.1. Sprint plus birth leaves both alive in 3/4, but loses the child in the fourth, leaving only 44.4 parent energy and score -0.649. These controls reject birth as a generally better open-ground escape. The setup uses exact arranged positions and active predator state, so it is fixture evidence only.

## Generated-map experiment

Four policies use only ordinary agent DTO observations, traits, age, energy and simulation time. They use one action for every observed living agent, never act on a newborn in its birth tick, and have no access to map coordinates, predator energy, fruit age or engine state. Engine state is used only for result diagnostics.

- Baseline: existing nursery policy.
- Renewal: baseline plus one birth per parent after age 80, with energy over 125, visible food/tree, no nearby observed predator and a globally budgeted population ceiling of seven. The normal nursery ceiling is six. Hidden individual aging thresholds remain unknown.
- Thrift: baseline but its stationary, safe nursery scan turns only every tenth idle tick. This saves up to 0.172 energy per second while idle, but also reduces angular search coverage and changes later trajectories.
- Combined: renewal and thrift together.

All runs use default 1600 by 1200 generated environments, founders, random obstacles, fruit, trees, naturally spawning predators and unchanged energy accounting. No fixture edits enter these runs. Three seeds and a 600-second horizon are a screen, not a 3000-second completion benchmark. Policies alter shared RNG consumption through reproduction and other interactions, so paired seeds are not identical future event streams. Capped survivors have censored survival times.

An earlier 300-second development screen had incorrect extra-slot accounting in renewal modes. It was superseded by the corrected run and is excluded from final conclusions.

### Final 600-second results

Each cell gives score / survival seconds; `600*` is horizon-censored, with agents still alive.

| Policy | Seed 1 | Seed 7 | Seed 42 |
| --- | --- | --- | --- |
| Baseline | 591.78 / 600* | 605.87 / 600* | 94.55 / 93.6 |
| Renewal | 378.81 / 373.0 | 423.11 / 413.3 | 94.55 / 93.6 |
| Thrift | 582.82 / 600* | 596.49 / 600* | 392.83 / 389.5 |
| Combined | 597.46 / 600* | 371.70 / 374.4 | 620.17 / 600* |

Thrift improves seed 42 survival by 295.9 seconds, with score +298.28; on the other two seeds survival is censored equally and score is lower by 8.96 and 9.37. Combined changes seed 42 from early extinction to reaching the horizon, but reduces seed 7 survival by at least 225.6 seconds. Thus there is a proven per-seed advantage under compliant ordinary-observation play, but no robust general advantage. The combination is particularly seed-sensitive.

Average scores in this small screen are baseline 430.73, renewal 298.82, thrift 524.05, combined 529.78. The combined mean hides a large failure. These are exploratory seeds, not held-out validation; no claim of broad superiority or 3000-second completion follows. The idle-scan change affects perception, paths, food and predator encounters together, so its score gain cannot be attributed solely to saved turn energy.

Rank within this review: (1) thrift as a small, cheap candidate for larger validation, (2) combined as a seed-sensitive research variant, (3) renewal-alone rejected, (4) emergency birth retained only as an arranged fallback effect. Further broad-seed/end-horizon testing is required before adoption; this bounded investigation establishes mechanics and screens the proposed combinations, not exhaustive policy optimality.

## Coverage and rejected inferences

Tested combinations cover aging with reproduction/food; reproduction with immediate predator selection; and tree-oriented colony behavior with sensing energy, renewal, predators and natural terrain. Exact tree harvesting/maturity optimization and terrain-transition controllers are owned by other investigators and were not duplicated here.

Rejected: same-tick collected food funds a birth; reproduction creates net energy; lower rotation energy guarantees better score; a four-seed arranged emergency-birth survival implies a deployable escape tactic. Birth placement is random, food can be contested, predator observations are stale, and an extra body also consumes food.

The most useful ordinary-policy candidate is measured sensing thrift. The tested extra-slot renewal rule is rejected as a standalone improvement: it reduced survival on seeds 1 and 7. No new rule-breaking input exploit is used or recommended.

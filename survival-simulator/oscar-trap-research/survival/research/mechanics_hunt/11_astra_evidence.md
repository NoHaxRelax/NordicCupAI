# Astra evidence audit 11

17 September 2026. Independent checks against vendored commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. The executable imports the engine directly, does not import another investigator's harness, and leaves engine methods and vendor files unchanged.

Run from the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/11_astra_evidence.py
```

Evidence: [executable](11_astra_evidence.py), [JSON](../../results/mechanics_hunt/11_astra_evidence.json). Assertions cover every case. These are mechanics tests, not a score benchmark.

## Findings and corrections

| Claim | Independent evidence and control | Classification and magnitude |
| --- | --- | --- |
| Duplicate IDs execute repeatedly | One walk gives x110, energy149.4; ten walks give x200, energy144.9; both age0.1. | Known reproduction. Rule-breaking input under the documented one-action-per-observed-agent contract. No compliant recommendation. |
| Newborn can act in its birth request | A turn after the parent birth costs the child `1/(2*pi)` and turns it 1 radian. The same child action before birth does nothing. | Known reproduction plus ordering control. Predicts an unobserved child ID, outside the observed-agent contract. |
| Move and turn are simultaneous in the request, but not in execution | Move10 plus turn pi/2 goes east10, then faces north. Also reproduced with ordinary founder traits and energy on intact default-sized seed11 and42 maps; predicted endpoint error0. | Contract compliant. Observation-only controllers can use the pre-turn relative movement frame. There is no need for absolute coordinates to apply this rule. Prevents systematic steering errors. |
| Birth eligibility uses energy after both movement and turning | Starting100.5: idle birth succeeds; walk10 or turn pi blocks birth. Starting100.7 plus walk10 succeeds, parent ends0.1. | Contract compliant. Requires only visible own energy/traits and requested costs. Fixture sets near-threshold energy; no claim of measured population benefit. Plan strict `energy - movement_cost - turn_cost >100`, with extra passive/aging reserve if parent survival matters. |
| Maximum sprint tunnels through an ordinary wall | At evolved sprint40, width30 crosses from x195 to235. Width30.001 fails, redirecting to x195,y160. Founder sprint20 fails too. | The known full-face example is a knife-edge fixture. Radius5 means the minimum full-face path is wall thickness+10, and sprint caps40 before terrain. Generated random obstacles are width/height30–100; all160 obstacles across seeds11/42 had minimum thickness>30 (min32.5349 and30.3632). This rejects practical generalization to full-face tunneling of those obstacles. It does not reject corner shortcuts or other endpoint effects. |
| Boundary clamp provides easy access to an inside-wall position | Evolved20/40 traits at energy150 cross x35→5, then clamp inside the boundary; the identical request at energy99 is limited to walking and remains outside. | Known effect plus realistic-energy control. Synthetic evolved traits and arranged initial position; no evidence of natural acquisition, sustainable feeding, or refuge score benefit. One action is compliant, but strategy remains fixture-only. |
| Action-list priority decides which agent gets shared fruit | Two overlapping agents compete for one fresh20 fruit. Reversing the action list does not change energies169.9/149.9. The earlier-created agent eats first both times. | Contract compliant mechanics. Fruit collection uses `env.agents` list order, not request order. Coordinate endpoints to distribute food; merely sorting request objects cannot give a younger agent priority. |
| Returned observations are one coherent post-step snapshot | First agent reports the fruit it just ate; second agent reports none. Active predator reported40 away is actually25 away; resting control reports and remains40. | Known predator-lag reproduction and cross-agent fruit extension. Observation-only planning should tolerate already-consumed fruit and up-to-one-predator-move lag. Predator motion, terrain, energy and direction mean subtracting a fixed15 is not a universally correct correction. |
| Retained dead-agent cache yields a returned ghost agent | Removing an agent retains its cache entry, but `get_agent_state` returns None because the live dictionary is checked first. | Rejected as a returned-agent exploit. Explicit diagnostic removal is privileged; cached references to other entities can still be stale. |
| Exact-distance targeting picks lowest ID reliably | Two agents at distance50 both appear; `min` picks the first observation. That observation order comes from a set of object identities. | Source-level nondeterminism warning, not a claim of measured cross-process reversal. A single fixed-seed result cannot establish ID tie priority. Avoid strategies depending on exact ties. |

## Fixture limits and interactions

Most probes use a 400×400 environment with its normal boundaries and an explicitly overwritten forest biome. Positions, fruit placement, predator energy/rest state and selected trait/energy values are arranged. Founder energy stays150 except the labelled threshold tests; evolved traits are explicitly assigned. Normal tree/predator spawning code remains enabled. Short one-step probes do not establish longevity. The two generated-map checks retain their complete generated terrain, obstacles, five founders,32 fruits,50 trees and normal spawning; no policy score claim comes from them.

Useful compliant combination: select a fruit endpoint in the pre-turn frame, account for movement/turn costs before choosing birth, and use distinct endpoints for hungry agents rather than request sorting. After a step, discard fruit targets plausibly collected by any teammate and apply conservative predator margins. These combine proven local rules; a measured score improvement remains untested.

The strongest audit downgrade is full-face wall tunneling: the exact30-width example is real but does not transfer to the sampled random obstacles. Boundary thickness is exactly30 and remains a separate effect with extra clamp behavior. Do not conflate these two cases.

## Source anchors and rejected interpretations

- `src/utils/simulation.py:step_environment`: request-order action execution followed by one world update; current dictionary enables an already-created child.
- `src/elements/environment.py:agent_step`: move, turn, then birth. `update_entity_position`: energy gate, cost, biome multiplier, endpoint collision search, then clamp. `_in_obstacle`: strict inequalities on expanded rectangles, not circle-to-rectangle distance.
- `src/elements/environment.py:non_agent_step`: agent-list iteration, observations before own fruit collection, all predator moves later. `get_agent_state` combines current own scalar state with cached observations.
- `src/elements/creature.py:observe` and `src/elements/predator.py:step`: list order and first-min targeting; no declared stable tie key.

Rejected here: interpreting movement direction relative to the newly requested heading; reserving exactly100 before motion for birth; sorting actions to allocate shared fruit; generalizing width30 full-face tunneling to width30.001; treating a resting predator as having15 units of observation lag; treating cache retention as returned dead-agent state. Remaining uncertainties are practical policy value, evolved-trait acquisition, corner-crossing frequency and hosted-engine equivalence. No remote validation was performed.

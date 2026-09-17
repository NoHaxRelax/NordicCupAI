# Mechanics hunt 03: observations, visibility, and relative coordinates

Tested locally on 17 September 2026 against the unmodified vendored simulator at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. No network, hosted validation, evaluation, submission, or remote target was used.

Run the assertions and regenerate the evidence:

```bash
PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy \
  survival/.venv/bin/python survival/research/mechanics_hunt/03_observation.py
```

Machine-readable results are in [`survival/results/mechanics_hunt/03_observation.json`](../../results/mechanics_hunt/03_observation.json). The script contains minimal positive cases and negative controls. Synthetic positions and direct sensing calls isolate rules; the generated-map check uses default map dimensions, traits, starting energy, and seeds 1, 42, and 2026.

## Ranked findings

### 1. `Edge` observations repeat ray hits, not unique wall segments

One vertical edge produced five identical `Edge` payloads because five visibility rays hit it. This is not a stale cache: `compute_visibility` appends the complete source edge once for every ray whose nearest collision is that edge, without deduplication.

This occurs heavily in normal maps. Across the first ordinary response for 15 default agents on seeds 1, 42, and 2026:

- 13 agents received at least one edge.
- 151 edge entries represented only 32 unique segments within their respective agent responses.
- 119 entries, or 78.8%, were duplicates.

Practical recommendation: deduplicate exact edge coordinates before map construction, collision reasoning, or counting obstacles. Retaining multiplicity may provide a noisy angular-prominence signal, but it is biased by the five fixed rays and extra corner rays. No score advantage has yet been benchmarked.

Evidence: `visibility_boundaries.edge_observations_are_ray_hits_not_unique_segments` and `generated_map_validation` in the JSON. Source: `src/utils/sensing.py:91-104` and `src/elements/creature.py:172-177`.

Classification: proven under ordinary observations, contract-compliant, and reproduced on untouched generated maps.

### 2. A response mixes several moments within the same tick

Agent actions all execute first. Each surviving agent then receives a cached observation before fruit collection. Predators move after every agent observation. Fruit and tree lifecycle work happens later still. The final response pairs those earlier observations with post-step energy, score, time, and living-agent membership.

Consequences:

- Known result reproduced: a predator reported at distance 40 had already moved to distance 25 when the response was returned. The normal maximum error from this phase ordering is one predator sprint step, 15 units.
- New extension: a fruit at distance 5 was reported after it had already been removed and scored. A naive controller can chase that phantom for another tick, wasting up to a founder's 20-unit sprint and 5.5 movement energy.
- This fruit phantom occurred without arranged geometry on default seed 1: agent 3 ended the first step at 169.9 energy while its response still contained a fruit at distance 7.0014. That is the normal 150 start, minus 0.1 passive cost, plus the fresh fruit's 20 energy.
- Another agent can appear by ID in a survivor's cached observations and then be eaten later in the tick. Unlike predators and fruit, this stale entry is fully correctable by discarding observed agent IDs missing from the response's top-level living-agent list.

Practical recommendation: subtract a 15-unit safety margin from predator separation, suppress fruit known to have reached contact, and cross-check observed agent IDs against living status IDs.

Evidence: `phase_timing.predator_pre_move_snapshot`, `consumed_fruit_remains_in_response`, `eaten_agent_stale_but_cross_checkable`, and `generated_map_validation.same_step_consumed_fruit_still_observed`. Source: `src/elements/environment.py:649-735` and `src/utils/simulation.py:21-42`.

Classification: proven mechanics. Policy corrections use ordinary response data and respect one action per living observed agent. True post-step positions and collection membership were read only to verify the engine cause.

### 3. A starvation death can preserve the following survivor's old cache for a second tick

This extends the already documented list-removal bug. When the first agent in the list died during passive upkeep, removing it caused the next agent to be skipped. Its age and passive cost were skipped, but so was its observation refresh. The predator still moved normally.

The survivor therefore received the previous tick's predator distance of 80 again while the predator's actual post-step distance had fallen to 50. The cache was now two predator steps, 30 units, behind.

This is detectable only indirectly in ordinary play. An unchanged age is the clearest marker that this agent was skipped, so a controller can distrust its observations for that response. It is rare and normally harmful because the skipped agent also misses fruit collection. Deliberately sacrificing an earlier list agent to trigger it is not recommended.

Evidence: `phase_timing.death_skip_preserves_old_observation_cache`.

Classification: proven in an induced normal-energy starvation fixture, one-action compliant, but not a demonstrated strategy advantage.

### 4. The vision polygon has strict boundaries and small range scallops

The object prefilter accepts distance and cone boundaries inclusively, but final visibility uses Shapely's strict `Polygon.contains`. An object exactly at range 200 or exactly at the 30-degree half-cone boundary is excluded; points immediately inside are included.

With no nearby obstacle corners, the circular sector is approximated by only five rays. Straight chords between them reduce effective range between ray angles:

- Default range 200 and cone 60 degrees lose at most 1.711 units. At 7.5 degrees, distance 199.5 was invisible while 198 was visible.
- Trait-cap range 400 and cone 90 degrees lose at most 7.686 units. At 11.25 degrees, 392.3 was visible while 392.4 was not.

This is a real sensing blind scallop, not occlusion. It is too small for a standalone tactic, but it argues for hysteresis around acquisition and loss boundaries rather than assuming a perfect circular sector.

Evidence: `visibility_boundaries.strict_vision_boundaries` and `polygon_chord_range_scalloping`. Source: `src/elements/creature.py:128-160` and `src/utils/sensing.py:44-63`.

Classification: proven with exact-coordinate fixtures using ordinary sensing and legal trait bounds. No score impact benchmarked.

### 5. Hearing has a sharp, inclusive, wall-ignoring boundary

At exactly the default 50-unit hearing radius, fruit was sensed directly behind the agent and through a wall. At 50.01 behind the wall it disappeared, while the same unblocked object at 50.01 was visible.

This confirms a sharp mode switch: within `distance <= hearing_radius`, angle and occlusion do not matter; immediately beyond it, the object must pass both cone and polygon checks. Controllers should add boundary hysteresis because a tiny displacement can change an observation from omnidirectional and wall-proof to absent.

Evidence: `visibility_boundaries.hearing_threshold_and_occlusion`. Source: `src/elements/creature.py:130-160`.

Classification: proven ordinary mechanics, contract-compliant. It reproduces documented hearing-through-wall behavior and adds the exact threshold control.

### 6. Coordinates are consistently egocentric, but the snapshot uses the post-turn frame

Movement is applied relative to the old facing, then the requested turn is applied, then observation angles and edge coordinates are computed. In the reproduction, a target at world offset `(0, 100)` became angle 0 after a `pi/2` turn in the same action.

Edge coordinates use local x forward and local y left. With the observer facing world positive y, a horizontal world edge from `(90, 150)` to `(110, 150)` became local `((50, 10), (50, -10))`.

`rel_dir` is the observer's bearing in the observed creature's frame. For a predator east of an agent, `rel_dir = 0` when the predator faces west toward the agent, and `abs(rel_dir) = pi` when it faces east away from the agent. This lets a policy distinguish approach from retreat without privileged state.

Evidence: `coordinates_and_identity.observations_use_post_turn_egocentric_frame`, `edge_coordinate_axes`, and `relative_direction_semantics`. Source: `src/elements/environment.py:496-566` and `src/elements/creature.py:73-93, 134-177`.

Classification: proven from ordinary response fields and known own actions, contract-compliant.

### 7. Non-agent identity and important state remain genuinely hidden

Fruit, trees, and predators have no observation IDs. Two colocated fruits produced identical dictionaries containing only type, distance, and angle. Predators add `rel_dir`, but still expose no ID, energy, or resting state. Fruit payloads expose no age, energy, radius, or ripeness. The agent's own absolute x, y, and facing are also absent.

Ordinary inference can recover relative object positions, observed-creature facing, local edge endpoints, and dead observed agents via ID cross-check. Absolute pose is recoverable only up to an arbitrary initial frame and becomes uncertain when collision redirection changes the requested movement path.

Evidence: `coordinates_and_identity.non_agent_identity_is_ambiguous`.

Classification: payload-level proof. Tracking fruit and predators across crossings remains an association heuristic, not confirmed identity.

## Timing controls and rejected hypotheses

- Partial observations in action-list order were rejected. Both agents completed their moves before either observation was computed: starting separation 100 became 80, not the partial 90.
- A normal parent action makes its newborn visible and returns the newborn's status at age 0.1 on the birth step. This does not authorize a child action in that request because the child was not in the input observation.
- Stable non-agent identity from list position was rejected. Objects are gathered through sets and non-agents have no IDs.
- The empty-edge fast path is broken: `observe(..., edges=[])` raises `IndexError` before reaching it. This is not useful on official default geometry. With a 1600 by 1200 map and chunk size 400, boundary edges cover every possible three-by-three chunk neighborhood; all tested default agents had local edges even when no edge was visible.
- An unseen newly spawned predator is not an immediate ambush under normal state. Predators spawn resting at zero energy and must recharge before acting. Detailed rest-transition work belongs to the predator-state investigation.

## Practical policy patch list

1. Canonicalize and deduplicate edge endpoint pairs per agent response.
2. Treat observed predator distance as pre-move and reserve at least 15 units beyond the desired safety radius.
3. Record fruit reaching contact and suppress it for the next decision unless independently reacquired.
4. Drop observed Agent IDs that are absent from the living status list.
5. If an agent's age does not advance, treat all its cached observations as potentially another full tick older.
6. Use epsilon or hysteresis around hearing, cone, and maximum-range thresholds.
7. Apply own turn before interpreting returned angles; edge local x is forward and y is left.

These changes are all compatible with the documented one-action-per-agent rule. This investigation found no need to use malformed requests, duplicate actions, predicted newborn actions, hidden engine state, or remote validation. It demonstrates mechanics and controller-hardening opportunities, not a measured score improvement or literal exhaustive proof.

# Seed recovery: data requirements and useful generator shortcuts

19 September 2026. Native simulator revision `282446421263e7bb11081d692f0ac443b99f9e20`.

**The generator code helps substantially. We likely already collect enough information to distinguish a 32-bit seed's map; the unresolved problem is efficiently computing the seed that explains it.** A new experiment demonstrates an inexpensive necessary condition on seeds using public boundary geometry and biome labels. A separate linear-algebra experiment quantifies the much larger data requirement for recovering an unrestricted RNG state.

These are different problems. Neither experiment recovers an arbitrary seed from public observations without enumeration.

**Latest follow-up:** A [30-second walking survey using the shared world model](seed_recovery_survey_2026-09-19.md) reduced the 100,000-seed range to one candidate for three in-range targets and no candidates for an out-of-range target. All three recovered seeds reproduced the full 300-frame public history. The same report explains uncertainty handling, shared-map alignment, collection settings and measured timings.

## How many measurements?

### Identifying a 32-bit seed

At least 32 bits of distinguishing information are needed in the ideal case to distinguish all 2^32 possible seeds. Under an approximately independent fingerprint model, use about 52 effective bits to bring the expected number of wrong matching seeds below one in a million: `2^32 * 2^-52 < 10^-6`. This is a model of collisions, not an inversion algorithm or a guaranteed bound for the native maps.

A single scalar measurement can supply many bits. The native rock dimensions span 30–100. If the error tolerance is `epsilon`, an unrelated map containing 160 random dimensions has approximate probability

```text
q = 1 - (1 - 2 * epsilon / 70)^160
```

of matching one specified observed length somewhere. This accounts for not knowing which rock generated it. For `k` independent, distinct lengths, the rough expected number of wrong matching seeds is `2^32 * q^k`.

| Length tolerance | Approximate information per length | Rough length count for expected wrong matches below 10^-6 |
| --- | ---: | ---: |
| 0.000001 | 17.74 bits | 3 |
| 0.002 | 6.78 bits | 8 |
| 0.2 | 0.74 bits | 71; a poor collection strategy |

**These are planning estimates, not verified minimum counts.** They assume roughly independent uniform dimensions, sufficiently separated query lengths, and independent matches. Visibility selects which rocks are observed; nearby length intervals overlap at coarse precision; measurements and seed outputs can be correlated. The earlier rounded-coordinate experiment actually produced a false match at tolerance 0.2. Full geometry and additional observations must verify a candidate.

A reasonable next collection target is **10–20 distinct rock dimensions at full available precision**, retaining their endpoints and associations, plus all agent/biome/fruit/tree evidence. This provides margin for ambiguities and validation; it does not promise unique identification for every seed. The native rotation experiment below already exceeds this target on all three tested maps.

### Recovering the unrestricted Mersenne Twister state

The RNG has a 19,937-bit state dimension. CPython's `random()` exposes 27 bits of one output word and 26 of the next. Consequently `ceil(19937 / 53) = 377` floats is only an information-count lower bound. Bit constraints can be dependent. See the [CPython 3.12.12 implementation](https://github.com/python/cpython/blob/v3.12.12/Modules/_randommodule.c).

Added [mt_float_rank_probe.py](../scripts/mt_float_rank_probe.py). It builds symbolic bit equations for the native MT recurrence, adds the alternating 27/26 visible bits, and measures their rank over GF(2). Every symbolic output word is checked numerically against the installed `random.Random` implementation using a known control seed.

| Exact consecutive `random()` values | Independent state constraints |
| ---: | ---: |
| 320 | 16,612 |
| 377 | 17,296 |
| 400 | 17,572 |
| 500 | 18,697 |
| 600 | 19,707 |
| 622 | 19,927 |
| **623** | **19,937: full rank** |

The experiment assumes a known starting twist boundary, exact original floats, no missing draws, and known order. It first reaches full rank at **623 floats**, checking 1,246 output words against CPython. Equation construction and rank calculation took 5.75 seconds locally. This is not a measured runtime for recovering a seed from observations.

Do not translate that into “623 rocks” or “623 visible events.” One complete rock has four generating floats, but there are only 80 random rocks, yielding 320 calls even with the entire map and correct ordering. Public edge transformations lose some original precision, rock creation order is hidden, and other events create gaps. More observations of the same static rock do not produce new RNG outputs. Partial-output state reconstruction with gaps and finite precision needs a separate rank/solver experiment; its required count can differ. The 32-bit seed restriction is deliberately absent from this unrestricted-state calculation.

Measurements: [mt_float_rank_probe_2026-09-19.json](mt_float_rank_probe_2026-09-19.json).

## Getting more useful data through ordinary actions

Added [seed_biome_prefix_probe.py](../scripts/seed_biome_prefix_probe.py). Each target uses the native default world, one empty tick, then six stationary 60-degree turns for all five agents. The collector receives only JSON-round-tripped public states. It records distinct rock lengths across views and estimates absolute poses from complete fixed-world-boundary edges.

| Target seed | Initial distinct rock lengths | After the turns | Agents localized from boundaries |
| --- | ---: | ---: | ---: |
| 3 | 7 | 28 | 4 of 5 |
| 11 | 10 | 54 | 4 of 5 |
| 20260919 | 7 | 38 | 2 of 5 |

The complete sequence ended at simulated time 0.7 seconds. Six 60-degree turns cost one energy per agent through the native turn-cost formula, in addition to the ordinary passage of time. All actions were local test actions; no live evaluation was performed.

Localization uses the known 1,600 by 1,200 dimensions, 30-unit wall thickness, directed edge endpoints and the fact that interior agents see the inner wall faces. The full boundary length identifies the world axis; rotating and translating its endpoints recovers the observer pose. This depends on the inspected native endpoint ordering and unrounded observations. Across 27 available localization checks, positions agreed with hidden simulator truth to within `2.6e-13` world units. Hidden positions were used only to validate the result; neither the collector nor the seed filter receives them. The probe handles possible integer-pixel alternatives around floating-point rounding boundaries.

The next collection steps should be:

1. Retain every agent's first frame, then survey its surroundings with turns. Deduplicate repeated rock dimensions and keep the full edge geometry.
2. Anchor agents that see a world boundary. Transfer localization through identified other-agent sightings or consistently matched shared landmarks.
3. Explore different regions and record **biome transitions at localized positions**. Use repeated landmarks to correct movement estimates: native collision handling can rotate a movement attempt, so commanded displacement alone is not exact odometry.
4. Collect fruit/tree geometry along the same route. Preserve object associations where they can be established and distinguish initial objects from later appearances.
5. Prefer observations that discriminate remaining hypotheses. Nearby samples inside one biome usually add less than a new boundary crossing or a landmark in an unexplored area.

Thousands of duplicate observations are not thousands of independent data points. Likewise, all pairwise distances between `n` points do not give `n*(n-1)/2` independent random measurements: a generic 2D arrangement has only `2n-3` continuous geometric degrees of freedom after translation and rotation are removed, for `n >= 2`. Birth mutations expose additional random outcomes later, but reproduction costs energy, introduces conditional draws and contains hidden constructor draws; it is not the first collection route to prioritize.

## A concrete shortcut enabled by the generator source

The ten Voronoi sites and their land types are generated **before the river, rendering, rocks and agents**. For a public sample localized at pixel `(x, y)`:

1. Generate just the candidate seed's ten sites and ten type choices.
2. Find the nearest site, using the generator's integer coordinates and tie ordering.
3. If the observed biome is non-river and the candidate land type disagrees, reject the seed immediately.

This is a necessary condition because river generation only overwrites land with `river`. It cannot turn a desert base cell into an observed forest cell. A river observation imposes no restriction in this preliminary land-only check. Candidates that pass still need river and geometry verification.

Crucially, we query each candidate map at an **already reconstructed world coordinate**. We do not need to generate that candidate's starting agents. This avoids the expensive initialization prefix that the earlier ID-indexed startup-biome comparison required.

The public rotation experiment supplied 4, 3 and 1 non-river samples respectively. Scanning candidate integers 0–99,999 gave:

| Target | Non-river samples | Survivors | Rejected |
| --- | ---: | ---: | ---: |
| 3 | 4 | 99 | 99.901% |
| 11 | 3 | 1,526 | 98.474% |
| 20260919 | 1 | 24,889 | 75.111% |

The complete 100,000-prefix scan, checking **all three targets**, took **2.386 seconds**. All three true seeds passed when checked individually, including 20260919 outside the scanned range. The new native worlds took about 1.48 seconds each to initialize, so early rejection avoids substantial work.

**This remains enumeration, and the survivor counts are not recovery results.** A linear extrapolation gives about 28.5 hours for this prefix-only implementation to examine all 2^32 integers while checking the same three targets; it excludes expensive survivor verification and is not a full-range benchmark. The first target's measured survival fraction would still leave millions of full-range candidates. We need more discriminating observations or a stronger constraint solver before treating this as practical in-game recovery.

Measurements, public samples and source hashes: [seed_biome_prefix_probe_2026-09-19.json](seed_biome_prefix_probe_2026-09-19.json).

## Combining biomes, spawn headings and seen rock edges

**Implemented and tested:** [seed_joint_constraints_probe.py](../scripts/seed_joint_constraints_probe.py) combines the early biome condition with reconstructed founder headings, founder positions, rock lengths and complete observed rock edges in world coordinates.

### Rock orientation exposes the heading

The earlier length-only fingerprint discarded orientation. The native `Environment.spawn_obstacle` constructs sensing edges with endpoints ordered along positive world x or positive world y. This differs from the winding order in `Obstacle.edges`; the collector depends on the actual environment sensing edges. See [environment.py](../src/elements/environment.py), `spawn_obstacle`, and [creature.py](../src/elements/creature.py), the `Edge` observation transform.

If an observed edge has local direction `phi`, the agent's absolute heading must be either `-phi` or `pi/2 - phi`, modulo 2*pi. A single directed edge leaves two axis hypotheses. Two perpendicular directed edges normally leave only one common heading. **A world boundary is not needed for this heading calculation.** A boundary additionally provides the absolute position.

For the stationary rotation sequence, subtract the sum of known turns to recover the initial spawn heading. This heading was generated by `uniform(0, 2*pi)` and supplies an interval constraint on an additional RNG output. Preserve angle wrapping, measurement tolerance and unresolved hypotheses; do not arbitrarily choose one of the two possible axes when the evidence is insufficient.

Across all three targets, the seven-frame collector recovered unique initial headings for all five founders. Their maximum angular errors against hidden validation truth were below `9e-16` radians. Hidden headings and positions only check extraction correctness; candidate matching receives the public-derived signature. If a service changes endpoint ordering or precision, this source-specific inference must be revalidated.

### Complete bounded search

For target seed 3, the collector obtained five initial headings, four absolute positions, 28 distinct rock dimensions and 24 distinct rock edges in world coordinates. It enumerated seed integers 0–99,999 without a prebuilt map catalog:

| Check | Remaining candidates |
| --- | ---: |
| Initial range | 100,000 |
| Public localized biome condition | 99 |
| Biomes plus founder headings | 1: seed 3 |
| Biomes plus founder positions | 1: seed 3 |
| Biomes plus rock lengths | 1: seed 3 |
| Biomes plus complete localized rock edges | 1: seed 3 |
| All checks together | 1: seed 3 |
| Replay all seven public frames with the known actions | 1: seed 3 |

The rows after the biome filter are individual comparisons against the same 99 survivors, followed by their intersection. Each feature family already isolated the true candidate in this experiment, so it does not demonstrate additional elimination from combining them beyond either strong check alone. It does demonstrate that their constraints agree and that the recovered candidate reproduces the public behavior. Replay compares all public fields, normalizing observation list order rather than comparing only selected geometry features.

The biome pass took **1.970 seconds**. Constructing all 99 candidate native worlds took **167.27 seconds in total**, including the candidate used for replay. These are accumulated phase timings, not end-to-end elapsed time. A Windows denial of an atomic result-file replacement interrupted the original run after 35 candidate evaluations; the completed run resumed from the saved temporary checkpoint. Collection, interruption and other overhead are not included in the phase total. The completed evidence records the checkpoint hash and resumption details.

Two additional controls checked seeds 0–15: target 11 left two biome survivors and uniquely matched seed 11 with all checks and seven-frame replay; target 20260919, outside that range, left five biome survivors and rejected all five with the subsequent checks. These controls also recovered all five starting headings, with four and two localized positions respectively.

Evidence:

- [Completed 100,000-seed search](seed_joint_constraints_completed_2026-09-19.json).
- [Known-seed control](seed_joint_constraints_control11_2026-09-19.json).
- [Out-of-range control](seed_joint_constraints_unknown_2026-09-19.json).

Four focused checks passed for one-edge ambiguity, perpendicular-edge resolution, subtracting known turns and missing-edge evidence. Native replay passed for both known targets. No simulator or policy code was changed for this experiment.

### What combining the evidence changes

The joint signature preserves more information already present in the observation payload and provides several ways to reject false candidates. It also supplies a better set of constraints for a future direct solver: early biome geometry, later rock geometry, and successful-founder headings and positions associated with agent IDs.

It does **not** make the late-stage checks free. The candidate's rocks and initial agent headings occur after the rendering RNG draws. This implementation still builds native worlds for biome survivors. Therefore the measured result establishes bounded seed identification with joint checks, not efficient recovery across all 2^32 seeds or algebraic inversion. More precise biome localization/filtering or a solver that handles the seed-to-observation equations is still needed for that broader goal.

Reproduce the full bounded search from `survival-simulator`, with a fresh output filename:

```text
python scripts/seed_joint_constraints_probe.py --output docs/seed_joint_constraints_repeat.json
```

The default cap is 200 native worlds. If a different target leaves more survivors, the probe saves the prefix result with `complete=false` and stops before native verification.

## What the code enables beyond enumeration

The source gives the precise forward equations and branch conditions, which are prerequisites for stronger inversion:

- Rock dimensions become intervals for underlying random fractions: `u_width = (width - 30) / 70`; localized positions add `u_x = x / (1600 - width)` and the corresponding y equation. Floating-point arithmetic requires interval treatment rather than assuming those divisions recover exact original fractions.
- Localized biome boundaries constrain the ten early site positions and type assignments. These are closer to the initial seeded state than the rock block. Repeated types can hide Voronoi boundaries, and rivers obscure base land, so visible regions cannot simply be read back as the generator's ten sites.
- Offline instrumentation can label every random call by object, purpose and order without changing its values. This supplies a controlled benchmark for a partial-output solver, after which the assistance can be removed. Instrumented hidden values must never be counted as player-visible evidence.
- A SAT/SMT model can constrain the original 32-bit seed and the early generator steps, rather than reconstructing an unrestricted 19,937-bit state or simulating every complete map. Runtime and required public observations remain unmeasured for this native problem.

The next useful milestone is a hybrid recovery experiment: collect and localize public geometry, use early biome constraints to reduce work, add precise rock equations, and recover a held-out seed outside any prebuilt catalog. The current evidence supports pursuing that experiment. It does not support either declaring unrestricted recovery impossible or claiming that a known number of public measurements already makes it efficient.

Reproduce from `survival-simulator`, selecting fresh filenames:

```text
python scripts/mt_float_rank_probe.py --output docs/mt_float_rank_repeat.json
python scripts/seed_biome_prefix_probe.py --output docs/seed_biome_prefix_repeat.json
```

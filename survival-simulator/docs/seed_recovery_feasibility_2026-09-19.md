# Recovering the simulator seed from rocks, trees, agents and biomes

19 September 2026. Simulator revision: `282446421263e7bb11081d692f0ac443b99f9e20`.

**Finding: seed identification is practical when the true seed is in a manageable candidate set. This was demonstrated using public rock edges and tree observations. General recovery of an arbitrary unknown seed has not been demonstrated.** The main obstacle is efficiently finding the matching seed, rather than obtaining a distinctive map fingerprint.

**Follow-up:** [Data requirements and generator shortcuts](seed_recovery_data_budget_2026-09-19.md) measures information collection through rotation, demonstrates absolute localization from public boundary edges, and uses localized biome samples to reject seed candidates before expensive world generation. It also measures the ideal ordered-float requirement for unrestricted RNG-state recovery. This improves the search approach beyond the initial map catalog without establishing general seed inversion.

**Joint-check result:** The same [follow-up report](seed_recovery_data_budget_2026-09-19.md#combining-biomes-spawn-headings-and-seen-rock-edges) now demonstrates recovering all five founder headings from directed edges, then combining biomes, headings, positions and rock geometry in a complete 100,000-seed bounded search. Only the correct seed survived, and it reproduced all seven public frames. An additional out-of-range control was rejected. Candidate enumeration remains necessary in this implementation.

**30-second survey result:** [Walking and looking with the shared world model](seed_recovery_survey_2026-09-19.md) reduced the same search range to one candidate for each of three in-range targets. Each matched all 300 public frames; an out-of-range target was rejected. This substantially reduced expensive native-world verification compared with searching after the initial stationary survey.

## What the code exposes

[SimulationCore](../src/core.py), lines 21–25, selects an integer between 0 and `2**32 - 1` when no seed is supplied, then creates `random.Random(seed)`. Explicitly supplied seeds need not follow that range. The repository says final evaluation uses preset seeds, but does not reveal those seeds or establish that they are small integers. A fixed seed is not necessarily a known seed.

The world shares this RNG between map generation, rendering, entity initialization and subsequent events. The order is:

1. Generate biome sites/types and the river, including a variable-length random walk.
2. Render the 1,600 × 1,200 biome surface, making **1,920,000 random palette selections**.
3. Generate 80 rectangular rocks, each with random width, height, x and y.
4. Generate initial agents, fruits and trees, with collision retries, biome-dependent tree acceptance and random initial tree growth.
5. During play, interleave reproduction, predator behavior, tree spawning/death, fruit spawning and predator spawning.

References: [environment.py](../src/elements/environment.py), lines 38–78, 249–260, 447–467 and 737–762; [biome.py](../src/elements/biome.py), lines 73–163; [simulation.py](../src/utils/simulation.py), lines 20–37.

The palette selection count is fixed for the default dimensions, but the number of underlying generator draws is not. `choice()` uses rejection sampling, and palette lengths differ by biome. Removing the renderer or skipping a fixed number of `random()` calls produces a different rock map. This follows from the inspected local standard library and the [CPython 3.12.12 random implementation](https://github.com/python/cpython/blob/v3.12.12/Lib/random.py).

## Rocks are the strongest starting point

The sensing code returns the **whole endpoints of each hit edge**, including repeated copies when several rays hit it. They are transformed into the observing agent's coordinate frame; they are not clipped to the visible portion of the edge. See [sensing.py](../src/utils/sensing.py), the `hit_edges` construction, and [creature.py](../src/elements/creature.py), the `Edge` observation construction.

For public endpoints `(x1, y1)` and `(x2, y2)`, calculate:

```text
side_length = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

Rotation and translation do not change this length. Therefore no absolute pose, heading, map origin or hidden simulator state is needed to extract a rock's width or height. Deduplicate repeated observations, and exclude the fixed boundary dimensions. Random rock sides lie between 30 and 100 units.

Candidate seeds can be generated offline and indexed by their rock-side lengths. Several observed dimensions can then eliminate almost all incompatible candidates cheaply. A single matching length should not be considered sufficient across a very large catalog; the probe's single-length successes below only apply to its 16 candidates.

Trees provide another fingerprint: convert two tree observations from the **same agent and tick** from polar to local Cartesian coordinates, then measure their separation. That separation is also independent of agent pose. Distances from different agents cannot be combined without first aligning their frames. Initial tree spacing is useful for confirming a candidate, but trees disappear and new ones spawn, so a fixed initial-tree catalog cannot explain arbitrary later observations.

## Local proof of concept

Added [seed_recovery_probe.py](../scripts/seed_recovery_probe.py). It builds a catalog from native worlds for seeds **0–15**, then independently regenerates three target worlds and gives the matcher only their JSON-round-tripped first-tick public observations. The target seed is an evaluation label, not an input to the matcher. The simulator and RNG were not patched.

The candidate catalog may use complete generated maps: those are known hypothetical worlds created by the investigator. The unknown target's fingerprint is derived strictly from public observations.

| Target | In catalog? | Distinct observed rock lengths | Observed tree-pair distances | Rock result | Tree result |
| --- | --- | ---: | ---: | --- | --- |
| 3 | Yes | 7 | 2 | Uniquely identified seed 3 | Uniquely identified seed 3 |
| 11 | Yes | 10 | 1 | Uniquely identified seed 11 | Uniquely identified seed 11 |
| 20260919 | No | 7 | 0 | No matching candidate | Insufficient tree evidence |

At full precision the matching tolerance was `1e-6` world units. Catalog generation took **28.24 seconds total**, averaging **1.765 seconds per seed** on this machine. Geometry matching against the completed 16-seed catalog took approximately **0.014–0.030 ms**, excluding feature extraction and catalog construction. These tiny timings are illustrative, not a general throughput benchmark.

The coordinate-precision experiment is important:

- With edge coordinates rounded to **3 decimals**, both known seeds were still identified and the unknown seed was rejected, using a `0.002` length tolerance.
- With coordinates rounded to **1 decimal**, both known seeds were still identified, but the unknown seed incorrectly matched candidate **5**, using a `0.2` tolerance.

This false match demonstrates why a recognizer must support “unknown” and independently verify candidates. Require full edge geometry under a consistent rigid transform, more observations, and/or tree-layout checks before treating a seed as recovered. Matching several unrelated lengths within a loose tolerance is insufficient.

Three focused checks also passed: rotation invariance/tree distances, unknown/empty/ambiguous evidence, and tolerance/boundary filtering. No complete game or remote evaluation was run; targets used one native empty tick.

Full measurements, candidate fingerprints, runtime versions and source hashes are saved in [seed_recovery_probe_2026-09-19.json](seed_recovery_probe_2026-09-19.json). Reproduce from `survival-simulator`, choosing a fresh output filename:

```text
python scripts/seed_recovery_probe.py --output docs/seed_recovery_probe_repeat.json
```

This is a small feasibility experiment, not an estimate of identification success across all seeds or all observation conditions.

## Including all starting-agent information and biomes

**Use the complete first public frame from all five agents.** The original probe already pooled rock and tree measurements from every agent, but did not use their biome labels, other-agent sightings, fruit observations or energy. These additional constraints should be included in a general recovery model. More constraints improve identification; they do not by themselves supply an efficient way to invert the seed.

The native engine exposes 11 fields per agent. Their value for recovery differs:

| Public information | What it contributes |
| --- | --- |
| `agent_id` | Associates measurements with the same agent and gives successful founder creation order. IDs 0–4 are deterministic, so they are not fresh random bits. |
| `biome` | One of forest, swamp, desert, grassland or river at this agent's current position. A useful categorical constraint on the generated map. |
| `observations` | Rock endpoints; fruit/tree distances and bearings; other agents' IDs, distances, bearings and `rel_dir`. These constrain positions, headings, object associations and visibility. |
| `speed`, `sprint_speed`, `max_energy`, `hearing_radius`, `vision_range`, `vision_angle` | Fixed founder values: 10, 20, 500, 50, 200 and pi/3. They define sensing/movement rules but reveal no seed-dependent founder trait draws. |
| `age` | Starts at zero; the first empty tick reports 0.1. It establishes timing rather than a random value. |
| `energy` | Starts at 150, normally 149.9 after that tick. Initial fruit pickups can raise it, providing a consistency check on nearby fruit and event order. |

Sources: [environment.py](../src/elements/environment.py), `spawn_agent`, `get_agent_state` and `non_agent_step`; [agent.py](../src/elements/agent.py), constructor; [creature.py](../src/elements/creature.py), constructor and `observe`.

Founders **do** have random x/y positions, headings and maximum ages. These values are not included directly in their public metadata. Each successful default spawn consumes four `random()` calls through `uniform()`: x, y, heading, maximum age. Failed position attempts consume extra x/y draws before retrying. IDs establish successful creation order, but not the number of skipped draws. Maximum age is hidden, unlike current age.

Other-agent observations can align otherwise separate local coordinate frames. In this implementation, `rel_dir` is the bearing back to the observer in the *target's* frame. For observer A seeing B, the heading difference is `wrap(angle + pi - rel_dir)`. Combine that with distance and bearing to relate their local maps. Only connected observations or matched shared landmarks establish such relations; do not assume all five agents see each other. Axis-aligned rock edges also constrain heading modulo right-angle rotations, while recognizable world boundaries can anchor position and orientation. Fruit positions add useful geometry even though fruit observations have no IDs.

### What biomes add, and what is actually visible

The public `biome` field is a **point sample under the agent**, not a nearby biome map, boundary description, Voronoi site coordinate or rendered pixel color. Five agents provide five labeled samples. As they move, record biome changes together with reconstructed movement and geometry: a transition constrains a boundary to the traversed segment, subject to movement and localization uncertainty. Repeated samples inside the same region are strongly correlated.

The native map has ten random Voronoi sites assigned four land-biome types, then a river mask overwrites part of the map. Site coordinates and type selections occur at the beginning of initialization, before the expensive rendering and rock generation. That makes reconstructed biome geometry an interesting possible route to constraints on early RNG outputs. However, neighboring sites can have the same type, rivers hide underlying regions, and the API does not expose the sites or their generation order. Observing a biome label is not equivalent to obtaining the random draw that selected a site's type. See [biome.py](../src/elements/biome.py), `Map_generator.generate`, and [environment.py](../src/elements/environment.py), constructor.

Five labels have at most `5**5 = 3,125` possible ordered patterns, or about **11.61 bits of distinguishing capacity**. Actual information is lower when labels are biased or correlated. Therefore the labels alone cannot generally identify one seed among all 2^32 possibilities. Trees and terrain are also correlated because biome affects tree acceptance; do not add their information as if independent. Biome-specific movement penalties confirm terrain and constrain motion, while all five native biome classes currently use the same baseline energy drain.

ID-indexed starting biome labels are cheap to compare once a candidate's starting agents are known. That comparison does **not** automatically avoid the rendering prefix: candidate agent positions are generated after it. The [follow-up experiment](seed_recovery_data_budget_2026-09-19.md) demonstrates a different approach: recover world coordinates from public boundaries, then query the candidate's early land-biome generator at those coordinates, without generating candidate agents.

### Native experiment: biome labels plus rounded rocks

Added [seed_start_information_probe.py](../scripts/seed_start_information_probe.py). It generates the same 16 candidate seeds and independently generates three target frames. The matcher receives JSON-round-tripped public biome labels associated with agent IDs, plus the earlier rock fingerprints. The simulator is unmodified.

| Target seed | Biome-only catalog matches | Rock-only matches at one decimal | Combined matches at one decimal |
| --- | --- | --- | --- |
| 3 | 3 | 3 | 3 |
| 11 | 11 | 11 | 11 |
| 20260919, outside catalog | None | **5, incorrect** | **None, correctly rejected** |

All 16 catalog entries had distinct starting biome patterns. This small-catalog result demonstrates added filtering, not general seed inversion or a population-wide uniqueness rate. At full coordinate precision, rocks already identified both known targets and rejected the outsider; adding biomes preserved those results.

Across generated frames, all six founder traits were constant and all ages were 0.1. Energy values were 149.9 or 169.9, consistent with no pickup or one initial 20-energy fruit pickup. The 16 candidate frames contained only three directed other-agent sightings in total, illustrating why inter-agent alignment cannot be assumed available immediately.

Measurements and source hashes: [seed_start_information_probe_2026-09-19.json](seed_start_information_probe_2026-09-19.json). Reproduce from `survival-simulator`, with a fresh filename:

```text
python scripts/seed_start_information_probe.py --output docs/seed_start_information_probe_repeat.json
```

This follow-up tests biome filtering, not a complete joint geometry/seed solver. That solver should retain all public fields, use fixed traits as model parameters, combine seed-dependent observations without double-counting correlated evidence, and validate any solution against the entire public frame and subsequent known actions.

## Why arbitrary seed recovery is harder

**Brute-force enumeration:** The fallback seed space contains 4,294,967,296 possibilities. At the measured native-generation average, enumerating the whole space would take approximately **240 CPU-years sequentially**. That is a simple extrapolation of the current generator, not a lower bound: an optimized generator, candidate filtering, precomputation or parallel hardware could reduce it. Native brute force is nevertheless unsuitable for identifying a new seed during a game. Its costly prefix occurs before the first rock, preventing an early rock mismatch from avoiding most startup work.

**Direct RNG-state reconstruction:** Python uses deterministic Mersenne Twister, not a cryptographic generator. That makes state inference possible in principle, but does not supply an immediate decoder for these observations. [Python's random documentation](https://docs.python.org/3.12/library/random.html).

CPython's `random()` combines 27 high bits from one generated 32-bit word and 26 from the next. Thus a reported geometric float is not a complete 32-bit generator output. `uniform()` transformations and observation-coordinate arithmetic introduce additional rounding. The classic operation of undoing MT's output transformation on full known-order words does not apply directly to a few observed rocks or tree births. [CPython 3.12.12 generator source](https://github.com/python/cpython/blob/v3.12.12/Modules/_randommodule.c).

Even the idealized complete, correctly ordered 80-rock block contains only `80 × 4 = 320` calls to `random()`, providing at most `320 × 53 = 16,960` output bits. That is less than MT19937's 19,937-bit state dimension, so this block alone cannot uniquely determine an unrestricted generator state by information count. This is an upper-bound argument: the agent actually sees less, with unknown rock-generation order. It does **not** rule out identifying a 32-bit original seed, whose possible starting states form a much smaller set.

A specialized solver might combine partial output bits, geometry, possible draw positions, additional tree/agent evidence and the integer-seeding constraints. Such a solver was not implemented or timed here. It is a research project, not an established shortcut justified by MT's lack of cryptographic security.

### Alternatives that do not require a seed catalog

**A candidate pool is not fundamentally necessary.** The catalog probe establishes that the observations distinguish maps, but does not test the fastest possible way to infer their seed.

- **Algebraic seed inversion:** Published Python-specific work reduces a 32-bit seed to two possibilities from six selected, complete 32-bit outputs in the first 624-word output block; further evidence distinguishes them. Its demonstrated positions include 0, 1, 2, 227, 228 and 229. This avoids enumerating the 32-bit seed space. The assumptions matter: these are not six arbitrary observed rock dimensions, and the native rocks occur millions of draws later. [Author's analysis](https://stackered.com/blog/python-random-prediction/) and [proof of concept](https://github.com/StackeredSAS/python-random-playground/blob/main/recover_32bitSeed.py).
- **Linear state reconstruction from partial outputs:** MT's state transition and output transformation are linear over bits. Sufficient known output bits at known draw positions can therefore become equations for the internal state, without naming any seed candidates. An implementation demonstrates recovery from truncated outputs. The native problem additionally requires recovering draw order, handling missing events, and expressing floating-point geometry as intervals rather than assuming exact original random fractions. [Original implementation](https://github.com/fx5/not_random).
- **SAT/SMT constraints:** A solver can represent unknown bits and seed/state relations, with additional hypotheses for observation-to-draw assignments and skipped draws. Existing projects demonstrate SMT attacks on MT and other generators. Their existence does not establish acceptable runtime for the simulator's millions of hidden, branch-dependent initialization draws. [RNGeesus project](https://github.com/deut-erium/RNGeesus).

The most useful next experiment would test partial-output state recovery with a controlled known draw order, then progressively remove that assistance until only public observations remain. This would separate insufficient information from solver limitations and unknown ordering. For future random-number prediction, recovering the current RNG state can avoid recovering the original seed entirely, although the event-to-draw synchronization problem described below remains.

## Can recovered seeds predict future tree generation?

Recovering a seed would reveal the initial complete map for the matching simulator version and settings. Predicting later trees also requires the **current RNG position and simulator state**.

New tree creation is particularly difficult to infer from isolated events:

- The global spawn test depends on time and current tree count.
- Placement can consume extra position draws on collisions.
- A tree attempt can be rejected according to an unseen biome.
- Trees grow, die and produce fruits in list order; many events are outside every agent's sensing range.
- Births and predator behavior consume the same RNG. Changed actions can shift later tree events.
- A tree first becoming visible is not evidence that it was born on that tick. Public tree observations have no ID, age or birth timestamp.

Consequently, use a recovered seed to initialize a private simulator and replay **all actual actions** from the start, verifying its predicted public observations against each incoming frame. Drop or resynchronize the hypothesis on disagreement. Different engine versions, settings, numerical behavior or entity iteration order can break an otherwise correct seed hypothesis. A seed match alone does not establish long-horizon prediction fidelity.

## Practical recommendation

Proceed with a **bounded candidate recognizer** if there is evidence that the seed range is small, maps repeat, or the candidate list is otherwise known. Index rock lengths, preserve observation precision, and verify the full geometry before accepting a match. Initial tree spacing is a useful independent confirmation. This is the route demonstrated here.

For an unrestricted fresh 32-bit seed, prioritize ordinary map reconstruction unless there is time for a separate solver project. The first milestone for that project should be recovering a deliberately hidden seed from public observations **outside a pre-generated catalog**, with a measured runtime and false-positive test. Future-event prediction would then need a separate action-replay synchronization test.

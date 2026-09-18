# Investigator 4: predator transitions, targeting, and gaze steering

Tested locally on 17 September 2026 against the unmodified vendored simulator at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. No remote API, evaluation, submission, publication, or vendor edit was used.

Run the complete probes from the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/04_predator.py
```

The complete evidence is in `survival/results/mechanics_hunt/04_predator.json`. Use `--skip-generated` for the fast isolated assertions only.

## Ranked findings

### 1. A tiny gaze offset can divert a far predator, but the generated-map advantage is not established

At distance at least 90, a predator treats an agent looking at it differently from one looking away. There is an exact-facing singularity: when the relative agent gaze is exactly zero, `sign(0)` is zero and the nominal pivot branch moves straight toward the agent. A relative gaze of only `+1e-9` or `-1e-9` makes the predator request a 45-degree lateral move in the corresponding direction. At exactly `±pi/2` it still pivots; immediately beyond that boundary it direct-chases.

This mattered in an arranged, flat-forest walking-retreat probe driven only by cached public observations:

| Gaze rule while walking away | Result | Minimum center gap | Energy at end |
| --- | ---: | ---: | ---: |
| Face predator exactly | Eaten after 3.8 s | 10.0 | 127.2 |
| Fixed +0.02 rad offset | Alive at 9.0 s horizon | 162.6 | 95.4 |
| Alternate ±0.02 rad | Alive at 9.0 s horizon | 162.6 | 95.4 |
| Alternate ±0.20 rad | Alive at 9.0 s horizon | 162.6 | 92.1 |

The agent had ordinary founder traits and 150 energy, requested only walk speed 10, and issued one action per tick. The policy used predator `distance` and `angle` from the ordinary cached observation, not current coordinates, energy, rest state, or a map. Initial positions, headings, flat terrain, absence of obstacles, and removal of random spawns were arranged fixture advantages. This is a bounded steering probe, not the excluded dedicated sprint-decoy experiment.

The negative control is important: a stationary agent died after 1.6 seconds in all four gaze variants. The offset helps only when movement preserves the far-range geometry.

Generated-map validation used default 1600×1200 maps, ordinary cached-observation actions, realistic starting energy, one explicitly requested default-energy starting predator, 60-second horizons, and seeds 1–9 plus 42. Seed 2 had no starting predator because its one-shot placement failed. Across the nine comparable seeds:

| Walking-retreat gaze rule | Total agents lost | Mean score |
| --- | ---: | ---: |
| Exact | 9 | 58.932 |
| Fixed +0.02 | 9 | 58.980 |
| Alternate ±0.02 | 8 | 58.981 |

Alternate gaze saved two agents relative to exact on seed 1, lost one extra on seed 42, and did not change deaths on the other seven seeds. This is not a demonstrated general score advantage. The mechanic is proven and cheap enough to try as a small modifier inside a competent escape policy, but the simple standalone response should not displace normal obstacle-aware escape logic.

### 2. Depletion does not cancel the predator's last lunge

The active predator sleep check happens after movement and after contact kills. In the positive fixture, an active predator started at 0.1 energy, spent to -0.45 while moving, contacted an agent, gained the agent's post-passive-drain energy, ended at 149.45, and stayed active. The distance-40 negative control made the same low-energy move without contact, ended at -0.45, and went to sleep.

This is an original-engine ordering effect, but the fixture sets hidden predator energy and active state. Player observations expose neither, so its practical use is defensive: never treat an apparently exhausted predator's next lunge as cancelled. It follows the one-action-per-agent contract.

### 3. Rest wake-up is strict, delayed, and exactly measurable only with hidden state

This reproduces the known sleeping-contact result and extends its boundary:

- A resting predator at energy 99 or exactly 100 recharges by 3 and skips the entire action/contact pass.
- A resting predator at `100.000001` wakes and acts in that tick.
- Starting at zero while overlapping an agent, it completes 34 harmless ticks (3.4 seconds), reaches energy 102 while still resting, then wakes and kills on tick 35.

Every successful generated starting predator also first woke on tick 35. Resting and energy are absent from the public payload, so a controller can only infer a possible rest interval from repeated stationary observations. Collision or wandering can imitate immobility, predator observations have no ID, and an unseen predator's recharge start is unknown. Treat 3.4 seconds as an engine measurement, not a safe public countdown.

### 4. Target choice has no commitment, while exact ties inherit unstable ordering

`Predator.step` chooses the minimum-distance observed agent every active tick. Swapping two synthetic distances from 40/41 to 41/40 immediately reversed the chosen movement direction. Equal distances select the first item in the observation list. Reversing a tied list reversed the target; adding a `1e-6` distance advantage selected the closer target regardless of list order.

The environment supplies agents from a set-derived local collection, so the tie is not an agent-ID rule that a controller can rely on. Deliberately becoming measurably closer can switch a predator; engineering exact ties cannot assign a stable decoy. These target-selection probes use synthetic observation lists and do not themselves constitute a player action tactic.

### 5. One active predator can kill an entire overlapping pile

Three overlapping agents were all killed in one active contact pass, and the predator filled to its 200-energy cap. With the same three agents and a zero-energy resting predator, all survived that tick and the predator charged to 3. This uses arranged overlap, but otherwise original interaction code and one idle action per agent. Practical implication: clustering does not limit a predator to one victim per tick.

### 6. Predator spawning is a one-shot placement attempt

An explicitly obstructed `spawn_predator` call returned `None`; a free-position control succeeded. The method does not retry. In the full generated-map runs, seed 2 requested one starting predator but produced none because the sampled position overlapped an obstacle. Timed random spawn opportunities can likewise be wasted.

This can reduce realized predator pressure, but players cannot choose the sampled point or observe a failed spawn. It is a world-generation fact, not a strategy.

## Coverage and rejected hypotheses

| Hypothesis | Result | Classification |
| --- | --- | --- |
| Sleeping contact is a durable refuge | Rejected: zero-energy overlap dies on wake tick 35 | Known effect extended; hidden-state fixture |
| Depletion cancels an active predator before contact | Rejected: contact/feed precedes sleep check | Proven ordering hazard; hidden energy fixture |
| A stationary gaze offset is enough to survive | Rejected: all stationary variants died in 1.6 s | Contract-compliant negative control |
| A slight gaze offset changes far pursuit | Proven in branch signals and 9 s walking fixture | Contract-compliant, ordinary observations, arranged world |
| The simple offset improves generated-map outcomes reliably | Not established: 9 vs 9 vs 8 total losses on nine comparable seeds | Real maps, ordinary actions, small 60 s sample |
| Predators keep a target until it is lost | Rejected: one-unit nearest swap switches immediately | Synthetic decision probe |
| Equal-distance ties consistently favor an agent ID | Rejected: first observation item wins, and environment order comes from a set | Not a reliable tactic |
| One contact pass kills at most one agent | Rejected: three overlapping agents all died | Arranged overlap fixture |
| Requested predator count is guaranteed | Rejected: obstacle collision silently wastes the one attempt | Generated-world effect, not controllable |

## Boundaries

- The vendored files were not changed. Synthetic fixtures call the original sensing, movement, energy, targeting, contact, and tick methods.
- Flat fixtures remove fruit, trees, extra predator spawning, and obstacles and explicitly label those advantages. Generated-map runs retain original terrain, obstacles, trees, fruit, spawning, and physics.
- All environment-step tests obey the documented one-action-per-living-agent rule. No duplicate IDs, predicted newborn actions, malformed requests, or multi-action input quirks are used.
- Diagnostic code reads hidden predator energy/rest state to measure transitions. The gaze policies do not use those fields.
- The generated comparison adds one requested starting predator to create matched early pressure; the upstream default starts with zero. One requested spawn failed, which is reported rather than replaced.
- These probes do not test wall acquisition, water baiting, dedicated sprint decoys, predator banishment, or replay rendering.

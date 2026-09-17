# Dedicated sprinter decoys and separate predator banishment

These are two different strategies. Neither requires breeding or starving the entire population. Tests use ordinary walk-10/sprint-20 agents and the unmodified upstream predator AI, movement, collisions, energy, sensing, and tick order. Source commit: `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`.

## What was tested

142 controlled runs across seven groups, including two playable replays. The worlds use the normal 1600×1200 dimensions, flat forest, physical boundary walls, arranged agents, one predator, and no new tree/predator spawns. Food cases have four fixed, already mature fruits at `(900,600)`, `(1060,600)`, `(1220,600)`, `(1380,600)`. They are finite resources, not energy injections. The predator starts awake at a specified energy, rather than at its usual newborn resting state. Ages start at zero with a normal-range senescence threshold of 90 seconds.

The direct decoy and relay controllers consume actual cached observations and reported energy. Adaptive sprint estimates use self-motion integration, which becomes inaccurate during wall sliding; this is recorded as a limitation. Banishment additionally receives the bait's true pose and an assigned destination so these tests address relocation feasibility, **not** solved map acquisition or localization. No controller reads hidden predator energy or resting status. Diagnostic measurements do read those fields. None of these are complete generated-map or remote validation runs.

## 1. Dedicated sprinter decoys: viable bursts, costly occupation

The useful rule is to keep one prepared bait close enough to remain the selected target, sprint only when the gap shrinks, and reduce the requested speed as the predator slows. Looking direction and movement are independent, so the bait faces its pursuer while moving away.

Source mechanics explain the constraints:

- An ordinary bait can already request 20 while a predator requests at most 15. Breeding speed is unnecessary for establishing the idea.
- A 150-energy bait with max energy 500 has only 50 energy above the sprint cutoff. Requests 15.1, 16 and 20 cost respectively 31.5, 36 and 56 energy/second including young passive drain, before turning. Once it falls below 100, a sprint request becomes walking at 10.
- Predators slow to 11 below their own 40-energy cutoff, then eventually rest. In the straight controlled pursuit, an initially 102-energy predator has about 9.5 active seconds followed by about 3.4 resting seconds. There is an opportunity to request about 11.1 during its slower phase, at only 11.5 energy/second, and stand during its rest.
- Reported predator positions precede its most recent movement. In a direct approach, a decision to wait can permit two 15-unit advances relative to the cached distance. Because capture occurs inside 15 units, a trigger of 45 or less has no margin. Triggers 30 and 40 failed quickly; 60–90 gives a more useful initial buffer. Walls, multiple predators and changing terrain invalidate a simple one-dimensional safety guarantee.

The ordinary pulse requests 16 only when the cached distance is below 60. The adaptive version estimates movement from consecutive observations and requests the observed predator speed plus 0.1, bounded by available movement. It waits when separation is sufficient. In the initial arranged encounter:

| Initial bait energy | Finite food | Ordinary pulse lifetime | Adaptive lifetime |
| --- | --- | ---: | ---: |
| 150 | None | 2.2 s | 2.2 s |
| 150 | Four mature fruits | 14.2 s | 21.0 s |
| 500 | None | 20.0 s | 28.1 s |
| 500 | Four mature fruits | 26.4 s | 39.1 s |

The 39.1-second run kept the bait selected for 99.7% of active predator ticks, caused three predator rests, and consumed about 481 movement energy plus turning and passive drain. It eventually crossed the sprint cutoff and was captured. It also spent much of the run following the boundary, so this is an arranged demonstration, not a general safe duration. Its four consumed fruits offered 240 nutrition, but some was wasted at max-energy capacity.

Five untuned heading, starting-gap, predator-energy and random-seed perturbations gave these medians with 500 starting bait energy:

| Policy | Median bait lifetime | Survived 60 s | Mean active target fraction |
| --- | ---: | ---: | ---: |
| Pulse 15.1 | 24.3 s | 1 / 5 | 68% |
| Pulse 16 | 24.2 s | 1 / 5 | 68% |
| Pulse 20 | 42.1 s | 2 / 5 | 48% |
| Adaptive | 31.2 s | 1 / 5 | 70% |

Survival alone is misleading: the full-20 bait often lived longer by losing the predator's attention. That does not establish occupation. Adding the same four fruit positions did not change these perturbed runs because the trajectories missed the fruit. A decoy needs actual food-route planning or a naturally suitable patch; food merely being nearby is insufficient.

### Relays and replacements

A reserve cannot stand in the predator's future path. The revised two-bait prototype keeps the reserve farther away, switches the active role when its energy falls below 200 and the reserve has at least 50 more energy, and asks the outgoing bait to open a larger gap. This is one simple heuristic, not an optimized relay.

- Two 150-energy baits: first captured after 2.2 s without food; final bait survived 22.6 s, but predator attention was held only about 10% of active ticks. This is escape, not a successful sustained relay.
- Two 500-energy baits: one handoff occurred, one bait was captured around 28 s, and the other was still alive at 45 s. They held attention only about 56% of active ticks without food. This did not establish continuous occupation either.
- Four fruits prolonged the first 150-energy bait's role, but did not make the pair a reliable renewable trap.

Reinforcements need to be positioned and fed **before** the current bait loses sprinting. A default newborn has only 75 energy and cannot sprint. Lower capacity does help at equal current energy: with 150 current energy, reducing max energy from 500 to 150 extended adaptive lifetime from 2.2 to 7.7 s in this encounter. A 75-energy newborn with capacity 150 lasted only 2.1 s, versus 0.6 s at capacity 500. Lower capacity is a possible specialist trait, not a substitute for feeding. Conversely, a larger capacity can hold more fuel if actually filled; it is not inherently harmful.

**Practical assessment:** prepared specialist baits can occupy one predator for useful tens-of-seconds windows. The tested relay is not yet a sustainable one- or two-agent service. The best next improvement is an observation-based food circuit and earlier coordinated handoff, evaluated by target retention and surviving workers, not bait lifetime alone.

## 2. Predator banishment: drag, break contact, then monitor return

This strategy deliberately releases the predator far from the food colony. It is not a decoy attempting permanent occupation.

The prototype leads east, waits until two observed positions indicate immobility, then walks beyond 310 units from that resting location along a clear exit. Using the resting window avoids paying for a long full-speed escape. The pose and exit clearance are privileged controlled inputs in this prototype.

In the representative full-energy, food-free case:

- Predator began at `(700,600)`, with two stationary colony agents around `(380,630)`.
- Bait dragged it roughly 860 units east.
- At 9.8 s the bait recognized immobility and left during rest.
- Bait remained alive through the 100-second test. In the 60-second replay it retained about 169 energy, having spent 271 on movement, about 0.3 on turning, and 60 on living.
- The predator returned within 250 units of the colony at 34.7 s, **24.9 s after release**, and subsequently killed both stationary colony agents.
- In the matched no-bait control it first entered the same radius at 23.5 s. Thus relocation bought about 11.2 s of initial grace in this case, but did not improve eventual stationary-colony survival.

There is no stationary “banished” state. After losing prey, a predator wanders at 11 units per tick, steers around edges, and periodically rests. It has no obligation to remain in the destination corner. A corner can also steer its wander back toward the colony.

The five additional paired scenarios did not show robust protection. For example, first colony approach changed from 67.8 to 24.4 s in one case and from 33.9 to 24.5 s in another, both worse. In two cases the bait died before achieving the planned release. A predator initially facing the colony attacked it before the distant bait successfully acquired attention. Nearby worker distractors also disrupted the planned route: when a worker became the closest perceived prey, the predator could switch to it and the longer relocation target was never reached.

**Practical assessment:** temporary displacement is real; permanent cheap parking is not demonstrated. Treat banishment as a way to buy a measured evacuation/harvest window, preferably into an already isolated region. It needs reacquisition or a guard, and should be compared against what the predator would have done without intervention. Sacrificing a bait also gives its remaining energy to the predator, potentially lengthening the next active period.

## Artifacts and reproduction

Run the complete bounded suite:

```sh
survival/.venv/bin/python survival/research/sprinter_decoys.py --phase all
```

Individual phases: `screen`, `adaptive`, `capacity`, `heldout`, `banishment`, `banishment-heldout`, `replays`. Each writes `survival/results/sprinter-decoys-<phase>.json` with source commit, script hash, parameters, energy accounting, target retention, deaths, rest cycles, boundary exposure and traces.

Playable replay files, compatible with the existing Survival Lab viewer's file loader:

- `survival/results/sprinter-decoys-adaptive-replay.json.gz`: prepared decoy, four finite ripe fruits, adaptive speed, eventual capture.
- `survival/results/sprinter-decoys-banishment-replay.json.gz`: successful drag and escape, followed by predator return and colony losses.

The viewer and its manifest were not changed. No external validation, submission or deployment occurred.

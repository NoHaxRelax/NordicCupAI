# Continuity and colony energy experiments

Goal remains near-universal 3000-second survival with mean score at least 2500,
no bait gaps, almost all predator victims being guides, high agent energy and
mostly ripe fruit consumption. None of the experiments below proves that goal.

All runs use seed 1883894846 and the pinned native C++ engine, ordinary energy,
native spawning, and observation-only policy inputs. No paid compute was used.
These are development runs on one repeatedly used seed, not a validation set.

| Variant | Seconds alive | Score | Estimated bait gap seconds |
| --- | ---: | ---: | ---: |
| Route fixes, overlap 20 | 1015.7 | 1058.04 | 242.1 |
| Route fixes, overlap 60 | 599.8 | 633.41 | 75.0 |
| Small high-energy colony, overlap 60 | 222.5 | 231.14 | 0.1 |
| Reserve donor at 60, dispatch overlap 20 | 1015.7 | 1058.04 | 242.1 |
| Reserve donor + Oscar r21 settings | 711.7 | 737.89 | 169.6 |
| Separate trap roles from Orchard workforce | 458.0 | 483.07 | 14.5 |

The high-energy variant consumed six of its twelve agents as bait and only
produced twelve agents total. Its only predator victim was a guide. Near-zero
gaps until early extinction are not success. Donor reservation made no outcome
change with the existing settings on this seed: viable candidates tend to
appear only when dispatch is already due. Neither experimental setting file
has been promoted to the default configuration.

Oscar r21 settings are copied from `no_predators_best` in
`origin/survival-simulator/oscar-overnight-cpp` at `02d185b`, file
`survival-simulator/oscar-overnight-cpp/configs/best-configs.json`. His reported
2599 mean score without predators does not transfer to this integrated policy.

The recorder now reads native `pop_events()` **only for evaluation**, recording
actual starvation/age-energy deaths, predator deaths, fruit age at eating and
time-weighted average agent energy fraction. Role attribution uses the action
that preceded the death, rather than the next policy decision. Full events are
in replay chunks; compact counts are in `native_evaluation` in each summary.
Fruit age at least 20 seconds is counted as ripe. A guide disappearing is no
longer treated as proof that a predator ate it.

Overlap 60 with the original survival settings: 94 gatherer energy deaths;
15 guides, 6 avoiding agents and 1 replacement eaten; 90.9% ripe fruit;
mean energy 27.6% of capacity. R21: 119 gatherer energy deaths; 25 guides and
19 other agents eaten; 81.4% ripe fruit; mean energy 19.8% of capacity.
Energy depletion/ageing dominates deaths. Ripe-fruit timing alone is unlikely
to solve the colony failure.

Next integration correction: bait/guides previously retained tree and fruit
claims even though their movement was overridden. The optional Orchard
`unavailable_agents` hook retains their observations but releases those claims
and excludes them from productive population/birth planning. Core supplies
current bait, incoming bait, retired bait and guides when `--release-trap-food`
is enabled. It is **off by default** after the negative same-seed result above;
the structural correction also changes birth planning and needs finer ablation.
Reserved donors remain
normal gatherers with reproduction and guide reassignment suppressed. Newly
assigned roles release their claims on the next tick.

Replay folders are under `logs/entrapment-iteration/route-fixed-*20260919`.
Each manifest records source hashes, survival overrides and overlap/reservation
settings. Policy and food-allocation changes need independent-map validation;
the observed-position bait-gap and predator-proximity metrics still do not
prove physical retention or replacement access for every predator.

Further upstream audit found that our Python Orchard copy still lacked the
boundary-anchor fix implemented in Oscar's native `_npolicy.hpp` at `02d185b`.
The small port chooses the wall side from the observed signed offset, accepts
only positions inside the 30-unit boundary walls plus agent radius, and permits
large corrections when direct boundary evidence contradicts odometry. It does
not substitute hidden world coordinates. An isolated same-seed run records its
effect; two fresh-seed baseline runs (204871 and 917263) were started beforehand.

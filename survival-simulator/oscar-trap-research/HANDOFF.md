# Survival Simulator handoff: wall bait and narrow-gap refuges

Prepared 17 September 2026 for the NordicCupAI team. This is a snapshot of ongoing local research, against upstream simulator commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. The engine is included unchanged. Start with [README.md](README.md) for setup and the file map. Replay files are intentionally excluded from this branch; historical replay paths in copied reports refer to local-only recordings. [SNAPSHOT.json](SNAPSHOT.json) records the capture window and hashes of the supplied source files.

## Decisions to carry forward

**Both traps are real mechanics leads. Neither is yet a proven autonomous full-game strategy.** A predator keeps choosing an audible agent even when an obstacle prevents reaching it. A stationary bait can exploit this at low running cost. The hard parts are finding a site through ordinary observations, delivering a predator without losing control of another, and feeding/replacing the holder while new predators spawn.

- **Wall bait has the most developed renewal controller.** An observation-only crew maintained an already arranged trap for 300 seconds on two naturally generated maps, with ten paid births per site, five/six descendant generations and zero captures when extra predator spawning was disabled. With native spawning, results were mixed. Six ordinary game starts yielded zero acquisitions in 180 seconds.
- **Narrow-gap refuge has strong short-horizon holding evidence and cheap entry.** A radius-5 agent fits where a radius-10 predator cannot. All 38 arranged single-predator native-site runs held for 60 seconds across 19 maps. Four native three-predator tests also held. Discovery during normal play, approach during arbitrary pursuit, feeding and replacement are unfinished.
- **These are bait-maintained traps, not permanent cages.** A bait that leaves, dies, loses attention or is displaced by a closer agent can release the predator. Gap withdrawal controls explicitly demonstrate escape.
- **Do not invest in more holding-only sweeps before solving acquisition.** Reuse the maintenance work, add a scout and a verified delivery route, and test the complete sequence against the same colony controller without trapping. Keep one predator per wall as the initial target; stacking is still unreliable.

## 1. Engine interactions that explain both traps

The source is in [survival/vendor/survival-simulator/src](survival/vendor/survival-simulator/src). The most useful files are `elements/predator.py`, `elements/creature.py` and `elements/environment.py`.

| Mechanic | Verified behavior | Consequence |
| --- | --- | --- |
| Target choice | Each active predator tick selects the nearest currently observed agent. There is no lasting commitment to an agent. | A guide, forager, newborn or bodyguard can unintentionally become the target. |
| Detection | Predator hearing/smell is omnidirectional to 60 units, including through obstacles. Beyond that, vision is directional, occluded, and has 250-unit range. | Hearing across a wall sustains pursuit even when vision is blocked. Losing hearing can end containment without killing the bait. |
| Asymmetric sensing | Normal founders hear to 50 units; predators hear to 60. | A predator can remain attached to a bait that cannot currently sense it. Absence from observation does not prove escape. |
| Geometry | Default agent radius is 5; predator radius is 10. Contact capture is below 15 units of center separation. | A passage can admit the agent and reject the predator, but the bait must also be deep enough to avoid contact at the mouth. |
| Steering | Direct chase turns at most 0.3 radians per tick. The predator pursues locally rather than planning a complete obstacle-avoiding route to bait. | A long wall can keep it oscillating against the near face; an end or corner can let it escape. |
| Speed | Per 0.1-second tick: agent walk 10/sprint 20; predator walk 11/sprint 15. | Default walking is insufficient for an open-ground chase, while sprinting consumes a finite reserve. These are distances per tick, not per second. |
| Energy and rest | Predators have no passive drain. Depletion causes rest, then recovery of 3 energy per tick until energy exceeds 100. Resting predators skip contact kills. New predators spawn resting at zero energy. | Exhaustion is temporary, not a way to kill a predator. Fresh rest creates roughly a 3.4-second setup opportunity. |
| Sprint eligibility | Sprint is disabled below 20% of maximum energy. A normal agent cutoff is 100; newborn energy is 75. | Newborns cannot immediately sprint away from danger. Plan walking-safe replacement routes. |
| Holder costs | A young stationary default agent pays about 1 passive energy/second. Hidden old-age thresholds are around 60–120 seconds; after the threshold, additional drain is `0.01 × age` each tick. | Standing still is cheap only while young. Feeding and renewal are essential for a long game. |
| Reproduction | A parent with more than 100 energy pays 100; a child starts with 75 and native randomized traits/age limits. Placement is generally 10–30 units from the parent, with engine fallbacks. | Birth is lossy and does not itself place the child safely at the anchor. A child needs food to reproduce again. |
| Score | Elapsed seconds + gross fruit energy/1,000 − captured agent reserves/100. | Trapping has no direct reward. Judge it by colony survival, food and capture losses after paying scouts, holders, guides and guards. |

Ordinary observations have relative predator bearings/distances/orientations, but no predator IDs, energy or resting flag. Fruit observations lack identity and maturity. Observations are cached before predator movement in the world tick. Debugger views and fixture diagnostics expose more information than a legal policy receives.

## 2. Wall bait: geometry and why it works

Put the holder near the midpoint of the far face of a long thin rectangle. Keep the predator on the opposite face, close enough to hear the holder. The wall blocks direct travel while the audible target continually reactivates pursuit. The holder need not run once established.

The supported starting range is **roughly 30–35 units thick and at least 70 long**, with the holder about 5–7 units off its face. Treat this as a tested range, not a universal guarantee. Length, heading, offset, nearby obstacles and hearing margin all matter.

Early controlled results from [predator_control.md](survival/research/predator_control.md):

| Geometry | Result |
| --- | --- |
| Thickness 30, length 70 or 100, offsets 0/10/20 | 6/6 full 60-second holds; no holder movement; energy 150 → 90. |
| 30 × 30 | Failure within 0.5–0.6 seconds. |
| 30 × 50 | Initially held, then failed at 38.8–51.7 seconds. |
| Thickness 31, 32 or 35; length 70; heading/offset sweep | 9/9 holds at each thickness. |
| Thickness 37 or 38, same sweep | 6/9 and 3/9 holds respectively. |
| Thickness 40 or 50 | Bait often survived because the predator lost attention and wandered away. This is not containment. |

A wider wall protects the bait physically but can push the predator outside hearing. A shorter wall allows the local steering to take the predator around the end. More clearance is therefore not automatically safer for the purpose of holding a predator.

### Availability is encouraging; discovery is still a different task

The first generated-map survey found 46 candidate walls among 800 obstacles across ten maps. Of 92 candidate-side placements, 39 were invalid because required creature positions overlapped other obstacles/boundaries. **52/53 valid arranged placements held for 60 seconds**, and all baits survived. Nearby trees appeared at 28/53 sites and initial fruit at 30/53, but this did not establish a safe or sustainable food route.

A newer census in [scout-wall-survey.json](survival/results/wall_deployment/scout-wall-survey.json), seeds 101–150, found:

- Geometrically eligible walls on 50/50 maps; median five, range one to ten.
- At least one physically usable wall on **49/50** maps; median three, range zero to eight.
- Median six usable faces and four faces with nearby protected-side trees. Two maps had no such food face.

This census uses complete native geometry. “Usable” means the selected holder/predator positions are clear, not that a scout has discovered the site or that a chasing agent can reach it. Opposite faces are not independent walls. Wall availability looks much less worrying than sensing, localization and delivery.

## 3. Wall acquisition: successful and failed approaches

Detailed evidence: [wall_acquisition.md](survival/research/wall_acquisition.md), with the named JSON files under `survival/results/`. These early feasibility controllers know true geometry/positions.

### Running around the wall

The solo controller walked or sprinted around a rectangle with seven units of clearance. Only **12/96** training cases held for 60 seconds, and every success began with the bait already on the far side. A separate 32-case geometry/heading test had zero holds. More energy or sprinting did not fix the loss of attention while routing around the wall. This rejects that controller, not all possible routes.

### A guide and a pre-positioned holder

The guide leads the predator toward the near face while the holder waits on the far face. In the tested escape attempts, 28/144 training cases and 7/36 new combinations held for 60 seconds. **Every successful transfer sacrificed the guide.** Peeling away early pulled the predator around the wall; peeling late allowed capture. Continuously tracking the predator with the hidden holder did not solve this.

In the representative replay, the guide is caught at 0.9 seconds with 132.25 energy remaining. This feeds the predator and incurs a score penalty, so it is an expensive acquisition even when it works.

The deliberate-sacrifice follow-up used guide starting energies 20, 75 and 150. Each energy had **12/16 holds**, with all 16 guides lost. All tested widths 30–35 held; all width-38 cases failed. A depleted 20-energy guide was sufficient for that short arranged walking route. This supports considering a low-value guide, but does not establish a net score advantage or safe arbitrary delivery.

### Rest-window acquisition

A fresh resting predator beside a suitable wall gives time to get to the far face without sacrificing a guide. In 96 privileged arranged cases:

| Initial predator energy | Walking holds | Sprinting holds |
| --- | ---: | ---: |
| 0, about 3.4 seconds remaining | 10/16 | 10/16 |
| 45, about 1.9 seconds remaining | 10/16 | 10/16 |
| 75, about 0.9 seconds remaining | 5/16 | 10/16 |

All baits survived fresh/middle rest, including cases that merely lost attention. Two late-rest baits were captured in each movement group. Facing away could carry a waking predator out of hearing before it turned back. New geometry tests held 9/12: all eight widths 30/35, one of four width 38.

The later **observation-only** acquisition module uses 0.3 seconds of observed stationarity as a rest heuristic. In arranged corner-vantage tests, it held **3/4 fresh-rest** opportunities, **0/4 late-rest** opportunities, and rejected four width-38 cases. Eligible paths took 1.6–1.7 seconds; late rest ended around 0.9 seconds. Stationarity does not reveal how much rest is left, so it cannot justify a blind commitment.

## 4. Wall maintenance: food, births and handoffs

### One legal replacement without food

Across 96 runs combining controls and birth timings, tested replacements preserved containment with no captures. Birth at 48 seconds extended mean lifetime to about **120.9 seconds**, versus stationary-founder lifetimes of 71.9/96.0/122.4 seconds for assigned old-age thresholds 60/90/120. Birth at 40 or 45 seconds gave about 114.9/118.6 seconds. A replacement can avoid senescence costs but wastes 25 energy in the conversion and overlaps two passive drains. It can shorten the lifetime of a naturally long-lived founder. Hidden age thresholds prevent perfect timing.

This buys one replacement. An unfed 75-energy child cannot fund another generation.

### Privileged orchard feasibility

[wall_orchard.md](survival/research/wall_orchard.md) established food-supported repeated generations before attempting observation-only deployment. It used an already acquired trap, arranged forest/orchard, true coordinates and true fruit energy, fixed founder age limits and no new predators. Native births, fruit dynamics and later global tree spawning were retained.

The final active-cohort controller reached 300 seconds in **three of four new-tree-enabled** cases, including both held-out geometries. The other case lost containment at 210.3 seconds and died at 244.4. Successful cases made 7/11/10 births and 9/14/10 handoffs, with zero captures. This is feasibility evidence under favorable conditions.

The key implementation lesson was to distinguish **working population from total living population**. A total cap of three allowed retired elders to consume every birth slot. The final design allowed three active workers and six total agents, rotated young holders and retired elders. Initial trees start producing around age 20 and can die after 50; initial patches disappeared around 57–63 seconds. Fruit matures over about 20 seconds and rots around 50. A nearby initial orchard is a temporary resource, not a permanent food supply.

### Observation-only maintenance on natural terrain

The later [wall deployment report](docs/survival-wall-deployment.md) is the stronger maintenance evidence. Its controller receives native observations and public time, without world coordinates, fruit energy, predator rest state or hidden age threshold. It:

1. Establishes local coordinate frames, registers nearby agents/newborns from relative bearings/orientations and corrects movement estimates using observed obstacle edges.
2. Recognizes a possible wall and opposite face from edges, then keeps workers on a protected half-plane within 350 units of the anchor.
3. Tracks fruit spatially and estimates age from first observation, normally waiting 15 seconds, accepting younger fruit below 85 energy and avoiding deliberate collection near capacity.
4. Moves the incoming holder to the anchor before releasing the outgoing holder, funds native births and retires aging workers.
5. Withdraws when it recognizes a predator on the protected side.

The initial site and creature placements were still selected/arranged by an evaluator using hidden geometry and food ranking. Policy legality is not the same as unarranged setup.

| Natural seed, extra spawning disabled | Continuous hold | Captures | Paid births | Maximum descendant generation | Handoffs | Fruit eaten | Gross food energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 300 s | 0 | 10 | 5 | 8 | 130 | 6,781.4 |
| 8 | 300 s | 0 | 10 | 6 | 9 | 85 | 4,274.6 |

Each crew pays 1,000 energy in births and ends with four living agents. Gross food includes capacity waste and retirees; it is not a marginal trap cost. The 300-second endpoint is a cutoff, not proof of indefinite renewal.

Important fixes retained in the development history: obtaining a first native observation before movement; allowing a newborn inside the planner clearance margin to move outward; distinguishing equal-length opposite faces; registering colocated newborn headings correctly; and retiring old workers before all active slots fill. A heading-registration bug had created an apparent predator on the protected side and caused false withdrawal.

### Native spawning and full-game prefixes

Against the same colony controller with wall recognition/acquisition disabled:

| Arranged site | Wall survival / first hold loss | Control survival | Wall / control score | Wall / control captures |
| --- | --- | ---: | ---: | ---: |
| Seed 7 | 300 s / none | 300 s | 306.64 / 287.85 | 0 / 13 |
| Seed 8 | 286.9 s / 139.5 s | 300 s | 268.57 / 288.04 | 11 / 9 |
| Seed 14 | 69.6 s / 68.8 s | 300 s | 63.44 / 278.66 | 6 / 23 |

An extra awake predator deliberately placed about 200 units into the protected side defeated both tested wall crews. Withdrawal is not a reliable two-sided defense.

Six native five-agent starts, seeds 11–14 and fresh held-out 21–22, reached the 180-second cutoff with **zero recognized/attempted wall acquisitions**. Their survival and scores belong to the ordinary colony behavior, not a demonstrated wall contribution. A full game can last 3,000 seconds. Same-seed pairs also diverge in RNG use, births, food and interactions, so they are not fixed-event counterfactuals.

## 5. Newer interactions: delivering more predators and using a bodyguard

This section summarizes saved **provisional** fixtures, rather than a finished policy. Code and runs were still evolving during snapshot capture. These runs use exact geometry/creature positions, flat forest, 150-energy founders, fixed walls, no food/new spawning and a 35-second horizon. They do not establish performance from observations.

[protocol-delivery-v1.json](survival/results/wall_deployment/protocol-delivery-v1.json) contains four cases per approach:

| Delivery | Holds | Interaction observed |
| --- | ---: | --- |
| First predator, straight approach to empty trap | 4/4 | All guides sacrificed. |
| Second predator, straight approach to occupied trap | 2/4 | Existing predator targeted the guide in all four; it left containment in two. Successful cases were at width 30/length 70. |
| Second predator, corner approach | 0/4 | Existing predator changed target and exited in every case. |
| Corner approach while moving the holder to track | 0/4 | Tracking did not prevent target switching and escape. |

Do not treat four guide headings as four independent maps. Also, the harness success condition uses more than 95% containment in the final 15 seconds; continuous containment must be checked separately through `old_first_exit` and the trace.

Bodyguard tests place an extra predator on the protected side at distances 20, 70, 160 or 240, either awake or freshly resting. A stationary guard succeeded in **0/8** in both saved versions. Active v1 succeeded in 2/8 (the two distance-240 cases); the revised sideways-retreat v2 succeeded in **1/8** (awake, distance 160). These are changing controllers and cannot be pooled as one success rate. The v2 file records failures when the intruder starts too close and when early movement loses the needed target relationship. A guard can sometimes draw a predator away, but “keep two agents at every trap” is not yet a working defense.

The design implication is to treat trap delivery as a coordinated operation. Reserve the holder, mark the predator-facing side, and avoid allowing an incoming guide to become a closer reachable target for already held predators. Test occupancy and release of the old predator explicitly. For the initial integration, one predator per wall is the simpler goal; a shared trap needs an additional arrival protocol, not just enough space behind the wall.

### How many predators will need managing?

The native per-tick spawn attempt probability is `dt × elapsed_time × 0.0001 / max(1, predator_count)`. Placement may be rejected by obstacles. Predators persist through rest. The new census propagates this count distribution with spawn-placement acceptance integrated over 50 maps; **it is an analytical model, not simulated survival or a conditional-on-survivors statistic**.

| Elapsed time | Mean predator count | Median | 10th–90th percentile |
| --- | ---: | ---: | --- |
| 60 s | 0.13 | 0 | 0–1 |
| 180 s | 1.12 | 1 | 0–2 |
| 300 s | 2.46 | 2 | 1–4 |
| 600 s | 5.32 | 5 | 4–7 |
| 900 s | 7.96 | 8 | 6–10 |
| 1,800 s | 15.78 | 16 | 13–19 |
| 3,000 s | 26.14 | 26 | 22–30 |

These estimates assume the game continues to that time. A median of three usable walls per map therefore suggests that one-per-wall containment alone will not cover the late-game population. That is an inference motivating stacking, alternative refuges or selective displacement, not a demonstrated optimal strategy.

## 6. Narrow-gap refuge: a different geometry using the same bait principle

Full report: [survival-gap-refuges.md](docs/survival-gap-refuges.md). Implementation lives under `water_deployment` because this investigation grew out of the water-trap work: [alternative_traps.py](survival/research/water_deployment/alternative_traps.py) and [pocket_survey.py](survival/research/water_deployment/pocket_survey.py).

A physical gap **wider than 10 and narrower than 20 units** can admit a default agent while excluding a predator. The bait walks about 30 units into a long passage and then stands still. The predator detects it from the mouth and repeatedly tries to approach, but cannot enter. Unlike thin-wall bait, either obstacle can be thick.

The fixed site screen uses gap 11–19, overlapping passage length at least 55, clear entry and creature starts, both orientations and both entrances, including boundary obstacles. The one-unit margin excludes exact tangency. At gap 11, the radius-5 agent has only 0.5 units of center clearance on each side, making localization accuracy particularly important.

| Test | Observed result |
| --- | --- |
| Constructed gaps 12/15/18; lengths 60/90; headings ±0.3 | 12/12 full 60-second holds, no captures. |
| Too-wide control, gap 22 | 4/4 captured at 0.5 seconds. |
| Three predators at constructed gap | 2/2 full holds. |
| Native seeds 1–20, one selected eligible site, headings ±0.3 | 18/18 full holds across nine maps. |
| Fresh native seeds 21–40 with unchanged screen/controller | 20/20 full holds across ten maps. |
| Three predators at native seeds 1/7/8/9 | 4/4 full holds. |

The early eight dry-route native runs are a subset of the 18 development runs. Do not add them again. The two headings at each selected site also are not independent maps.

All successful rows maintained both predator proximity and detection of the bait in every post-acquisition check, including active ticks. Holding means all tested predators within 75 units of the entrance and all detecting the bait. This distinguishes containment from an agent merely surviving after disengagement.

### Availability and operating cost

The screen found eligible entrances on **19/40 maps**, with 36 entrance candidates in total: 15 on seeds 1–20 and 21 on seeds 21–40. Some are opposite ends of the same passage. Only nine maps had a full-speed entry route, but all 19 selected sites held across the tested headings, including slower swamp/desert terrain. The longest eligible passage was selected before outcomes were examined.

Entry costs were **2.25 movement energy on dry land**, or **2.81–4.5** on slower terrain. No further movement/turn energy was spent holding; the bait retained approximately 85.5–87.75 energy after 60 seconds from a normal 150-energy start. Passive drain and later senescence remain. The tested moving water pair cost about 5.8 total energy/second versus about one for a young stationary gap bait, but this comparison excludes scouts, replacement workers and food support.

These tests retain native obstacles, terrain and food dynamics but arrange starting creatures, disable extra predator spawning, and supply the mapped target plus initial pose. The controller dead-reckons using its public own biome. It does not use hidden predator energy/rest for action selection, but it is **not an autonomous site-discovery controller**. Zero odometry error in the selected straight-entry tests is not a localization guarantee during exploration or collision.

### Limits and negative results

- Withdrawing the bait at ten seconds released the predator from the mouth region at 13.3–13.4 seconds in two controls. One fleeing bait was later caught. No walk-away cage was established.
- Simultaneously holding three arranged predators is promising, but does not show safe sequential delivery, arrivals from opposite ends, arbitrary headings or later spawn locations.
- Foragers and replacements can become closer targets. Legal gap handoff, birth placement and repeated generations were not established in this namespace. Wall renewal results must not be credited to gaps without new tests.
- An exploratory one-way pocket idea tried to exploit chase step 15 versus untargeted walk step 11. A one-unit raster survey of 20 native maps found enclosed predator-clear pockets but no screened opening in the 11–15 range. A refined borderline opening was about 15.16. This finite search found no usable permanent prison; it is not a proof of impossibility.
- Agents and trees are non-solid, so they cannot form a body wall. Predator exhaustion only causes temporary rest.

## 7. Recommended teammate implementation sequence

1. **Keep the colony baseline and add an optional trap state machine.** Separate searching, site confirmation, holder placement, guide approach, acquisition confirmation, maintenance and abandonment. Track failures at each stage, rather than reporting only overall survival.
2. **Build observation-based scouting.** Reconstruct wall faces or a two-obstacle passage from observed edges. Confirm clear approaches, hearing margin, food access and a replacement route. Store map confidence and correct odometry at landmarks. The current wall groups do not merge established disconnected maps reliably.
3. **Solve one-predator acquisition first.** Use confirmed fresh rest where possible. Treat stationarity alone conservatively. If using sacrifice, favor a depleted guide and measure its actual reserve loss; avoid risking the breeding reserve.
4. **For wall maintenance, reuse the observation-only controller and its archived v7 baseline.** Preserve legal birth cost, replacement overlap, active/total population distinction and protected-side food policy. Test missing replacement recovery and two-sided threats.
5. **For gaps, first validate a real pursuit into a discovered passage, then handoff.** Three-predator occupancy is a reason to test staged arrivals, not to assume they are safe. Include opposite-mouth and nearer-worker controls.
6. **Evaluate on new generated starts with native spawning and the full horizon.** Compare against the same colony baseline, recording score, discoveries, rejected sites, acquisition rate, uninterrupted containment, captures, food, births, holder/guide/guard costs and extinction. Preserve every failure replay and the policy hash.

The strongest current conclusion is that **low-motion containment works, while discovery and coordinated acquisition dominate deployment risk**. A trap should remain optional until its complete use improves the colony's result.

## 8. Evidence, provenance and remaining work

| Read next | Purpose |
| --- | --- |
| [Wall deployment report](docs/survival-wall-deployment.md) | Observation-only maintenance, native-food renewal, spawning controls and zero-acquisition starts. |
| [Wall acquisition report](survival/research/wall_acquisition.md) | Solo routing, guide sacrifice, native sites, birth timing and rest windows. |
| [Orchard report](survival/research/wall_orchard.md) | Privileged multi-generation feasibility and population-cap lessons. |
| [Gap report](docs/survival-gap-refuges.md) | Gap widths, native-map tests, multiple predators, withdrawal and pocket search. |
| [Wall final summary](survival/results/wall_deployment/summary.json) | Compact v7 evidence with source/policy hashes. |
| [Wall experiment ledger](survival/results/wall_deployment/experiment-ledger.json) | Earlier versions, failures and interruptions. |
| [Mechanics investigation](docs/survival-mechanics-hunt.md) | Supporting mechanics and interaction checks beyond these two traps. |
| [Replay debugger](survival/debugger/README.md) | Native/state playback, per-agent inspection and future recording. |

The snapshot also includes other Survival strategies and result summaries so dependencies and context are available. Their presence is not an endorsement or a new summary of those strategies.

At capture, scout/guide/bodyguard development and replay backfilling were still active. Later local files are not automatically in this branch. Replay files and embedded replay data are not included in this handoff. Historical score-only files cannot reproduce an original trajectory from a seed alone. Where backfill exists, its receipts label footage as a **new reproduction**, sometimes with an archived controller and sometimes a disclosed current-controller substitution. Do not describe these as recovered original runs. Some old metric-only cases still lack footage; consult the included audit files.

All conclusions are local simulator evidence. No new competition validation, submission, deployment or paid compute was performed for this handoff. Many fixtures are selected, arranged and dependent; table fractions are not competition success-rate estimates. Native object-set ordering and differing RNG histories can change results across processes/platforms. Keep the recorded run, policy hash and exact setup when comparing changes.

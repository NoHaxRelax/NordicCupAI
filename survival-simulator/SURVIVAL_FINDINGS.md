# Predator-free survival investigation

Date: 18 September 2026.

Scope: the completed V10 experiments described here precede the later addition
of population target ceilings (12/4/2 at 0/900/1,800 seconds) and graphical speed
controls. The result tables below do not evaluate those later changes.

## Objective and status

The objective is to keep the species alive for the full 3,000-second episode,
prioritizing survival over fruit score. The investigation also tested coordinated
reproduction and a gradual shift from exploration toward conserving energy.

**Full-horizon survival is not established.** The earlier saved baseline and the
completed experiments all suffered extinction. Unit tests verify behavior; they
do not demonstrate that the policy survives the game.

## Current policy: completed five-seed results

The current policy, including the conservation ramp and population floors 6/4/2,
failed to reach 3,000 seconds on all five tested seeds.

| Seed | Survival | Final score | Survived full episode |
| --- | ---: | ---: | --- |
| 42 | 1,681.2 s | 1,761.64 | No |
| 1001 | 2,198.2 s | 2,340.14 | No |
| 1007 | 1,380.8 s | 1,437.69 | No |
| 1042 | 2,117.1 s | 2,211.33 | No |
| 2026 | 1,693.5 s | 1,769.82 | No |
| **Mean** | **1,814.2 s** | **1,904.12** | **0/5** |

On the three seeds shared with the saved original BO baseline, mean survival is
**1,898.7 seconds versus 2,065.0 seconds: an 8.1% regression**. Mean score is
1,996.39 versus 2,206.60, a 9.5% regression. The current collection of changes
therefore cannot be described as an overall improvement over that reference.

All five runs entered the shared-coordinate population phase within 5.7–21.4
seconds. All 606 recorded deaths were classified as energy depletion. These
results point to the later food/energy/generation cycle as the unresolved issue,
rather than a long delay establishing the shared coordinate system.

## Does the conservation dial help?

The clean three-seed comparison isolates the time ramp, keeping the same scanner
and all other policy settings:

| Seed | No ramp | Conservation ramp | Survival change |
| --- | ---: | ---: | ---: |
| 1001 | 1,667.6 s | 2,198.2 s | +31.8% |
| 1007 | 1,861.2 s | 1,380.8 s | -25.8% |
| 1042 | 2,099.8 s | 2,117.1 s | +0.8% |
| **Mean** | **1,876.2 s** | **1,898.7 s** | **+1.2%** |

Both variants had **0/3 full-horizon survivors**. The dial adds just 22.5 seconds
to the mean, with large opposing effects on two seeds. This is a small,
seed-dependent result, not evidence of a reliable survival improvement. Mean
score rises from 1,976.80 to 1,996.39, about 1.0%.

The idea demonstrably reduces optional action costs, but the tested schedule
does not solve survival. Its setting should remain experimental rather than
being treated as an established optimum.

Resource totals across those three completed episodes were:

| Metric | No ramp | Conservation ramp |
| --- | ---: | ---: |
| Fruits eaten | 7,975 | 7,840 |
| Pooled mean energy per fruit | 37.84 | 37.38 |
| Fruit energy collected | 301,808.2 | 293,058.8 |
| Births | 363 | 365 |
| Estimated total agent-seconds alive | 41,435 | 42,072 |

The ramp collected 2.9% less food energy while estimated agent-time increased
1.5%. These are not direct measurements of movement expenditure. Agent-time is
approximated from 60-second population samples, and final food totals depend on
episode duration. At the same 1,200-second checkpoint, ramp-on had collected less
food energy on all three seeds, while the population effect was mixed. Energy
saving and food discovery both matter.

The [detailed paired report](runs/survival-v10-no-ramp/paired-findings.md) includes
per-seed resource counts, matched-time results, and the calculation method. Only
this one conservation timetable was tested; its timing was not optimized.

## Does a larger early population help?

With conservation enabled in both cases, increasing the population floors from
6/4/2 to 12/8/3 produced:

| Seed | Current population floors | Larger population floors |
| --- | ---: | ---: |
| 1001 | 2,198.2 s | 1,765.7 s |
| 1007 | 1,380.8 s | 1,379.8 s |
| 1042 | 2,117.1 s | 1,081.4 s |
| **Mean** | **1,898.7 s** | **1,409.0 s** |

That is a **25.8% reduction in mean survival**, with 0/3 full-horizon survivors.
Mean score falls from 1,996.39 to 1,499.97. The larger floors should not be adopted
on this evidence; the current configuration remains at 6/4/2. Although wider
populations helped one earlier policy version, that benefit did not carry over
to this one.

Across the five current-policy episodes and the two three-seed comparisons,
**all 11 fresh episodes ended in extinction before 3,000 seconds**.

## Why energy becomes scarce

These values come from the local simulator, not assumptions about a typical
survival game:

- The time multiplier on new tree spawning halves every 300 simulated seconds
  (`0.5 ** (time / 300)`). The actual spawn probability also depends on existing
  tree count and biome. Trees themselves die, so a strategy that works during
  the abundant opening can fail later.
- Ordinary passive consumption is about one energy per second. A founder walking
  continuously spends about five additional energy per second; sprinting is much
  more expensive, at about 55 movement energy per second.
- Aging begins at an individual hidden age between 60 and 120 seconds. At age
  100, an affected agent spends about ten additional energy per second. Keeping
  an aging, well-fed parent alive can consume energy that could fund successors.
- A birth costs the parent 100 energy and gives the child 75. Reproduction transfers
  energy into a younger agent but loses 25 energy immediately and adds another
  agent's upkeep. More births are not automatically better.
- Fruit starts at 20 energy, gains about two per second, reaches roughly 60 after
  20 seconds, and rots after about 50 seconds. Immediate eating helps an endangered
  agent, but systematically eating fresh fruit reduces the food energy available
  to the colony.
- Movement is charged before biome slowing. A route through swamp or river can
  cost substantially more than its straight-line distance suggests.

Late failures were not simply a lack of fruit anywhere in the world. For example,
V10 seed 1007 still had 23 fruits when its final agent died; seed 42 had 25. Those
fruit counts are diagnostic simulator truth: the policy cannot see all of them.
The problem includes finding food, reaching it affordably, allocating it, and
maintaining the next generation before existing carriers run out of energy.

## Changes in the current policy

All decisions use public observations and the inferred shared map. Hidden fruit
ages, individual aging thresholds, and the complete world are used only in
evaluation records.

- **Travel and feeding:** evaluate affordable paths around observed rocks, include
  inferred biome costs, avoid repeatedly retrying blocked routes, and keep local
  greedy feeding from overriding central reservations.
- **Population planning:** estimate viable population 30 and 60 seconds ahead,
  use observed food supply, stagger births, and allow bounded overlap between
  generations. Healthy young agents retain feeding priority near their next meals.
- **Aging:** estimate extra consumption from public energy changes after accounting
  for actions. Confirmed aging remains remembered after a meal. Aging parents can
  transfer stored energy to needed successors instead of waiting for ordinary
  breeding cooldowns.
- **Rest and discovery:** rest near recently observed trees, leave unproductive
  trees after a bounded wait, and retain affordable short searches when distant
  survey destinations cannot be reached.
- **Evaluation:** added full-horizon, predator-free benchmarks with checkpoints,
  resumption checks, death details, food/population histories, and source/config
  fingerprints.

Several changes fix concrete affordability or navigation problems. Their combined
survival performance still has to be judged by the episode results, rather than
by the plausibility or complexity of the rules.

## The conservation dial

The new `harvest.conservation` settings are:

| Setting | Value tested |
| --- | ---: |
| Ramp begins | 300 seconds |
| Ramp reaches its final setting | 1,800 seconds |
| Optional scouting at the beginning | 100% of the time |
| Optional scouting at the end | 30% of the time |
| Scouting cycle | 10 seconds, staggered by agent |
| Stationary scan interval | 6 seconds initially, 18 seconds finally |

The ramp is linear. Optional head sweeping while patrolling also fades out.
Food collection, travel to observed orchards, river relocation, and predator
escape remain responsive. Breeding still runs while an agent rests. Pauses retain
the navigation destination and do not falsely accumulate time stuck.

Each uninterrupted stationary scan makes a full turn. If another task interrupts
it, a complete sweep restarts at the next stationary opportunity.

A bounded action-level check at the final setting measured 18 seconds of optional
walking and three scans per minute. Including ordinary upkeep, that is about
153 energy per minute, versus 370 for continuous walking and six-second scans.
This demonstrates the saving for optional scouting; it is **not** a measurement
of the whole colony's actual energy use. Real agents also collect food, wait,
reproduce, and age.

## Comparison method and limitations

The current experiments use fresh episodes, no predators, the same seed sets,
and a 3,000-second horizon. Extinction ends an episode early. Biome inference uses
deterministic simulation-time scheduling instead of adapting to wall-clock speed.
No interrupted or partially completed episode is counted as a final result.

The clean control retains the same scan implementation but moves the ramp to
3,000–4,500 seconds, keeping its intensity at zero throughout the episode. All
recorded state fields match the enabled run through 240 seconds on all three
comparison seeds, before the ramp begins at 300.

An initial control using `conservation.enabled=false` was abandoned: that flag
also restores the historical scan implementation, changing behavior before the
ramp. Its partial outputs are preserved under `runs/survival-v10-off` and are
excluded from conclusions. The toggle should eventually disable only the ramp
while retaining the same scan implementation; that cleanup was deferred when
this investigation was stopped for reporting.

The larger-population comparison changes only the early/middle/late population
floors from **6/4/2 to 12/8/3**. It tests all three floor changes together; it cannot
identify which one helps or hurts individually.
In particular, `survival_population` also affects desired young population and
near-extinction breeding/reserve conditions throughout the episode. This is
therefore not a pure test of scouting headcount or spatial coverage.

Three matched seeds can reveal large failures but are insufficient to establish
robust general improvement. The saved original baseline is a reference for the
whole policy, not a one-variable ablation of the new conservation dial.

## Earlier experiments

The following completed comparisons use the same three seeds: 1001, 1007, and
1042. They show that the larger collection of survival rules has not yet earned
its complexity through better full-game results.

| Policy / experiment | Mean survival | Survived 3,000 seconds |
| --- | ---: | ---: |
| Saved original BO baseline | 2,065.0 s | 0/3 |
| Original foraging combined with V7 reproduction/aging changes | 1,659.0 s | 0/3 |
| V7: successor funding, retirement, unknown-age wait protection | 1,768.1 s | 0/3 |
| V8: route-energy and allocation changes | 1,583.5 s | 0/3 |
| V8 with population floors 12/8/3 | 1,806.1 s | 0/3 |
| V9: revised population forecasts and nearby-meal protection | 1,662.5 s | 0/3 |

The larger population improved V8's mean survival by about 14.1%, but remained
below the saved baseline. A separate two-seed V7 comparison improved survival by
about 11% when compulsory swamp evacuation was removed: swamp is expensive to
traverse, but can contain a substantial part of a map's food supply. That finding
supports accounting for biome cost without automatically abandoning productive
terrain.

Restoring original foraging alone did not restore baseline performance when
combined with the newer reproduction and aging rules. These experiments combine
several changes and do not establish one universal cause of the regression.

There is also evidence that nutrition and discovery need attention. On seed
1007, the original saved baseline ate 2,226 fruits at an average 41.25 energy,
with 216 fruits rotting. V10 ate 1,756 at 32.40 energy, with 691 rotting. These
totals cover episodes of different lengths, so they are diagnostic observations,
not normalized estimates of a single rule's effect.

## Remaining opportunities identified during review

Two small public-observation reproductions exposed feeding opportunities worth
testing next. Neither has been shown to explain the full episode regression:

1. Fruit birth bracketing uses only the observer who first creates the track.
   If that observer just arrived, another observer's valid preceding empty view
   can be missed. This matters more now because unbracketed fruit is never held
   for ripening. Fuse the available observers' evidence before deciding that age
   is unknown.
2. The renewal-age cutoff can reject a short, safe ripening wait. A public test
   with age 54, energy 150, and known-fresh 52-energy fruit rejects a roughly
   1.9-second wait, although aging cannot begin before 60. Waiting would add about
   1.5 net energy after conservative upkeep. Evaluate safety over the actual wait
   instead of using that cutoff alone.

Both reproductions are saved as
[reproduction.py](runs/survival-v10/review-evidence/reproduction.py), with their
results in [output.txt](runs/survival-v10/review-evidence/output.txt). The relevant
code is in [harvest.py](src/utils/controllers/harvest.py), around
the fruit-observation loop and the ripening-wait eligibility check.

Further work should isolate small changes against a saved reference. In
particular, measure food energy obtained per agent-second, wasted movement,
birth costs, and young-agent starvation alongside survival time. A policy can
save energy while also discovering less food; one metric alone is misleading.
Continue using the full 3,000-second horizon: the earlier 600-second BO training
episodes mostly cover the resource-rich opening.

The current policy should be treated as experimental. Keep the original saved
baseline as a comparison, and avoid another large tuning run until a small set
of changes demonstrates a repeatable full-horizon benefit. A sensible first
follow-up is to fix the missed shared birth evidence and test it independently,
before adding more population or movement rules.

## Validation and reproduction

The current source passed **498 tests**, including conservation scheduling,
essential-task overrides, navigation pause/stall behavior, reproduction forecasts,
public metabolism inference, and route energy checks. A separate focused audit
also passed 66 conservation/navigation/expert-policy checks.

Tested controller source fingerprint:

`631e4eb2db3ca2e06498ad4314937d2db6cb2d7a8b81db9d7912d4ddba8fe293`

Raw results and exact experiment configuration are retained in:

- [Conservation enabled](runs/survival-v10/summary.json): five seeds.
- [Clean control](runs/survival-v10-no-ramp/summary.json): three seeds.
- [Larger population](runs/survival-v10-wide/summary.json): three seeds.
- [Saved baseline comparison](runs/fresh-original-forage-v7/baseline-comparison.json):
  original baseline values and a separate earlier ablation.

The `runs` directories are local experiment artifacts; the result tables in this
report preserve the conclusions even when those directories are not distributed.

From `survival-simulator`, run a new checkpointed validation with:

```powershell
.\.venv\Scripts\python.exe benchmark_survival.py --seeds 1001 1007 1042 --workers 3 --seconds 3000 --output runs/survival-next
```

Repeat the same command to resume an interrupted run. Use a new output directory
after changing source or configuration; the manifest rejects incompatible
resumes. This benchmark explicitly enables centralized harvesting and disables
predators. Ordinary runs retain their configured behavior.

For the rendered predator-free policy:

```powershell
.\.venv\Scripts\python.exe local_playground.py --seed 42 --no-predators --central-harvest
```

No further policy changes or experiment rounds were started after the request
to finish the current comparison and report the findings.

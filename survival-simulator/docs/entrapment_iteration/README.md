# Crowded trap iteration

Working branch: `survival-simulator/lucas-experimental`. Oscar's survival module
is unchanged. Policy changes use ordinary observations and known/observed walls;
only benchmark fixtures/evaluation read native world state.

## Measured delivery results

Native 30-preloaded + one newcomer benchmark. Full-energy fixture bait,
normal-energy full-start guide, native predator sensing/movement. All original
30 must remain held, the newcomer must settle, and the replacement side must
stay clear. Same criteria as `docs/guide_multi_1000_results.md`.

| Development variant | Passed / 12 | Passed / 100 |
| --- | ---: | ---: |
| Existing 55-unit stop, nearest-predator selection | 9 | 68 |
| 40-unit stop, nearest-predator selection | 4 | — |
| 55-unit stop, observed-motion target association | 10 | 72 |
| 40-unit stop, observed-motion target association | 7 | — |
| Require newcomer within bait hearing before stopping | 9 | — |
| Restrict stopping to the front approach lane | 10 | 65 |
| Front-lane stop plus recovery into predator-width paths | — | 67 |
| 55-unit stop, tracking plus route recovery (current default) | 9 | 72 |
| Crowd-preference sacrifice guard | 9 | — |
| Side-offset handoff | 6 | — |
| Bounded wait for observed stationary predator | — | 63 |
| Distant, sight-checked sacrifice (opt-in) | 5 | — |
| Prefer a central valid site | 5 | — |

Each column uses identical map/encounter seeds across variants. The 12-map and
100-map encounter lists differ. These are repeated development measurements,
not independent estimates of general reliability. The tracking change won
seven cases and lost three versus baseline. Four of the 100 maps had no site.
The historical 1,000-case result remains unchanged; it does not evaluate these
new policies. No 95% delivery reliability is claimed.

The larger tracking-only benchmark is preserved as tag
`entrapment-20260918-track55-100`; `source-tags.json` records its exact commit.
Every batch now freezes its source before running, so live edits cannot alter
later cases. All tick traces and frozen source remain under the ignored
`logs/entrapment-iteration/` directory. Summaries/seeds/hashes are versioned here.

## Corners and replacement

An ordinary inside corner caught bait in 16/16 native approaches; the 15-unit
crevice control caught it in 0/16 over 30 seconds. Plain corners are unsafe.

The new corner selector finds short staggered pockets (overlap 8–10.3 units)
and requires the distance from bait to **every collision-valid predator centre**
to be at least 15.05 units. It also requires rear access and a usable handoff.
A 100-map survey found candidates on three maps, all already having ordinary
crevices. After shifting the final approach around supporting walls, native
trials on seeds 1, 11, and 49 all retained their original 30 predators and all
three replacement agents arrived and survived. Only seed 11 delivered the
newcomer. This is not a delivery reliability estimate for corners.

`available_sites` uses corners only as fallback when crevices are unavailable.
An occupied corner remains eligible during revalidation even if exploration
later discovers a crevice; discovering one must not evict existing bait.
The four missing-site maps in the 100-case development sample still have no
qualifying site with this fallback.

`guide_multi.py --replace-bait` starts an actual full-energy replacement at the
rear staging point during final hold. It walks into the channel; after arrival,
the old bait is explicitly exhausted and retention is measured against the new
bait. This isolates physical handoff access. It does not test travel from a
random location, candidate scheduling, or natural old-bait life estimation.
A paired 12-case check of the selected default passed the same 9 cases with
and without physical replacement. All 12 replacements arrived and were alive
at the end; no additional failed outcomes were introduced in this sample.
Those wider scheduling concerns are exercised separately by native games. Replacement position/energy/
alive status are retained in the tick trace.

## Native game checks

- Seed 0, 600 native seconds: 13 agents alive, 6 predators, maximum 5 near bait,
  maximum 4 continuously nearby for 30 seconds, 14 overlapping replacements,
  zero **estimated** bait-gap seconds after first arrival. This used an earlier
  tracking/40-unit candidate and the initial bystander-avoidance wrapper.
- Seed 3, 200 native seconds: 22 agents alive, 3 predators, maximum 3 near bait,
  5 overlapping replacements, zero estimated bait gaps. Includes sharing fresh
  teammate predator observations into the avoidance wrapper.
- Seed 3, 300 native seconds: 21 agents alive, 4 predators, maximum 3 near bait,
  9 overlapping replacements, zero estimated bait gaps. Survivor release logic
  was exercised without errors but no guide met its release condition in this
  game, so successful release is not demonstrated by this run.

These are short integration games, not whole-game guarantees or matched-policy
score comparisons. Proximity is a proxy for capture, and bait occupancy is based
on the policy's estimated localization. Every native tick is recorded. Replay:
http://localhost:9063 (seed 0, 600 seconds, native sprites).

The 100-case tracking-plus-route-recovery overview is at http://localhost:9064.
It shows all outcomes and provides on-demand native-sprite replays from frozen
source. Re-rendered multi-predator outcomes/final states must match the recorded
evaluation before display. The original tick traces are also downloadable.
Restart it with `python scripts/guide_batch_viewer.py
logs/entrapment-iteration/track-rejoin-100 --port 9064` from the simulator folder.

## Implementation and limits

- `my_guide.py`: associates the newcomer by observed motion rather than always
  choosing the nearest crowd member. Identity remains ambiguous at overlap and
  after long occlusions; native observations have no predator IDs.
- `guide_pathfinding.py`: if collision/survival steering puts a guide inside the
  predator-width margin, take a clear agent-width step back into a reachable
  predator corridor. A previously stuck case (development index 25) then passed.
- `bystander_avoidance.py` plus `core.py`: avoid known hearing/vision exposure and
  occupied bait areas, including fresh shared sightings in an aligned frame.
  Unknown predators and motion/localization error prevent an absolute guarantee.
- `core.py`: surviving guides may return to normal work after ten seconds of
  observed proximity to bait. Native-game evidence for a successful release is
  still missing; close overlapping tracks can have ambiguous identities.
- Crowd-preference and side-offset sacrifice experiments regressed on the small
  paired sample and are not enabled. The default retains the measured 55-unit stop.
- `--vision-delivery` is an explicitly experimental benchmark option: route
  outside an 85-unit bait exclusion radius and sacrifice at a distant front
  point only with observed sight alignment and wall clearance. It is not enabled
  by the native coordinator. It passed only 5/12 versus 10/12 for tracked
  ordinary delivery, so it is not recommended.
- A recording-only hearing audit of the 200-second native game counted 27
  exposed bystander agent-ticks out of 28,560, and zero hearing-exposed ticks
  involving predators within 40 of bait. This does not audit vision or prove
  safety; there is no matched baseline.

## Remaining experiments

The nine-way delivery-radius/walking-threshold screen found no improvement over
its 17/20 control. Bounded waiting regressed to 63/100 and is not enabled.
The waiting variant passed 72/100 on fresh maps (batch seed 19092026). The
selected default is being evaluated on the same fresh cases. Central-site
ranking regressed to 5/12 and is not enabled. Native-game coordination checks
and a near-distance sight guard remain in progress. Smaller stopping distances
and front-lane restrictions regressed and should not be promoted from intuition.
Do not read 96.5% map-site availability as a delivery success rate.

## Compute budget

Authorized Runpod cap: **$10 total**. **$0 spent** and no resources created.
Four creation requests (CPU3 32-vCPU twice, CPU5 32-vCPU once, CPU3 16-vCPU once)
returned HTTP 400, "no longer any instances available with the requested
specifications." Existing team pods were not changed. All completed experiments
ran locally. `runpod-budget.json` records this separately from other team work.

Account usage remaining is not exposed by the available tools. The requested
20% account-usage stopping threshold cannot be measured automatically here.

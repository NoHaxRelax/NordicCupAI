# Real-map approach research — 2026-09-17

## Current result

The most promising development candidate is `policy_v18_escape_lookahead.py`: continuous gaze and close follower spacing, observation-based reacquisition, static-map routing, and a short escape planner near walls. It repaired four of five known wall failures. Four fresh maps produced four eventual captures, but only three were attributable guided deliveries: on map 10048 the guide died at 3.2 seconds and the predator independently reached bait at 53.4 seconds. Do not report this as 100% guiding success. The requested >95% reliability is not established.

No further simulations are scheduled in this round. The user-authorized usage stop is 40% remaining. This round is local and is not included in pushed commit `983e6ebe6fc39ff455c452297244277ee515d177` on `survival-simulator/lucas-trap-slopsesh1`. Large recordings remain ignored by Git.

## Test scope

Native generated maps; bait predeployed at depth 5; infinite agent energy. Predator starts at a random valid fixture point, awake with full energy. Guide starts 55 units away facing it. Predator movement, energy, rest and ambient spawning are native after setup. No child is spawned. The initial adjacent segment is not guaranteed clear for the larger predator; these cases remain in the denominator.

Runtime controllers receive native observations, public time, role IDs and detached static map data only. Hidden target selection, positions and deaths are evaluator inputs, never controller inputs. Perfect static map information is allowed. These experiments test an already-engaged single-predator delivery, not predator discovery, bait deployment, finite-energy operation or repeated delivery of 33 predators.

## Frozen fresh-map results

Do not pool different versions or fitted repairs into one reliability rate.

| Policy | Map / fixture seeds | Horizon | Original strict passes |
| --- | --- | --- | --- |
| v8 latency spacing | 10001–10008 / 9301–9308 | 120 s | 4/8 |
| v10 approach lane | 10012–10027 / 9412–9427 | 180 s | 11/16 |
| v13 close spacing + reacquisition | 10028–10031 / 9528–9531 | 180 s | 3/4 |
| v15 direct emergency escape | 10032–10047 / 9632–9647 | 180 s | 9/16 |
| v18 escape lookahead | 10048–10051 / 9748–9751 | 180 s | 4/4 eventual capture; 3/4 attributable delivery |

Different seed sets and small samples prevent a clean ranking by these fractions. v18 remains a development candidate, not a validated replacement.

### v18 individual results

All recorded 180 seconds and 1,801 native frames. Guide sacrifices occurred in all successful cases.

| Map | Group | Delivery time | Outcome | Receipt suffix |
| --- | --- | --- | --- | --- |
| 10032 | Known failure | 17.8 s | Pass | d270d47a |
| 10038 | Known failure | 5.2 s | Pass | 24e3ac64 |
| 10040 | Known failure | — | Guide dies at 3.3 s, no delivery | f70f0c17 |
| 10043 | Initial fitted diagnostic | 13.3 s | Pass | 64827ef5 |
| 10045 | Known failure | 17.5 s | Pass | adf5f436 |
| 10048 | Fresh | 53.4 s | Autonomous capture after guide death at 3.2 s; guiding failure | 71450a5b |
| 10049 | Fresh | 10.9 s | Pass | 9c7e1c00 |
| 10050 | Fresh | 10.4 s | Pass | 874c953f |
| 10051 | Fresh | 83.3 s | Pass after losing/reacquiring pursuit | fa2cfb66 |

The four-case regression batch repaired three failures; adding the initial fitted diagnostic gives four of five. Fresh successes had zero recorded physical/target losses through 180 seconds.

## What changed and remaining failure modes

Early repairs widened the final approach gate, moved travel spacing closer to hearing range, added recovery after losing observation, and avoided sprinting away before map localization. Direct emergency escape still caused deaths around walls. Sol's four-tick direction commitment repaired one of five fitted wall failures; wider static wall clearance repaired neither of two diagnostics.

v18 searches six movement steps with a beam of 12 candidates, using two predator speed hypotheses (11 and 15), observed position/heading, static terrain and approximate native collision response. Native predator observations precede predator movement: the planner explicitly advances the unseen prior movement before considering the next guide action. Sol independently corrected its initial stale-step criticism and found no privileged runtime input.

Limitations remain: approximate collision/localization; assumed pursuit of the guide; incomplete perception, target switching, rest and watched-pivot modeling. The short horizon can still choose an unrecoverable route. Map 10040 remains an early death, and fresh map 10048 exposes the same broad safety problem. Next work should inspect these two native recordings before adding policy complexity, then freeze a candidate for a larger fresh-map batch with attribution fixed in advance.

## Scoring caveats

Original immutable receipts count two seconds resting near the mouth as provisional capture, even before active bait acquisition. This can create a later apparent target loss on waking. Also, eventual autonomous arrival long after a distant guide death can be misattributed to guidance.

`audit_capture.py` produces separate post-hoc sidecars from native recordings and target traces. It requires active bait acquisition before treating rest as contained, and flags clear distant post-death arrivals. It does not prove causal attribution in every case. Original scores remain unchanged. In v15, map 10047 has stable active bait acquisition from 16.6 seconds despite an earlier rest-only provisional target loss. Map 10033 arrives at 175.1 seconds, too late for the required 30-second tail. Sol's fitted v15 map 10014 was also an autonomous capture, not a repaired delivery.

## Map availability and retention, separate from approach reliability

Static short-overlap census: 127/128 maps (99.2%) have eligible geometry with a clear approach. This is bait-site availability, not delivery success. Conservative long-overlap geometry had clear approaches on only 53/128 maps.

Prepared 33-predator native-map tests passed 12/12 maps for 120 seconds. They do not demonstrate transporting 33 predators. The older full-game prepared-33 attempt was stopped safely at 1,587.5/3,000 seconds: all 33 held, bait alive, zero physical losses; incomplete horizon.

One v3 single-predator delivery on map 5101 / fixture 9203 completed 3,000 seconds with zero losses and bait alive (13ea7366), with 30,001 verified native frames. This is one full-game reference, not general reliability evidence.

## Recordings and safe operation

Every completed run retains each native 0.1-second frame, including failures. Streaming recorders journal frames/events and passed equivalence checks against the original recorder. The original vendor simulator is unchanged. Capped fresh worker processes and a guard enforce available-RAM, disk and usage limits; see `RESOURCE_INCIDENT.md` for the IDE OOM incident and partial recovery.

Every-frame viewer: http://127.0.0.1:9055/research/real_map_visualization/index.html

Overview and independent Ask Sol: http://127.0.0.1:9055/research/strategy_visualization/index.html

The memory-heavy 9053 library is stopped. Use `debugger/catalog.py --verify-stream` to verify replay discovery/integrity without loading full recordings. Evidence lives in `results/simple_chase`, `results/short_overlap_sol`, and post-hoc `active_bait_audits`. Preserve incomplete spool journals and immutable receipts. Do not clear a quota stop without new user authorization.

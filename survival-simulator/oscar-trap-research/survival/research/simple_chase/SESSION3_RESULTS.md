# Real-map reliability, boundary gaps, and replaceable bait

Active research, started 2026-09-17 after the user authorized all remaining usage. The old 40% stop is superseded; shared stop threshold is now 0. Infinite agent energy and perfect static map remain permitted. Guide survival and sacrifice are equally valid delivery outcomes. No runtime privileged creature data is allowed.

Current targets: a frozen controller reaching >95% single-encounter delivery reliability; preserve rear access for replacement bait; count boundary-wall gaps. Whole-game retention and sequential delivery of 33 predators remain separate, unproven goals.

## Static geometry

The original selector did include the four native boundary wall rectangles (30 units thick), but its centered axial approach excluded their narrow gaps. The new pure selector in `../replaceable_sites/selector.py` allows a small lateral approach offset and requires a radius-5.01 path through the opposite mouth to the depth-5 bait goal. On the same 128-map census: old short-site geometry 127/128; new replaceable geometry 128/128; eligible boundary sites 88/128. These figures establish geometry availability only.

Root v21 selects centered nonboundary replaceable sites preferentially, with boundary/offset sites available as fallbacks. It uses the explicit selected runup rather than reconstructing a centered one. The forced-boundary diagnostic on map10000/9204 died at3.3s before reaching the site (c18d1c7b); this is an approach failure, not evidence against boundary holding.

## Native replacement proofs

Sol recorded three prepared native-map handoffs: two on map10000, including the exact boundary-wall candidate used by that failed approach, and a 10.1038-unit boundary gap on map10144. In the boundary proof (e51cd55e), the new bait reached the goal at13.3s, the old bait reached the rear exterior at29.2s, and both lived through60s. Goal coverage was continuous across all601 native frames and the predator was held through the final30s. The nonboundary proof is8569aee9. Both are clearly labeled prepared fixtures, not autonomous replacement or general reliability evidence. Serialization failed after original replay save; separate audit receipts were reconstructed from the preserved original replay, without fabricated frames.

## Guide survival and release

`../release_validation/release_mixin_v2.py` uses ordinary observations to detect a predator near the bait and significantly farther from the guide, waits20 ticks, then retreats on its current side of the gap. It never reads the predator's target. In the v13 development case10028/9528, delivery11.9s, release15.4s and safe retreat16.0s, guide alive and predator retained through180s (4528b76f). The earlier v1 release attempted to return a guide already behind the gap toward the front and stopped safely; it did not complete retreat (c8d37001). Preserve both.

Integrated v21+release on10040/9640 completed300s: delivery17.7s, release21.8s, retreat22.5s, guide alive (cdd19a1f). All five fitted v21 regressions strictly passed300s. The fresh v21 four-map batch passed3/4 inclusive; map10144 had no default eligible site. A new narrow-gap fallback repaired that same map in a separate fitted300s run. Nine completed v21/v23 recordings pass strict scoring, but these mix fitted and fresh cases and omit the separately recorded unsupported attempt; do not report9/9 as general reliability.

## Safety controller

Sol v19 added a curved relative-frame escape before map localization and triggered lookahead earlier. It kept10040 alive but lost pursuit, so it was not promoted. v20 also penalizes predicted separation beyond55; both10040 and10048 passed180s with attributable handoffs. The frozen fresh16 batch (maps10128–10143 /fixtures20128–20143) completed300s. Strict result11/16=68.75%. Failures:10134 unsupported,10138 guide death1.0,10139 guide death7.9,10142 delivery too late for30s tail,10136 autonomous arrival after ~50 active seconds without guide pursuit. The original harness reports12/16; the reconciliation is `results/reliability_eval/development_scores/V20_RECONCILIATION.md`. Preserve original receipts unchanged.

Independent native-source audit found that stationary observed predators can be awake and collision-stuck, so stationarity must not disable safety planning. Other remaining issues include fixed-speed hypotheses, watched-pivot/visibility outside hearing, and anonymous predator identity switching. A separate robust candidate is being developed; the v20 batch remains unchanged.

## Reliability protocol

`../reliability_eval` defines a fixed fresh59-case benchmark at300s. It requires a recent observed-in-evaluation guide-target to active-bait handoff inside the intake, confirmed containment, acquisition by270s and no subsequent losses. Rest cannot initiate acquisition. A living guide is accepted. Unsupported, missing and failed cases remain in the denominator. Every native frame and recursive local policy dependency hashes are verified. The latest frozen protocol hash is recorded in `results/reliability_eval/FROZEN_PROTOCOL.json`; old unused drafts remain in protocol history. 59/59 would give a one-sided95% exact lower bound of0.950492. This benchmark has not yet run.

Viewer: http://127.0.0.1:9055/research/real_map_visualization/index.html
Session3 checkpoint is being saved on survival-simulator/lucas-trap-slopsesh1. Large native replay/chunk assets remain local and are git-ignored; receipts and reproduction sources are versioned. Original receipts and source versions are immutable. The prior snapshot is SESSION2_RESULTS.md.

## Current follow-up experiments (ongoing)

- Root v24 combines replaceable rear-access sites, narrow-gap fallback10.1, short-overlap fallback10.3, and surviving-guide release. Frozen fresh development16 maps10160–10175 /fixtures+10000 completed300s with strict11/16. Failures10164/69/70/71/75 were guide losses before attributable capture. All11 captured predators stayed held. Exact one-sided95% lower bound is0.45165. This is separate from reserved59.
- Sol terrain candidate filtering repairs the known10139 river-entry death in an initial180s probe. It rejects ordinary spacing moves into slower terrain while the observed predator is closer than65. An integrated version and300s validation are pending.
- Static exact-model search of the10138 corner found no surviving path by tick13 among32 headings/1000 diverse states after the mandatory blind tick. This is strong finite-search evidence of an initially forced-loss pocket, not a mathematical impossibility proof. It remains a failure in the unchanged denominator.
- Robust0/11/15 speed hypotheses pass10040 but fail10138. One native reserve child also fails10138; a bounded eight-birth alternative is now testing separately, with explicit extra guide cost.
- Static terrain-cost A* repairs the exact boundary10000/9204 approach: guide death48.2, active bait acquisition48.4, retained through300s. It fails10138 and10139, so it is not a general safety repair.
- A five-encounter sequential native-map pilot (map10040, fixture22040,75s spacing,600s) completed4/5 attributable colony deliveries, with no subsequent losses. Guides act independently on ordinary observations; the evaluator permits colony-level assistance by any guide. Version1 lacks exact pre-action insertion frames, although every native tick and all spawn coordinates are preserved. Version2 introduces pairs immediately before the scheduled native frame and records actual assisting guide IDs plus recursive hashes. No33-predator transport claim follows from this pilot.
- Boundary rear replacement proof on10144: new bait reachesgoal12.8s, old bait exitsrear28.7s, both alive and predator held through60s. Original replay title mistakenly mentions10000; separate audit corrects map identity. This is prepared replacement, not autonomous general reliability.

Current frozen protocol v3 SHA256:429cc7b2d0439ac1756acea3f0fe521e68fced933f95bb7ebabe175a257f8016. Its3s handoff bound counts predator-active time, excluding native rest, and reports wall time separately. Reserved59 has not started.

## Additional native evidence

Prepared33-crowd bait replacement passed on native10144's narrow boundary gap10.1038 (e41b8719): new bait reached the goal, old bait exited rear, both remained alive, goal coverage continuous, all33 physically held through90s including final30s.901/901 native frames verified by catalog. Predator starts are arranged near the mouth and may overlap; this is replacement/retention evidence only.

The first repeated random-arrival pilot completed600s with4/5 attributable colony deliveries, all four retained and bait alive. One predator remained undelivered. Original receipt88686549 and6001/6001 native frames preserved. Cross-guide assistance occurred, so do not present these as five independent single-guide trials.

Eight bounded native reserve births rescued the previously fatal10138 pocket: eight guide deaths, child6 survived, native guide target35.2→bait37.8→physical intake39.0, continuous bait/rest through300s (9f53c5f9). This fitted proof uses ordinary DTOs and native child positions/mutations. Frozen burst fresh8 maps10200–10207 is running. A separate emergency-only birth rule is testing10164 and is not part of that frozen batch.

A potential scorer correction is under independent paired validation, not silently applied: start attribution at the actual native guide→bait target switch within3 active seconds, then require an unbroken bait/rest chain until physical intake. Current v3 anchors3s at physical entry and can reject legitimate longer travel after bait becomes the target. Preserve all v3 results, the autonomous negative cases, and any future v4 paired comparisons.

## Terrain and reserve follow-up checkpoint

Solv26 passed4/4 fresh maps10180–10183 and three separate fitted regressions. Solv28 adds an8-unit uncertainty margin around slower terrain to every movement branch. It passed fitted10164/10139 and3/4 fresh10212–10215. Failure10212 repeatedly maximized one-step separation and eventually reached the map boundary; safer lookahead remains unresolved. Small fresh batches are not >95% reliability evidence.

Rootv27 emergency reserves fresh4 and originalburst fresh8 are still running. The burst batch has one pass, two delivery failures, and one native move-distance contract error among its first four completions. All remain in the denominator. A root finite three-step conservative escape candidate survived10164 through60s but had not delivered; its reserve variant lost all guides in10138 at1.1s. It is not promoted.

The exact static-obstacle broad phase in integrated_guide/spatial_geometry.py matched14,064 original geometry queries. Native trajectory equivalence is being checked separately; do not assume timing optimization alone establishes controller equivalence.

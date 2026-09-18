# V29 and V30 diagnostics

V29 combined the V28 terrain barrier, exact staged selector, release/sweep behavior, rejected-pose cache fix, and emergency-only reserve births. Its single-guide 10164 regression passed. Its eight-reserve 10138 run was a legacy harness success but causal scorer V4 failed `no_unbroken_guide_to_bait_chain`; the reserve variant is not promoted.

V30 replaced V28's one-step max-separation terrain substitution with a short static viability check and made the final handoff continue through the mouth once the follower was close. Frozen hash: `82c1fd608132feb50a479249547ad9896bba602b9ef28077d554290ec022093d`.

| Map / fixture | Why tested | V4 result | Timing |
|---|---|---|---|
| 10169 / 20169 | distant autonomous capture under V24 | pass | guide death, handoff, acquisition all 31.3 |
| 10171 / 20171 | slow-river final lane | pass | guide death/handoff/acquisition 78.9 |
| 10224 / 20224 | V28 one-step barrier death | pass | death 33.9, handoff 34.2, acquisition 34.3 |
| 10212 / 20212 | boundary/MPC failure | fail | death 100.3, no capture |
| 10175 / 20175 | boundary target-loss failure | fail | death 5.8, no capture |

Passes retained the predator through 300 seconds. These are fitted diagnostics, not a reliability estimate.

The viability floor of 50/65/80 assumes a 15-unit predator everywhere and is too restrictive in uniform slow terrain. Empty option sets also fall through to the original action. A later integrated candidate should derive the conservative predator bound from static terrain around the observed predator and retain a first-step safety bound when no multi-step option exists. Wide-clearance terrain routing is independently promising for the two remaining boundary failures.

Receipts and causal score sidecars are under `results/short_overlap_sol/v30_viable_terrain_final/`.

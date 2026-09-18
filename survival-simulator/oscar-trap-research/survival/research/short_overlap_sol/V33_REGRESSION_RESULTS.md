# Root V33 independent regression

Independent 300-second native replays of frozen `integrated_guide.policy_v33_adaptive_guard:Guide`:

| Map / fixture | V4 result |
|---|---|
| 10169 / 20169 | pass; acquisition 9.6 |
| 10171 / 20171 | fail; death 13.3, no capture |
| 10175 / 20175 | pass; acquisition 2.5, guide alive until 192.9 |

V33 therefore regresses the slow-river final case that Sol V30 repaired, and is not a clean replacement. Receipts and score sidecars are in `results/short_overlap_sol/v33_regression/`.

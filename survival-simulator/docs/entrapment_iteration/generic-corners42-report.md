# Generic corner detector: frozen static survey and native validation

## Detector contract

`generic_corner_detector.py` accepts only `width`, `height`, and closed obstacle
rectangles. It has no map-seed branches. It constructs exact AABB configuration
spaces for radius-5.01 bait/replacement agents, radius-10.01 predators, and a
radius-11.01 guide-plus-predator approach.

Each returned site contains the ordinary guide fields:

- `goal`: agent-valid bait point at least 15.05 from every predator-valid
  centre;
- `handoff`: main-component radius-11.01 point 25--38 from bait;
- `mouth`, `inward`, `cross`: local guide-frame geometry;
- `replacement_entry` / `other_mouth`: radius-5.01 rear staging point outside
  the main/front predator component;
- access, clearance, region-area, and evaluation metadata.

The complete straight rear segment must be agent-valid and at least 15.05 from
the nominal handoff. This prevents the detector from labeling the front
approach itself as the rear in diagonal pockets. For future observed-wall use,
unknown space must be treated as blocked; the detector should run only on
closed reconstructed rectangles and known arena boundaries.

## Fixed 100-map geometry survey

The fixed set is the four supplied missing-site seeds followed by seeds 0--95.
The frozen v2 survey is `generic-detector-fixed100-v2.json`.

- generic candidates: 100/100 maps;
- genuinely added maps with no ordinary site: 7;
- added seeds: `1214781215`, `1036448012`, `1989373803`, `1035850866`, `14`,
  `35`, `42`.

This is geometry availability only, not usable-site reliability.

## Native validation of the seven added maps

The first automatically ranked site was used unchanged on each map. Six fixed
encounter seeds per map were declared in the manifest:

`1382370471, 854619562, 1643999355, 2085883922, 1374628056, 772838390`.

Every case used the unchanged current `my_guide`, stock random visible
guide/predator encounter sampling, 30 preloads, one newcomer, native movement,
and actual walking bait replacement. The rear-clear evaluation is contact
clearance to that site's own full replacement segment.

| Map seed | Full passes / 6 |
| ---: | ---: |
| 1214781215 | 5 |
| 1036448012 | 0 |
| 1989373803 | 4 |
| 1035850866 | 6 |
| 14 | 6 |
| 35 | 6 |
| 42 | 5 |
| **Total** | **32 / 42 (76.2%)** |

All 42 replacements arrived and were alive at end. No case violated the
site-specific rear-segment contact rule. All ten failures were newcomer
delivery/retention failures; no initial trap failed under the corrected
evaluator.

The systematic failure is seed `1036448012`: its first-ranked site at
`goal=(1018,388)` failed 6/6 guide deliveries. The same map has a second
detected candidate at `goal=(276,1042)`; detector v1 happened to rank that site
and it passed 6/6, but changing the ranking after seeing these results would be
held-out optimization. The frozen default remains v2 and the 0/6 result is
reported as-is. Candidate ranking needs an independently motivated/fresh-map
study, probably emphasizing runup topology and safe-region area.

Seed `1989373803` illustrates the difference from manual positioning. The
manually studied site (`goal=(498,211)`) passed 14/20 random encounters. The
generic detector selected `(503,216)` with the opposite valid orientation and
passed 4/6. Both completed every attempted replacement; guide delivery is the
main variance.

An earlier v1 native batch reported six `initial_trap_failed` outcomes on seed
1989373803. Those were an evaluator bug: default Python arguments retained the
manual seed's rear segment after generic sites were substituted. Replacement
movement itself used correct coordinates. The v2 results above use dynamic
per-site evaluation and supersede those six classifications.

## Artifacts and hashes

- `generic_corner_detector.py` — SHA-256
  `00553eb1e2d32dae02700e3767c3129006802eb6a913a948b2741705aca4cb91`.
- `generic-detector-fixed100-v2.json` —
  `717143375b75b18285c9afd106f6b315192b78caec962650830db8eb761aba0a`.
- `generic-native-v2-7x6/results.json` —
  `9f119e2fa7dccb81fa48d59f79fb1bc6bd279aaa8a29587308275c932b486e5c`.
- `generic-native-v2-7x6/manifest.json` and 42 per-case every-tick traces.
- `guide-policy-final20/results.json` (manual original-map placement) —
  `2859b4c372bd832ea0742ade05d0b445d44e7a347d20e23f0d266677725977e5`.

The detector is promising at roughly 76% under this small, development-selected
protocol. It does not support a 95% claim and is not ready for production
without fresh-map candidate-ranking validation.

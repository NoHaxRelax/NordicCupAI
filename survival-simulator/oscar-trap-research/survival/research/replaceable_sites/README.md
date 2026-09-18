# Replaceable static sites

`select_site(static_map)` is a pure static-map interface. It accepts `width`,
`height`, and obstacle rectangles (mapping or four-number forms), and returns
the legacy site fields used by `policy_v5_short_site` plus:

- `runup` and `approach_lane_offset`, including an offset clear approach for
  obstacle-to-arena-wall gaps;
- `other_mouth` and `replacement_entry` for the opposite channel entrance;
- `second_access_clear`, a radius-5.01 static path to the depth-5 bait goal;
- `geometric_replacement_access_only`, which prevents interpreting the static
  predicate as a demonstrated safe handoff near an occupied trap.

Native maps create four 30-unit-thick arena walls as obstacle indices 0–3.
The frozen selector already iterates these rectangles, but its axial far/hold
lane rejects every boundary-pair site in the 128-map census. The new selector
searches small cross-lane offsets after the finite obstacle ends and reports
the chosen lane explicitly. It leaves the channel mouth and depth-5 goal in
their compatible locations.

The preserved-map census (seeds 10000–10127) compares the exact frozen short
predicate with overlap at least 20 against the replacement-access predicate:

| Static predicate | Maps |
| --- | ---: |
| Frozen v5 axial short site | 127/128 |
| Frozen v5 site using a boundary wall | 0/128 |
| New site with opposite-mouth bait access | 128/128 |
| New site using a boundary wall | 88/128 |

These are map-availability counts. No agent, predator, controller action, or
simulation step was used. In particular, the 128/128 result does not establish
predator delivery reliability or show that a replacement bait can safely pass
an already contained predator.

One separate prepared native-map fixture on seed 10000 tests that missing
dynamic step. A replacement entered from the opposite mouth and reached the
depth-5 goal at 13.2 seconds. The old bait began its rear exit at 28 seconds and
reached the opposite exterior at 29.2 seconds. At least one bait stayed within
6 units of the goal on every frame; both agents were alive at 60 seconds, and
the predator remained held throughout the final 30 seconds. The replay retains
all 601 native frames. This is one arranged continuous-handoff proof, not an
autonomous replacement or population-level reliability result.

A second prepared proof uses the exact boundary candidate selected on native
map 10000: bottom arena wall index 1 with obstacle 72, mouth
`(195.858, 1161.190)`, and a -4-unit offset approach lane. The replacement
reached the goal at 13.3 seconds and the old bait reached the opposite exterior
at 29.2 seconds. Goal coverage was continuous, both agents survived, and the
predator remained held for the final 30 seconds. Its 601-frame native replay is
catalog-valid. This establishes the prepared hold and rear handoff at that
boundary site; it does not repair the separate guide-delivery failure en route
to the site.

Fresh map 10144 remains correctly unsupported under the frozen defaults.
Static regeneration with no simulation steps found that reducing overlap from
20 to 10.3 recovers no site. Lowering only the minimum gap from 10.9 to 10.1
recovers two sites; widening the maximum from 19.1 to 19.9 is unnecessary. The
narrowest gap is 10.1038, leaving only 0.0838 units beyond a radius-5.01 bait's
diameter. It is an analysis candidate, not validated usable geometry, and the
original unsupported map remains in the fresh v21 denominator.

The opt-in `expanded_selector.py` freezes the proposed frontier separately:
minimum gap 10.1, maximum gap 19.1, minimum overlap 10.3, depth 5, opposite
access required. It does not alter the validated selector defaults. Map 10134,
which the old v20 selector rejected, has three sites under the existing new
defaults; it does not require the expanded frontier.

The narrowest map-10144 candidate (gap 10.1038, boundary wall 1 plus obstacle
11) passed one prepared 60-second native handoff. Replacement reached goal at
12.8 seconds, the old bait reached the opposite exterior at 28.7 seconds,
coverage was continuous, both survived, and the predator remained held for the
final 30 seconds. All 601 native frames verify. This supports physical passage
and prepared retention for one near-diameter gap, not approach reliability or
general safety at the expanded threshold.

Run the checks with the requested interpreter:

```sh
PYTHONPATH=research /home/Ucals/projects/NordicCupAI/.venv/bin/python research/replaceable_sites/verify_contract.py
PYTHONPATH=research /home/Ucals/projects/NordicCupAI/.venv/bin/python research/replaceable_sites/census.py
```

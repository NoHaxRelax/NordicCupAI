# Normal-guide benchmark and generic detector

## Seed 1989373803, finalized fixture

The scratch override uses the unchanged current `models/entrapment/my_guide.py`
and the unchanged `scripts/guide_multi.py` loop. Only static site selection and
the rear evaluation predicate are replaced.

- bait: `(498, 211)` (globally predator-contact-safe, approximately 15.36 from
  the radius-10 predator configuration space);
- handoff: `(512, 184.5)` (in the main radius-11 reachable component, 29.96
  from bait);
- replacement entry: `(498.546, 283.584)`;
- rear-clear rule: no predator centre within 15.05 of the complete replacement
  segment, because a generic diagonal pocket has no meaningful channel axis.

The 20 encounter seeds were frozen in `guide-policy-final20/manifest.json`
before execution. New guide/predator pairs use the stock random visible
encounter sampler: 80--160 apart, at least 250 from bait. Guide energy is full
only at start. Policy inputs contain ordinary observations, bait, known walls,
handoff, mouth, tick, and time; no live predator state or identity is supplied.

Results: **14/20 full passes**. Fifteen newcomers met the delivery criterion.
Five cases failed delivery/retention, and one delivered but temporarily lost an
original predator (`initial_min_held=29`). All 20 replacements arrived and were
alive at end; none violated the replacement-segment contact rule. This is a
single-map repeated encounter result, not a general reliability estimate.

## Generic detector

`generic_corner_detector.py` is seed-independent and accepts only a static map
contract: `width`, `height`, and obstacle rectangles. It constructs radius-5.01
agent, radius-10.01 predator, and radius-11.01 guide-plus-predator configuration
spaces. It emits candidates with:

- a radius-5.01-valid bait at least 15.05 from every radius-10.01-valid centre;
- a handoff in the main radius-11.01 reachable component, 25--38 from bait;
- a radius-5.01 straight rear access path whose staging point is outside the
  main/front predator component;
- ordinary `guide_multi` fields (`goal`, `handoff`, `mouth`, `inward`, `cross`,
  `replacement_entry`, access and clearance metadata).

The detector selects a bait near the safe region's front boundary to improve
the old-bait preference margin during replacement. It does not use map seeds.
Observed-wall integration can run the same function once closed rectangles and
arena dimensions are reconstructed; incomplete/unknown space must remain
blocked until observed.

On the fixed set consisting of the four supplied missing-site seeds plus seeds
0--95, it returned candidates on 98/100 maps and added candidates to seven maps
with no ordinary site: the four supplied seeds and seeds 14, 35, and 42. This is
only a geometry survey. The automatically selected sites have not yet received
a native multi-map benchmark and must not be treated as 98% usable coverage.

## Artifacts

- `guide_multi_corner_override.py` — finalized single-map site override.
- `guide_policy_batch.py` — frozen two-worker batch runner.
- `guide-policy-final20/{manifest,results}.json` and per-case tick traces.
- `generic_corner_detector.py` — isolated seed-independent detector.
- `generic_detector_survey.py` and `generic-detector-fixed100.json` — fixed-map
  survey and complete candidate records.

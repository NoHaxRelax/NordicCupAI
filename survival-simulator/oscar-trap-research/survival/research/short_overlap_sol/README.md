# Short-overlap retention validation

This experiment tests whether the fixed depth-5 arranged trap still holds a
33-predator crowd when the two wall faces overlap by much less than the prior
55-unit minimum. It does not test real-map delivery.

The representative matrix was fixed before seeing results: overlaps 10.3,
20, and 30; gap widths 11, 15, and 19; and seeds 731 and 732. Every completed
120-second simulation records every 0.1-second frame with native rendering.

`run.py` executes the existing `depth5_test/run.py` fixture after two explicit
source substitutions: allow overlap down to 10.1 and write to this study's
results directory. The native simulator, controller, scoring, and recorder are
otherwise unchanged. The base source hash is stored in `summary.json`.

These 18 arranged cases can reject obvious short-wall failures, but cannot by
themselves demonstrate greater-than-95% reliability over native maps. A claim
of at least 95% reliability with a one-sided 95% exact binomial lower bound
requires 59 successes in 59 independent cases; failures increase that sample
requirement. Map-site availability (127/128) and retention conditional on a
site must be reported separately.

## Corrected interpretation at the 60%-remaining stop

These runs **do not test short-wall retention**. The fixture adapter relaxed
the harness assertion but kept the existing `ObservedGapPolicy` unchanged.
That controller rejects observed edge lengths outside 54.9--100.1 and rejects
overlap below 54.9. Consequently, in every case from overlap 10.3 through 54,
the controller found no station and never deployed the bait from its initial
position outside the mouth. The bait deaths at 7.1 seconds measure failure to
deploy under a long-wall-only controller, not failure of the short geometry.

This confound was discovered after the runs and all originals are preserved.
Each started run has a correction sidecar, and `audit-summary.json` records
the invalidation. The sharp 54/55 transition is the controller's explicit
software threshold; it is not evidence of a native collision boundary.

Matched positive controls at overlaps 55 and 65 survived the full 30 seconds,
acquired the predator, ended with it physically and jointly held, and recorded
no loss in all six cases. They establish that the unchanged controller and
adapter operate for the geometry the controller accepts, but say nothing
about shorter walls. The summary's `joint_success` is false for these positive
controls because its final-30-second tail includes the pre-arrival seconds;
the raw acquisition, loss, final-state, and survival fields establish the
stated short pilot outcome.

The quota sentinel appeared while overlap-70 controls were running. Those
three partial recordings were saved at 4.8--8.3 seconds and must not be
counted as completed controls. The overlap-90 controls were never started.

No conclusion about short-wall retention or greater-than-95% reliable map
coverage follows from this batch. The existing 127/128 count remains a static
geometry hypothesis only. It needs rerunning with a frozen controller that
actually recognizes overlap down to 10.1, followed by independent real-map
tests. A short-overlap-aware controller still needs to be frozen and tested;
no such variant was simulated before the quota stop.

Artifacts:

- `results/short_overlap_sol/summary.json`: 18 confounded crowd runs.
- `results/short_overlap_sol/frontier-summary.json`: 15 confounded
  single-predator frontier runs.
- `results/short_overlap_sol/positive-controls-summary.json`: six completed
  positive controls, three quota-stopped partials, and three unstarted cases.
- `results/short_overlap_sol/audit-summary.json`: corrected validity audit.
- `results/short_overlap_sol/replays/`: a native every-frame replay for every
  simulation that started, including failures and quota-stopped partials.

## Corrected v2 evidence

`short_overlap_policy_v2.py` freezes the same observation-only policy with its
minimum observed edge length and overlap relaxed from 54.9 to 10.1. It does
not receive fixture coordinates or dynamic predator state.

The deployment screen passed all 21 arranged cases: overlaps 10.3, 20, 30,
40, 50, 54, and 55 crossed with gap widths 11, 15, and 19. In every case the
policy mapped a station, moved the bait to depth 5, acquired the predator,
kept the bait alive, and ended the 30-second run with the predator held.

The 33-predator screen passed all 18 arranged cases: overlaps 10.3, 20, and
30 crossed with all three gap widths and two seeds. Every case acquired and
held 33/33 for 120 seconds with zero physical loss and a living bait. These
are prepared direct arrivals, not guiding tests.

An exact-native-map screen then selected the lowest-overlap clear candidate
at or above 20 on the first 12 census seeds, without result-based selection.
Maps 10000--10005 completed 120 seconds and passed 6/6 with 33/33 held and
zero physical or joint loss. The resource incident interrupted maps 10006,
10007, and 10008 at 84.3, 70.6, and 40.4 seconds; each held 33 with no loss at
the interruption but is a partial, not a pass. Maps 10009--10011 did not
start. `results/short_overlap_sol/native_v2/interrupted-audit.json` preserves
the exact classification and receipt hashes without loading large replays.

After the resource-safe streaming recorder became available, fresh capped
processes completed maps 10006--10011. The first-12 sequence therefore passed
**12/12** completed 120-second native-map cases: every map acquired and held
33/33 prepared arrivals, kept the bait alive, and recorded zero physical
loss. Maps 10008--10011 use event-streaming recorder v2 at 400-pixel native
render width; every 0.1-second frame remains present. Earlier interrupted
attempts remain separate and excluded from the denominator.

This is promising conditional-retention evidence, but twelve completed maps do
not establish greater-than-95% reliability. Unsupported maps must remain in
the denominator, and a one-sided 95% lower confidence bound above 95% needs
at least 59/59 independent passes. Full 3,000-second retention and actual
guide delivery on short native sites also remain separate requirements.

A 3,000-second prepared-arrival run on native map 10007 (overlap 20.263,
gap 17.937) was stopped gracefully at 1,587.5 seconds to stay below its hard
memory ceiling after the user reprioritized approach reliability. At the stop,
the bait was alive, all 33 predators had been acquired, zero physical losses
had occurred, and final joint hold was 33. This is positive partial evidence
only; it is neither a full-game pass nor a retention failure. The streaming
recording and receipt were published normally under
`results/short_overlap_sol/native_v4_stream/`.

The geometry explains why short walls can protect the bait. Native obstacle
collision uses a strict axis-aligned box expanded by the predator's radius
10, while a kill requires center distance below 15. For a gap below 20,
overlap at least 10, and bait at depth 5, the two expanded wall boxes cover
the open portion of the bait's radius-15 kill disk from 10 units before the
mouth through at least 20 units behind it. Thus a correctly placed bait is
geometrically unreachable in this ideal local configuration. This supports
bait safety; it does not prove indefinite predator retention, successful
guiding, or map-wide reliability.

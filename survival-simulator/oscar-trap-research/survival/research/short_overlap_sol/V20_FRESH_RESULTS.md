# Frozen v20 fresh-map batch

Policy hash: `279338f85da8051981fd6fec328dcbf8bfafdb18161a4106f3ba5cb165fb5d02`

The policy was frozen before declaring maps 10128–10143 and fixtures
20128–20143. Each supported case ran for 300 native seconds with infinite agent
energy, a predeployed depth-5 bait, an adjacent awake predator, 320-pixel native
rendering, and every 0.1-second frame retained. No case was excluded or tuned
mid-batch.

| Map | Outcome | Entry / delivery | Guide death | Notes |
| --- | --- | --- | ---: | --- |
| 10128 | pass | 8.7 / 8.7 | 8.6 | direct bait handoff |
| 10129 | pass | 86.9 / 87.8 | 226.3 | guide alive through delivery |
| 10130 | pass | 49.4 / 49.4 | 137.6 | guide alive through delivery |
| 10131 | pass | 41.1 / 41.1 | 40.8 | direct bait handoff |
| 10132 | pass | 24.9 / 25.4 | — | guide survived horizon |
| 10133 | pass | 45.7 / 45.7 | 45.5 | direct bait handoff |
| 10134 | unsupported | — | — | frozen selector found no >=55-overlap site |
| 10135 | pass | 24.0 / 24.0 | 23.9 | direct bait handoff |
| 10136 | pass | 64.2 / 64.2 | — | guide survived horizon |
| 10137 | pass | 35.9 / 36.8 | — | guide survived horizon |
| 10138 | fail | — | 1.0 | constrained river/top-boundary start |
| 10139 | fail | — | 7.9 | entered river with insufficient lead |
| 10140 | pass | 5.3 / 5.4 | 239.6 | guide alive through delivery |
| 10141 | pass | 23.7 / 24.1 | — | guide survived horizon |
| 10142 | fail | 273.5 / 273.7 | 153.9 | only 26.3 seconds remained; required tail is 30 |
| 10143 | pass | 14.1 / 14.1 | 14.0 | direct bait handoff |

The immutable result is 12/16 strict harness passes (75%). Among maps accepted
by the old site selector it is 12/15 (80%). Inclusive reliability is therefore
far below 95%. Every strict pass had zero physical and target losses after
delivery. Map 10142 acquired the bait target at 38.2 seconds, so its late entry
is not classified as an unrelated autonomous arrival; it nevertheless fails
the predeclared 30-second validation tail.

## Failure diagnosis

Map 10138 starts the guide in river terrain near the top boundary. The
lookahead is active continuously, and an instrumented two-second replay shows
that its first-step separation prediction matches native behavior to rounding
precision. For example, predicted versus actual separation is 40.0000 versus
40.0001 at 0.2 seconds and 21.6093 versus 21.6093 at 0.9 seconds. The planner
therefore fails through action selection rather than hidden simulator state or
first-step model error. A safety-weighted variant and a no-hold variant both
still die. The recorded v20 path reaches separation 20.1 at 0.9 and dies at
1.0.

Map 10139 repeatedly restores spacing until a terrain transition. At 7.6
seconds the guide is in forest with separation 33.7. It enters river at 7.7,
where the requested movement is strongly reduced; separation falls to 29.0,
then 17.3 at 7.8, followed by death at 7.9. This independently supports a
preventive terrain-transition lead requirement rather than another reactive
emergency threshold.

Receipts and native replays are under
`results/short_overlap_sol/v20_fresh16/`.

# Research results checked 2026-09-19

The last saved campaign checkpoint is 2026-09-19 04:08:37 UTC. It records
`status=inconclusive`, `stage=done`, generation 2.1, and six LLM reviews.
Live Runpod reads confirm all four campaign Pods are `EXITED`. The local
dashboard is showing a saved report because SSH is unavailable after shutdown.

## Completed reference-Python development comparisons

All rows below use the same eight development seeds, 0–7. These are reused
development maps, not fresh confirmation or final holdout results.

| Policy | Mean survival (s) | Worst survival (s) | Outcome |
| --- | ---: | ---: | --- |
| Starting configurable control | 596.5 | 122.7 | Initial incumbent |
| Generation 1.1 finalist | 865.6 | 33.4 | Rejected: regression and insufficient evidence |
| Generation 1.2 finalist | 822.2 | 122.8 | Accepted; still the selected best |
| Generation 1.3 finalist | 837.8 | 339.3 | Improvement over the new incumbent not established |

The accepted improvement is +225.7 seconds, or 37.8%, in mean survival.
Mean score rises from 633.58 to 848.89 (+34.0%). Full-horizon survival remains
0/8 at the 3000-second horizon. The worst map is effectively unchanged.
The recorded paired improvement interval for generation 1.2 is
[107.9, 346.5] seconds; repeated development exposure limits generalization.

The accepted policy is candidate
`dc61d16c5a37ca55df8eaaf4c8a3941211721c8667342a0ae14c84f9f9c92897`,
source snapshot
`5f1de1717a446f2a6404c546141578dc34000fb3c83893b95ead03a75be2551c`.
It enables emergency reproduction, sets maximum parent age to 70,
emergency reserve to 50, target fraction to 0.24547214066187814, and guide
bait buffer to 4.590298472442903. Safe delivery holds and role budgeting
remain disabled. The measured gain belongs to the combined package;
individual scalar effects have not been isolated.

## What the smaller experiments suggest

The generation 1.2 native four-map factorial measured mean survival of
468.45 seconds for the control, 682.68 for delivery-hold steering, 1039.58
for bounded reproduction, and 1014.83 for both. Reproduction was the stronger
change on this panel, and adding delivery steering to it did not help.
Focused tuning brought the selected native screening result to 1078.60 seconds.
These native values are not directly comparable with Python scores.

The generation 1.3 fleet package improved mean survival by only 15.6 seconds
against the newer incumbent in Python, with recorded paired interval
[-224.2, 303.4]. It improved the worst observed map but did not establish a
mean gain. Measured policy CPU per 1000 agent decisions increased from
2.081 to 3.788 seconds, about 82%. This is package-level cost on differing
trajectories, not an isolated function benchmark.

All three satellite studies completed 16 candidates and 16 Python validation
cases each. Their results informed subsequent reviews; they do not independently
promote a policy over the main campaign's newer incumbent.

## Incomplete work

Generation 1.4 encountered a missing diagnostic temporary file, and the major
search encountered a missing `progress.json.tmp` file, according to the saved
research reviews. These are infrastructure failures, not demonstrated policy
regressions. The saved major screening table contains 13 completed candidate
records, but no completed main Python promotion comparison from that stage.

The campaign reached generation 2.1 and then ended with an inconclusive final
assessment. A final selection was frozen and the final-holdout step is marked
finished, but the overall assessment is incomplete. The dashboard deliberately
does not collect holdout results; no held-out improvement is claimed here.
The exact final stopping reason is absent from the cached dashboard and still
requires the terminal report on the stopped Pod's persistent volume.

## Cost and sources

Runpod billing returned $9.526859734942263 for 2026-09-18 through the current
2026-09-19 billing bucket: $9.502554179205617 for the four Pods and
$0.024305555736646056 for network storage. Billing can settle later and retained
storage continues to accrue charges. The authorized cap remains $40.

Sources: `runs/runpod-launch-20260918/dashboard-latest.json` (saved comparisons,
decisions, research reviews and final checkpoint); live Runpod Pod status,
Pod billing and aggregate billing reads. No experiments were launched or
holdout outcomes read for this report.

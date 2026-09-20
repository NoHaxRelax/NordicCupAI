# Action-bug pipeline handoff

20 September 2026. Branch: `survival/action-bug`.

## Scope and hard boundaries

This branch contains the duplicate-action negative-energy transfer pipeline, the observation-only contact predictor, the score ceiling, C++-engine diagnostic tools, and compact evidence. It is validation-only work. Do not call the competition evaluation endpoint.

- Keep every validation result below 2,500 score. The current code can cap predicted transfer payout, but a conservative ordinary-score guard and margin are still required.
- Submit one validation at a time and check the live queue first. There is no cancellation mechanism.
- Do not use the laptop for heavy simulation, replay processing, or catalog scans. Oscar requested Hetzner for this work.
- Preserve the Hetzner EUR 25 spending cap and automatic cutoff. Never read, expose, or modify `/root/.hetzner-cutoff-token`.

## Current result

The final `predict_contact` selector transferred 34 of 34 bursts on four untouched seeds, compared with 25 of 45 for `contact_nearest`. Mean score rose from 7,576 to 9,872, while mean survival changed only from 1,248.2 to 1,269.4 seconds. Across ten final-version games, it transferred 85 of 86 attempts, or 98.8%.

This establishes much more reliable transfers in local games. It does not establish longer survival, HTTP capacity, or hosted parity. The one remaining final-version miss came from a wall that the observer had never seen, which redirected the predator.

The selector improves reliability by:

- registering observer motion from static edges;
- associating consecutive predator sightings and rejecting ambiguous or stationary tracks;
- forecasting both the unobserved prior predator move and the pending move;
- accounting for turn limits, competing prey, slower terrain motion, target-biome crossing, and remembered walls;
- requiring one unit of clearance inside the strict 15-unit contact radius;
- confirming a transfer from the next public score plus disappearance; and
- remembering confirmed frozen-predator locations so later agents are not sacrificed to sleeping predators.

The implementation uses only public observations and prior actions. Hidden engine state is used only by the diagnostic harness to classify outcomes.

## Score safety

`Harvester.apply(..., payout_limit=...)` reduces the turn count before a burst so its predicted transfer payout fits the remaining score budget. Both the legacy selector and `predict_contact` fail closed on non-finite or non-positive limits.

`night_agent_server.py` adds:

- `NIGHT_SCORE_CEILING` for the absolute pre-burst score budget;
- `NIGHT_TRANSFER_SCORE_MARGIN` for unmodelled same-tick score; and
- the existing `NIGHT_SCORE_GUARD` plus `NIGHT_SCORE_RESERVE` to stop ordinary collection.

The endpoint idles all live agents when the score is missing, non-finite, or too close to the ceiling. The local driver mirrors this with `--score-ceiling` and `--score-margin`.

Before any scored validation, run a fresh Hetzner-local seed set with a target below 2,500. A conservative starting point is ceiling 2,350 with at least 150 points of margin. This is a starting configuration, not a validated guarantee.

## Validation API evidence

One clean 400k validation completion has been established. A 1M payload was not submitted to the validation API.

1. Attempt `e821c86f29cf46b1ba03a94f36bf32d2` emitted 400,000 actions, then an unexpected package-service restart interrupted the endpoint. It finished with score 76.8792 and `Connection refused`. It is invalid as a capacity result.
2. Attempt `5443c40f9a3d431699cb72514e777962` ran from 06:12:49 to 07:17:09 UTC and finished with score 1339.9967 plus `Connection refused`. The endpoint logged 400k no-op bursts at 06:12:51, 06:22:29, and 06:47:26. A service restart at 06:47 caused the third emission. This run is also invalid as a clean one-shot capacity result.
3. Attempt `68c44bb352da413abde5e52298a912b4` started at 07:44:51 UTC after deployment of the one-service-lifetime guard. It emitted exactly one zero-cost 400k burst at simulated time 0.1. The next callback, at simulated time 0.2, arrived 15,119.303 ms after the 37 MB JSON response was fully serialized. The validation completed cleanly at 07:55:38 with score 2145.6172 and no errors.

The 15.1-second measurement includes response transfer plus the organizer's receive, JSON decode, action validation, action loop, next simulator step, and next request transfer. It excludes this endpoint's response construction because its timestamp is taken after `JSONResponse` has serialized the body. It is therefore a close upper bound on the organizer-side post-response work, not a pure parser benchmark.

The public reference uses a 10-second timeout while waiting for the agent HTTP response, but parses and applies received actions afterward in a sequential loop. The current validation service reports a 1,200-second accumulated wall-clock response-wait limit. This is not the simulated-time horizon: the game can simulate 3,000 seconds over 30,000 ticks, and a huge action list still advances only one 0.1-second tick.

Hetzner-local JSON and Pydantic boundary measurements were:

| Actions | Payload | Serialize + decode + validate | Peak RSS |
| ---: | ---: | ---: | ---: |
| 400,000 | 37.2 MB | 1.276 s | 600.6 MB |
| 1,000,000 | 93.0 MB | 3.210 s | 1,419.6 MB |

Those measurements exclude the organizer's simulation action loop. The clean 400k result characterizes that exact payload once; it does not justify another 400k or a 1M validation probe.

## Live Hetzner state at handoff

The endpoint is active at `http://46.62.244.29:9052/predict`, using `/opt/nightserve/nightsim/serve/pred_best.json` and the native C++ runtime.

The live service has this capacity configuration:

```text
NIGHT_HARVEST={"budget":20000,"max_harvests":12}
NIGHT_NOOP_BURST=400000
NIGHT_NOOP_MAX_BURSTS=1
NIGHT_NOOP_LATEST_TIME=5
NIGHT_SCORE_GUARD=2400
NIGHT_SCORE_RESERVE=130
NIGHT_TRANSFER_DELTA=50
```

The live source includes the service-lifetime one-shot guard, score-ceiling support, and the callback-delay measurement from commit `6b6fc53`. The completed probe has consumed its one allowed burst. A service restart clears in-memory state and arms another burst, so do not restart or submit another validation in this state.

The deployed endpoint reports `service_once: true`, `sent: 1`, and `next_callback_gap_ms: 15119.303309`. The score guard remained below its trigger and the terminal validation score was 2145.6172. Before any policy validation, disable the no-op drop-in, retain the score safeguards, syntax-check, restart only while no validation is active, then verify the root endpoint reports `actions: 0`.

## Code and evidence map

- `nightsim/serve/contact_predictor.py`: observation-only transfer admission.
- `nightsim/serve/contact_predictor.md`: integration and limitations.
- `nightsim/serve/harvest.py`: selector routing and payout cap.
- `nightsim/serve/night_agent_server.py`: validation endpoint, one-shot no-op mode, score guard, and latency logging.
- `nightsim/serve/sim_harvest.py`: paired local C++-engine driver with selector and score-cap options.
- `research/action_bug_debug/diagnose.py`: recorded full-game diagnostic harness.
- `research/action_bug_debug/test_contact_predictor.py`: ten public-observation regression tests.
- `research/action_bug_debug/summarize.py`: cohort aggregation.
- `results/action-bug/contact-predictor-20260920/REPORT.md`: final analysis.
- `results/action-bug/contact-predictor-20260920/summary.json`: exact aggregate metrics.
- `results/action-bug/action_burst_bench.cpp`: standalone native C++ burst benchmark.
- `results/action-bug/capacity_bench.py`: JSON and Pydantic boundary benchmark.
- `results/action-bug/reference_step_bench.py`: published Python reference-step benchmark. It requires the reference dependencies, including pygame.

Per-game JSON traces are included. The 460 MB compressed replay set is intentionally not committed. It remains in the unversioned research workspace under `survival/results/action-bug/**/replays/`. The 30 new recordings were retrieved but not catalog-verified because laptop-heavy catalog processing was stopped at Oscar's instruction.

## Reproduce

From `survival-simulator/oscar-overnight-cpp` after building the native extension:

```bash
python3 nightsim/build.py
python3 research/action_bug_debug/test_contact_predictor.py
python3 nightsim/serve/sim_harvest.py \
  --config nightsim/serve/pred_best.json \
  --seeds 401-404 \
  --variants base,harv \
  --selector predict_contact \
  --budget 200000 \
  --max-harvests 100 \
  --contact-margin 1.0 \
  --workers 2 \
  --out /tmp/action-bug.jsonl
```

For a score-capped local trial, add `--score-ceiling 2350 --score-margin 150`. Run heavy trials on Hetzner, not the laptop.

## Recommended next sequence

1. Run the regression tests and a fresh, score-capped C++-engine seed set on Hetzner.
2. Compare normal population control with the requested reduced-population variants. Prior no-predator evidence says aggressive culling and minimal populations were worse, so do not assume fewer agents survives longer.
3. Disable the live 400k no-op configuration while the validation queue is empty. Retain the deployed one-shot, predictor, and score-cap code.
4. Verify the exact endpoint configuration and response size locally.
5. If the local capped distribution stays safely below 2,500, submit only one validation and wait for its terminal result before any further run.
6. Do not repeat the clean 400k capacity probe or attempt 1M against the validation API without a materially new, organizer-approved question.

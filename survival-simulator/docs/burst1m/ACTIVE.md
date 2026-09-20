# 1M burst strategy and monitoring

The user replaced seed-recovery development with the fast1m strategy from the
“Bug-performance” task. Source: codex/burst-score, commit8721a3d. Only its native
runtime and serving dependencies are imported; the other task/worktree is unchanged.

Selected configuration: docs/burst-score/local-winner.json. Budget1,000,000 repeated
finite turn actions; cooldown0; predict_contact selection;100 maximum harvests.
Pilot evidence:68 maps, mean45,276.9,95%CI42,754.9–47,903.0;68/68 above20k;
597/601 successful transfers. These are selection results from local simulations,
not a hosted guarantee. No official1M attempt has been submitted here.

Status: PREPARED, NOT RUNNING. Awaiting the user's answer: local simulations or
hosted validation on Hetzner. Earlier instruction forbids official validation;
do not infer a reversal merely from “change strategy”. The question is pending.
The imported wrapper endpoint passed an in-process health/empty-game check only.
No live service/configuration has been modified and no game was started.

Entrypoint: models/burst1m/server.py; build oscar-overnight-cpp/nightsim/build.py first.
It explicitly selects the winner, disables inherited no-op capacity probes and
score ceilings, and uses the public-input NightPolicy adapter with fixed seed0.
The native simulator seed never enters the policy adapter. It retains Oscar's
harvest implementation and contact predictor unchanged.

GET /health reports callback time, score and delta, simulation time, living agents,
bursts, confirmed transfers, response bytes/time and errors. Per-request telemetry
is written to BURST1M_HEALTH_LOG (default/tmp/burst1m-health.jsonl); set a persistent
path for deployment. Likewise set BURST1M_LATENCY_LOG. Run-health is awaiting_game
until requests arrive; the local empty-game smoke test is not an actual run.

User explicitly asks continual run-health updates. The existing heartbeat is
repurposed to monitor every minute once a run is active and report score trend,
transfers, population, response latency/errors/stalls. Do not keep doing seed
research or duplicate jobs. At completion report final score and run health and
pause the monitor. Preserve all existing production endpoints until run scope is
resolved and deployment is ready. Never SSH pc.

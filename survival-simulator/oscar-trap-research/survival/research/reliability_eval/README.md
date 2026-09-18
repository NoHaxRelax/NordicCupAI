# Frozen guide-delivery reliability evaluation

This protocol uses 59 fixed, fresh map/fixture seed pairs. Do not edit the
protocol or scorer after looking at these outcomes; any revision requires a new
seed set and experiment identity. The denominator is always 59. Unsupported
maps, failures to start or finish, missing receipts, invalid replays, and scorer
errors are failures.

A pass requires every native frame, a completed 300-second horizon, stable live
bait, native chase-gate evidence that the predator followed the active guide, a
guide-target to active-bait-target handoff within three seconds at the intake,
an active bait target while physically contained, two seconds of confirmation
that may include native rest after that active handoff, acquisition by 270
seconds, and continuous containment through the 300-second horizon. Rest alone
cannot start acquisition.

Guide survival is neutral. A living guide may withdraw once the predator has
switched to bait, while guide sacrifice near the bait is also valid. The test
does not require a particular guide outcome. A separate controlled policy test
would be needed to compare deliberate withdrawal with deliberate sacrifice.
The scorer reports `guide_release_observed` and `guide_returned_to_staging`
policy events as separate survival/release evidence.

Run only after the policy module and class are frozen:

```bash
python research/reliability_eval/run_frozen.py \
  --policy package.module:PolicyClass --workers 1
```

The runner invokes the existing v2 streaming recorder, keeps every 0.1-second
native frame, caps each worker at 3 GB RAM, and preserves every receipt/replay.
It freezes the recursive import closure of local Python policy dependencies, so
an imported mixin or helper cannot change between cases unnoticed.
It never calls competition services. This is a single already-engaged predator
test; it does not measure 33 sequential deliveries or full-game retention.
At 59/59, the exact one-sided 95%
Clopper-Pearson lower bound is about 0.9505, just above the requested threshold.

## Robust safety candidate

`policy_robust.py` preserves v21 and subclasses it with a conservative safety
planner. Apparent stationarity never disables escape logic. Each prediction
depth branches the unknown native state across 0, 11, and 15 units, checks the
candidate against every child, and only then clusters the surviving uncertainty
to six geometric extrema. The score retains progress toward the current route
stage and penalizes loss of the hearing envelope.

This conservatism can stall or take a longer route when contact is uncertain.
That trade is intentional: with no public predator energy/rest state, treating a
wall-blocked predator as safely asleep can kill the guide after it unsticks.
Future watched-pivot behavior is not modeled. The hearing penalty steers plans
toward 55 units and only the first action is used before replanning; predictions
beyond the native 90-unit unconditional chase gate should be treated as low
confidence rather than exact.

`benchmark_policy_robust.py` exercises only a synthetic public-DTO-derived
planner state. It does not instantiate the simulator. After correcting the
zero-speed branch to preserve native resting heading, the recorded v2 25-call
wall-adjacent check had a 52.6 ms median and 71.5 ms maximum on this machine.
The v1 benchmark receipt remains preserved.

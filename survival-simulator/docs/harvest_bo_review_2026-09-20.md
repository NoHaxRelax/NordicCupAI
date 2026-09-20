# BO launch review

Requested objective: maximize **overall mean + mean of the lowest 10**, both
over the same **1,000** complete game scores. Keep both components, the
worst-100 mean and minimum as diagnostics. Validate finalists on fresh seeds.
Keep duplicate actions at the minimum required for horizon-long sleep.

Reviewed source: adjacent `NordicCupAI-bug-validation/survival-simulator/
exploit_lab/native/{bo_worker.py,bo_launch.py,build_harvest.py,harvest_controller.inc}`.
These source files have not been modified in the adjacent checkout.

## Launch blockers found

- Worker maximizes a lexicographic feasibility/drain-cost objective, not tail score.
  Its tail is worst 100, not the newly requested worst 10.
- Worker searches 2,000–300,000 duplicate actions, including insufficient sleep
  debt, instead of deriving the minimum safe drain from the remaining horizon.
- Worker seeds policy RNG from the world seed and does not forward base policy
  parameters: `policy_init(seed_key(s), {})`.
- Pods use different world-seed cohorts, making direct candidate comparisons
  confounded. Use common world seeds and different optimizer sampler seeds.
- Default study is in memory and exports only at completion. Persist trials
  incrementally and save per-seed scores with exact configuration/provenance.
- Export considers any non-null trial value, including pruned partial trials.
  Rank only COMPLETE trials with exactly 1,000 unique finite game results.
- Launcher uses Linux shell commands on the Windows host, incorrect default
  checkout traversal, and a build pipeline that can hide compilation failure.
- Pod cleanup uses name prefixes rather than a manifest of task-created IDs.
- The new native controller is an older implementation: distance closing is
  treated as predator motion without correcting observer movement; sleepers
  are not filtered; retries drain again; mortality-chain prediction and the
  current sleep certification logic are absent. It cannot substitute for the
  reliability-tested controller without porting and regression verification.

## State

The search-space document now specifies overall mean plus worst-10 mean. A C++ objective helper and
regression test check exact tail arithmetic and reject partial/non-finite
cohorts. They are not yet connected to the reviewed BO worker. No six-pod BO
launch has occurred. Launch must wait for the controller and worker fixes and
a successful end-to-end smoke test.

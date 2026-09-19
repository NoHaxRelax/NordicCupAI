# Upstream refresh before recovery

Fetched origin on 2026-09-19 at approximately 06:42 UTC. The working branch
`sim-optimization` has no tracking branch and contains extensive intentional
uncommitted work. Remote refs were updated without merging unrelated challenge
changes or replacing that work. `origin/main` remains `c598102a`.

## Lucas: 37a3418f8e2f3066b0cc59abf064220baff0f5ac

The five-minute bridge installed this commit on the recovery Pod at 06:47 UTC.
Its pinned source is
`/workspace/lucas-bridge/37a3418f8e2f3066b0cc59abf064220baff0f5ac/source`.
The campaign is still prepared at generation 2's opening boundary; the normal
boundary fetch will read the current mirror before the coding review.

New policy changes include configurable overlap for bait replacements, release
and reassignment of stalled replacements, predator avoidance while travelling,
wall-aware bystander vision, restored Orchard reproduction, and the tuned
Orchard module/configuration. The upstream replay is a single visual inspection
game, not evidence of a reliable whole-game improvement. The final documentation
commit `37a3418f` leaves the tested `af647be5` policy bytes unchanged. It reports
extinction at 503.6 seconds, score 503.23 versus 632.18 for the previous same-seed
baseline, and 95.3 seconds of estimated bait gaps (longest 68.9). Continuity is
still unresolved; do not describe these additions as proven improvements.

An isolated overlay onto the accepted candidate's trusted harness ran 117 tests
and failed to import one test module because upstream bystander avoidance lacks
our `DEFAULT_CONFIG`. The other nine tests in that module therefore did not run.
The existing accepted candidate previously passed all 126 trusted tests.
Receipts and the complete test log are under
`/workspace/recovery-ops/upstream-checks/af647be56d74a66b9e50823c2ac97e43fdb07cb3/`.

This requires selective adaptation, not a wholesale replacement:

- Preserve the configurable avoidance signature and `DEFAULT_CONFIG`, corner
  selection option, observation association, decision traces and guide tuning.
- `ExperimentalEntrapment` overrides the main action loop and Orchard instance.
  Copying only `core.py` does not activate all the new reproduction or travel
  behavior in the optimizer's actual policy. Adapt those paths deliberately.
- Evaluate replacement continuity and survival/reproduction separately, then
  their combination. Preserve the accepted bounded-reproduction configuration.
- Measure bait gaps, replacement travel deaths and stalls, captures maintained,
  survival, absorbed fruit energy and policy CPU. The additional wall intersection
  checks may add compute; their cost has not been measured in our campaign.

## C++ references

Pinned source archives are available locally in
`runs/runpod-launch-20260918/upstream-refresh/` and on the Pod under
`/workspace/recovery-ops/upstream-references/`.

- `oscar-fastsim` at `51ca0680826c7f7c1428b4642f8f0c9e8372b403`
  adds a native Orchard policy and an entirely C++ policy/engine loop. Its
  upstream speed claim applies to that Orchard policy, not our combined
  exploration/trapping controller. The current campaign already uses the C++
  environment for screening. Replacing the combined controller needs separate
  observation-boundary and decision-equivalence verification, plus a new
  frozen protocol if engine/evaluator code changes.
- `oscar-overnight-cpp` at `5d8022a11319692425937bb5d4d4971eadd2c61d`
  includes a conflict-aware wall map and diagnostics. Its notes report better
  map/site validity, but no conclusive full-game survival gain from refuge.
  These are results from another controller. Investigate the analogous failure
  in our mapping before importing the mechanism or claiming the same effect.

The supervisor receives these findings as a cumulative development journal
entry, including the pinned reference locations. The frozen campaign source,
candidate identities, previous best and holdout remain unchanged. No optimization
or final evaluation was launched during this refresh.

After this refresh, the user authorized credential installation and recovery
resumed at 06:54 UTC. See [the recovery handover](research_recovery_20260919.md)
for the running supervisor, shutdown guard and launch receipt.

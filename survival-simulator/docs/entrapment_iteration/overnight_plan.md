# Overnight entrapment research — 18–19 September 2026

Deadline: 08:00 Europe/Copenhagen on 19 September (06:00 UTC).
Work on `survival-simulator/lucas-experimental`; push useful checkpoints.

## Two separate metrics

1. Attempt success: deliver one new predator to 30 already held predators,
   retain the group, keep the rear replacement entrance clear.
2. Map coverage: find at least one candidate site whose attempt success rate
   exceeds 50%. A single success on a map cannot establish this.

`trap_site_coverage.py` screens several crevices and corner pockets per map,
then evaluates the selected candidate on independent encounter seeds. Report
both the empirical fraction over 50% and the stricter fraction whose per-site
Wilson 95% lower bound exceeds 50%. These bounds are not simultaneous bounds.
Candidate search is capped: missing a good site does not prove none exists.
Selection is an offline evaluation of site potential, not a claim that the live
scouting policy already knows which candidate wins.

## Work allocation

- Root: repeated-site evaluation, Runpod, integration, hourly branch review,
  source archives, results, and promotion decisions.
- Sol corner investigation: native evidence for additional corner arrangements,
  preserving physical rear access; isolated scratch work.
- Sol delivery investigation: trace-driven explanation of failures and one
  isolated observation-only candidate; include controls in the paired sample.
- Promote candidates only after paired checks and independent encounters.
  Keep Oscar's survival module unchanged except for deliberate integration of
  verified upstream updates. Native-game checks follow promising isolated work.

## Compute and usage

Runpod cap remains $10 total. The new 16-vCPU CPU3 pod costs $0.48/hour and is
recorded in `runpod-budget.json`. Download evidence before terminating it.
Prefer sustained independent-encounter batches over many tiny tuning rounds.
Scale only when the measured throughput and remaining budget justify it.

Codex account quota is available in local session `rate_limits.primary` events;
previous notes saying it was unavailable were wrong. Monitor the newest event
and clock, not just the goal token counter. Pace investigations against hours
remaining, using longer experiments if quota is falling too quickly. Leave
room for collecting results and handover. Do not publish account session data.

Every hour, fetch all remote branches and inspect changed survival-related
files on any branch, not just Oscar's named branch. Record the last checked
remote revisions locally; inspect upstream changes before merging/adapting.

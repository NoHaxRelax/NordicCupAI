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

## 20:42–20:48 UTC checkpoint: new upstream and free compute

- Latest observed account quota: 49% used, 51% remaining. Estimated Runpod
  spend $0.223 at 20:42 UTC; active jobs continue under the $10 cap.
- Fetched all branches. Orchard upstream advanced to `a7a7d63484d9bc1af956727175fe2c773ee7203a`;
  new C++ branch `survival-simulator/oscar-fastsim` is `696bbd86272c27d9edba56238d3ec3b9469faaf9`.
  Upstream reports roughly 2680 mean score without predators across 32 fresh
  runs. This is upstream evidence, not our integrated agent's measured score.
- `ssh pc` works: 12 logical CPUs, approximately 10 GiB RAM. Dedicated work
  directory `~/entrapment-research-20260918`, Python 3.13.14 and pinned packages.
  Six fully recorded native games now compare an isolated upstream candidate
  against the first six seeds of our baseline native batch (seed stream
  2026091902). Six workers leave capacity for fast-engine screening.
- Candidate imports Oscar's exact module and winning JSON configuration. It
  removes our age-55/old-age birth veto for gatherers and explorers: that veto
  contradicts upstream's heir age 55.9082. Bait and guides still cannot spawn.
  Native engine reproduction requires energy, not an age-under-55 rule.
  This candidate is isolated; production remains unchanged pending evidence.
- Sol verified short predator-enabled C++/Python lockstep and measured engine
  steps about 26–29 times faster, but only about 1.5 times end-to-end with our
  policy. Controlled fixtures require mutation APIs absent from C++ snapshots;
  retain Python for 30+1 experiments. Build an isolated C++ full-game screening
  harness first, preserving Python full-frame replay and final verification.
- Corner work now prioritizes a previously rejected map with a safe diagonal
  pocket and actual successful replacement; randomized normal-policy trials
  are required before claiming added reliable map coverage.
- Staged front approach gave 10/12 versus 9/12 on reused development cases,
  including a new retention failure. Do not promote from this small result.

Next allocation: finish ongoing independent site validation and paired guide
comparison; use the free PC for survival integration and C++ screening. Spend
remaining paid compute on larger independent entrapment validation after
selecting a candidate. Reserve final hours for ten recorded games, artifacts,
visualization, version control, and paid-resource cleanup. The requested 95%
success is a target, not a result currently established.

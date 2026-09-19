# Operator priority: reliable predator capture

Guidance ID: `predator-capture-20260919-01`.
User request: give the optimizer more focus on making predator capture work well.
This is a research priority, not a measured result or a change to the objective.

At the next available review, give predator capture an explicit place in the
research plan. Prefer one of the available hypothesis slots for a bounded
capture/containment improvement or an isolated test of the existing capture
system. Keep room for feeding, population replacement and other survival work.
If evidence argues against a capture experiment this round, explain why and
identify the next useful capture investigation; do not silently omit it.

Inspect the complete capture chain: finding suitable traps, assigning guides,
attracting and keeping predator attention, delivering predators to traps,
confirming containment, maintaining bait, and replacing bait without escape.
Pay particular attention to guide deaths, abandoned deliveries, predators
escaping during handoff, and starving or aging bait. Consider whether role
budgeting releases guides or bait too early, or leaves too few workers to feed
and replace them. Test interactions with the current reproduction/role-budget
variants using a small baseline/A/B/A+B comparison when feasible.

Use development diagnostics to ask:

- How many predators were reliably contained, and how long did it take?
- How long did containment last, and what caused escapes or failed recapture?
- How many attempts succeeded, timed out, or lost their guide/bait?
- Did bait replacement preserve containment continuously?
- How many agents died during capture, and how much energy, worker time and
  policy compute did it consume?
- Did better containment reduce later predation and improve mean/worst survival?

Distinguish a policy claim of capture from observed sustained containment.
State definitions, observation windows, missing events and uncertainty. A
stationary predator or a bait-role assignment alone is not proof of capture.
Use recorded policy state, event clips and spectator screenshots to diagnose
failures, but keep policy decisions based only on public observations. If
coverage is insufficient, state which evidence is missing; do not fabricate
success rates or modify the frozen evaluator to obtain them in this campaign.

At major boundaries, explicitly review capture-system reliability and complexity,
and give compatible Lucas-branch capture improvements particular attention.
Consider a structural module change only when repeated failures support it.

Keep the existing survival/score promotion rules, Python validation, immutable
candidates, previous best, four-subgeneration schedule with adaptive stopping,
$40 campaign budget, shutdown deadline and untouched final holdout. Capture
metrics help choose and explain experiments; they do not override the objective
or justify promoting a policy that fails the existing survival checks.

In the review's `evidence`, acknowledge this guidance ID as an operator priority,
separately from empirical evidence. In `direction`, say what capture work is
being tested or why it is deferred. The resulting review becomes part of the
cumulative research journal.

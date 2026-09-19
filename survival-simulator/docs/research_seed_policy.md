# Seed diversity and generalization

Guidance ID: `seed-diversity-20260919-01`.
User instruction: do not keep optimizing against the same few seeds.

Treat seed overfitting as a research constraint. Repeatedly tuning against a
small panel can exploit those maps even when no seed-specific behavior is coded.
Fresh seeds help measure generalization; rotation alone does not remove the
effects of repeated adaptive selection.

## Required schedule for the next protocol version

- Predetermine and persist the seed schedule before observing its outcomes.
  Use a reproducible schedule RNG and record panel IDs, purpose, assignment and
  prior exposure. Never reroll a panel because its results are unfavorable.
- Give each subgeneration a fresh development screening panel. All candidates,
  ablations and their incumbent control in that comparison use the same seeds.
  Focused tuning may reuse the panel inside that subgeneration; subsequent
  subgenerations receive fresh panels. Preserve the configured number of games
  per candidate unless a separate budgeted change is justified.
- After selecting a finalist on screening evidence, evaluate that frozen
  candidate and the previous best on a separate, fresh paired comparison panel.
  Do not repeatedly test alternatives against that panel until one passes.
  Once results have informed a decision or LLM review, mark the panel exposed.
- At major boundaries, use fresh screening seeds for broader BO and a larger
  fresh comparison panel for the selected challenger. Compare Lucas imports
  against the same incumbent on a preassigned paired panel. Distinguish all
  screening, selection and confirmation evidence in the journal.
- Keep the final holdout disjoint from every development panel and unopened
  until final selection is frozen. A failed holdout is a reportable outcome,
  not permission to tune against it or try another candidate on it.
- Log every assigned game, including failures, timeouts and unfavorable seeds.
  Report panel size, number of distinct seeds, candidate exposure, paired gains,
  variability and worst cases. Do not compare raw means across different panels
  as if the difference were a policy effect. Do not count repeated games on the
  same seeds as additional independent maps.

A small fixed regression panel may still be useful for debugging, but it must be
labelled as reused evidence and cannot establish generalization. Keep the policy
observation-only; seed IDs and panel membership belong to the evaluator.

## Current running campaign limitation

The existing `research-night-1` protocol uses fixed screening seeds 0–3,
comparison seeds 0–7 and major-boundary seeds 0–11. The parallel capture studies
also reuse development panels. They do not currently implement rotation, and
their repeated results must not be presented as independent generalization
evidence. The separate final holdout remains protected.

An LLM review cannot change evaluator scheduling through a policy proposal.
Do not edit the frozen campaign configuration, historical candidate snapshots,
existing results, trusted evaluator or seed metadata to simulate compliance.
The rotation requirement needs a new, tested scheduler/protocol version and a
recorded transition. Preserve the previous best, journal, $40 total budget,
shutdown guards and final holdout through that transition. Until then, explicitly
label reused-panel results as development evidence.

For each subsequent review, acknowledge this guidance ID in `evidence` as a user
instruction, separately from measured results. State the actual panels used,
whether evidence is reused, and any missing fresh-seed validation in `direction`.
Never claim rotation has been implemented based only on reading this document.

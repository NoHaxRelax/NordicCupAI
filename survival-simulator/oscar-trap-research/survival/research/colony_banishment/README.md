# Managed colony and temporary predator banishment

Start with [the report](../../../docs/survival-colony-banishment.md). This is
personal Nordic AI Cup strategy research, local only. It is independent of the
mechanics bug hunt, wall acquisition and water baiting work.

## Current artifacts

- `policy.py`: frozen version 5, observation-only colony and optional banishment.
- `run.py`: original-physics runner, fixture parameters, measurements and replays.
- `suite.py`: frozen held-out generated maps, fixtures and oracle diagnostics.
- `summarize.py`: paired score, survival, harvest, exposure and cost summaries.
- `checks.py`: interface/measurement/source checks and replay schema checks.
- `adapter.py`: factories for the existing debugger recorder.
- `recording_support.py`: mandatory recording and unique, atomic result files.
- `backfill.py`: resumable reproduction of historical metrics-only runs.
- `verify_catalog.py`: checks every colony replay against the live lab catalog.
- `policy_v1.py` through `policy_v4.py`, matching runners and `ITERATIONS.md`:
  preserved failures and intermediate algorithms.

The policy receives a list of ordinary cached agent DTO dictionaries and time.
It returns exactly one finite `ActionRequest` per living observed agent. It has
no environment handle. The optional `oracle_poses` input is used **only** by
separately labelled oracle runs. Relative map frames and static landmark memory
are estimates, not known world coordinates or terrain maps.

## Reproduce

Run from the repository root using the existing interpreter:

```sh
survival/.venv/bin/python survival/research/colony_banishment/checks.py
survival/.venv/bin/python survival/research/colony_banishment/suite.py generated --workers 2
survival/.venv/bin/python survival/research/colony_banishment/suite.py fixtures --workers 2
survival/.venv/bin/python survival/research/colony_banishment/suite.py oracle --workers 2
```

The suite verifies the frozen policy hash and archived strategy harness hash
before running. Recording instrumentation has its own current harness hash.
It writes into `survival/results/colony_banishment/`; rerunning a phase preserves
existing results and adds a unique suffix. Complete frozen parameters and seeds are in
`heldout-v5-freeze.json`. Version 5 canonicalizes observation order and numeric
precision, after version 4 exposed null-arm divergence. This does not couple
future policy-dependent engine RNG consumption or establish Linux parity.

For a shorter new experiment with a fresh output name:

```sh
survival/.venv/bin/python survival/research/colony_banishment/run.py \
  --scenarios forest --seeds 22 --modes control banish \
  --horizon 180 --native-render --out local-example.json
```

For a historical algorithm, use `run_v1.py` through `run_v4.py` and a new output
name. Their policy imports/hashes point to versioned snapshots. `suite_v4.py`
replays the superseded protocol; its historical import repair changes the runner
hash, while the policy hash must still match the saved freeze.

Every run is recorded, including controls, failures, nursery comparisons,
oracle diagnostics and sweep cases, across all five saved versions. `--replay`
is retained as a compatibility flag; it is no longer necessary. Lightweight
state recording is the default. Use `--native-render` for close visual inspection
with the upstream simulator drawing methods. `--every 10` stores ordinary frames
every second, while the recorder checks every step and retains critical events.

Each run saves a uniquely named replay under `replays/` and durable metrics under
`run-results/` before it reports completion. The live Survival Lab automatically
discovers these files; use **Refresh list** after the server scan. This research
does not edit the shared debugger or catalog. The offline HTML export is a
separate snapshot maintained by the coordinating task.

## Historical replay coverage

The original inventory contains 229 reported runs, of which 20 had recordings.
The remaining 209 configurations are reproduced using their saved policy version,
seed, scenario, settings and horizon. Every new recording is labelled
**NEW REPRODUCTION** and links to its original result row. It cannot recover
the missing original frames. Original results remain unchanged, and outcome
comparisons expose differences, especially for the order-sensitive older versions.

```sh
survival/.venv/bin/python survival/research/colony_banishment/backfill.py --audit-only
survival/.venv/bin/python survival/research/colony_banishment/backfill.py --workers 4
survival/.venv/bin/python survival/research/colony_banishment/verify_catalog.py
```

Backfill resumes from completed `run-results/` files. The current coverage and
per-run comparisons are in `replay-coverage-audit.json`. `live-catalog-verification.json`
records the last live check. `archives/` retains the pre-instrumentation runners
as text; `recording-adoption.json` identifies their hashes. The original research
tables continue to describe the original evaluations, not the later reproductions.

To record an ordinary generated-map run using the debugger's existing CLI:

```sh
survival/.venv/bin/python survival/debugger/record.py \
  --policy "$PWD/survival/research/colony_banishment/adapter.py:make_banishment" \
  --seed 201 --seconds 300 --every 5 \
  --output survival/results/colony_banishment/replays/adapter-example.json.gz
```

Use `make_colony` for the same colony with banishment off. Neither factory is a
hosted deployment or a submission.

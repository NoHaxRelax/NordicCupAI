# Sprinter relay iteration

Main findings: [../../../docs/survival-sprinter-relay.md](../../../docs/survival-sprinter-relay.md).

This is personal, local strategy research using the pinned upstream engine. No API calls, hosted validation, submissions, pushes, UI changes or renderer changes are involved.

## Current code

- `controller.py`: observation-only orbit acquisition, active/reserve assignment, rest-window handoff, reserve forage, paid births, capacity-aware newborn preparation, and independent pair ownership for two predators. `radial` and `mpc` retain the unsuccessful initial controller families.
- `run.py`: arranged geometry fixtures and diagnostic observers that delegate to the exact native engine methods. It asserts one action per observed living agent. It records actual predator decisions, captures versus starvation, nominal and received food energy, reproduction cost, target gaps, births, trait mutations and controller events.
- `evaluate.py`: six fixed held-out starting configurations; matched two-worker controls; equal-initial-population controls; two-predator pair tests; native recordings for the four representative demonstrations. Every sweep and control run also records automatically.
- `renewal_screen.py`: fixed tree stands versus original stochastic tree generation, over 180 seconds.
- `verify.py`: source turning-circle geometry, observation immutability/action contract, replay schema and time ordering.
- `summarize.py`: aggregate final results, excluding the invalid baseline and historical experiments.

Run from the repository root with the existing interpreter:

```sh
survival/.venv/bin/python survival/research/sprinter_relay/evaluate.py --phase heldout
survival/.venv/bin/python survival/research/sprinter_relay/evaluate.py --phase workers
survival/.venv/bin/python survival/research/sprinter_relay/evaluate.py --phase multiple
survival/.venv/bin/python survival/research/sprinter_relay/renewal_screen.py
survival/.venv/bin/python survival/research/sprinter_relay/evaluate.py --phase replays
survival/.venv/bin/python survival/research/sprinter_relay/verify.py
survival/.venv/bin/python survival/research/sprinter_relay/summarize.py
```

Every simulation run now records automatically, even when no `record` argument is passed. Recordings use unique policy/version/seed IDs under `survival/results/sprinter_relay/replays/`; per-run metrics use `runs/`, and immutable aggregate snapshots use `batches/`. Existing results are never overwritten. State recordings sample every 0.5 seconds while capturing every simulation step and retaining critical-event frames. The demonstration phase uses the original native renderer. The historical top-level summary files remain the evidence behind the report's original tables; `summarize.py` continues to summarize those historical files.

Survival Lab automatically discovers these files. No viewer or catalog files are owned or changed by this strategy task. Check [the live viewer](http://127.0.0.1:9053/) and its recording picker.

## Historical iterations

`pilot.json`, `iterate.json`, `orbit-screen.json` and `relay-screen.json` preserve 44 exploratory cases. They use the earlier diagnostic runner, not the final native-event instrumentation. `controller_before_relay.py`, `controller_stage2.py`, `controller_stage3.py` and `run_exploratory.py` preserve the relevant implementation stages. The three screen scripts explicitly select the historical controllers. Their traces and failed hypotheses are not pooled into final comparison statistics.

`heldout-invalid-single-baseline.json` is excluded: its `orbit_single` label incorrectly dispatched the old MPC pursuit controller. The corrected `heldout.json` uses the orbit controller for both single and relay conditions. The early `relay-150-trees.json.gz` and `relay-500-trees.json.gz` files also remain as historical replays; use the four files listed in the main report for the final iteration.

## Limits of the policy interface

The policy receives cached DTOs and time only. Its initial bait IDs and pair membership are assigned experiment roles. It derives a coordinate frame from visible agent IDs, relative bearings and relative headings, then uses odometry. The fixture's true coordinates, map, predator energy and resting flag are never policy inputs. This is still arranged acquisition on clear forest, not a generated-map navigation or localization solution. Observationless initial partners, collisions, terrain transitions and nearby identityless predators remain limitations.

## Replay coverage audit

The replay requirement in `survival/AGENTS.md` applies to every completed simulation, including failed policies, controls, sweeps and the short verification fixtures. `replay_support.py` provides the shared always-on recorder and immutable publication. `verify.py` now records both of its simulations too.

The audit covered **151 saved experiment rows**. Four already had original replay links; **147 metrics-only cases were newly reproduced**. They are explicitly labelled `NEW REPRODUCTION`, with original file hash, case index, configuration, selected controller version and limitations. This does not reconstruct missing original frames. The 60 cases without an exact historical controller hash use the closest preserved stage and disclose that limitation. Two excluded-baseline reproductions had different retention; the original results remain untouched.

There are **156 strategy replay files** in total: the six preserved old recordings, 147 experiment reproductions, two recorded verification reproductions, and one native-rendered takeover reproduction. The live catalog confirmed all 156 at 2026-09-17T13:49:02.663397+00:00. The detailed proof is [replay-audit.json](../../results/sprinter_relay/replay-audit.json).

Resume or audit without duplicating completed backfills:

```sh
survival/.venv/bin/python survival/research/sprinter_relay/backfill.py --group all
survival/.venv/bin/python survival/research/sprinter_relay/audit_replays.py
```

`backfill-receipts/` maps every recreated case to its new replay and measured results. `versions/*.py.txt` preserves the pre-recording harness source for historical hash inspection. The original missing footage, overwritten intermediate runs and unlogged executions cannot be recovered from aggregate metrics.

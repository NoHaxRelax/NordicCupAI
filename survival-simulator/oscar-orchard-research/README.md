# Survival Simulator: no-predator population research (orchard policy)

**Read [HANDOFF.md](HANDOFF.md)** for the question, the measured results, the policy design, the version history, the late-game failure mode and the recommended next steps.

This folder is a self-contained snapshot of Oscar's local research tree, taken 18 September 2026 around 16:15 Europe/Copenhagen while a 72-run pod batch and six local runs were still in progress (see HANDOFF section 7). Later results are not included automatically.

```
survival/research/orchard/orchard.py       the policy (OrchardPolicy), observation-only
survival/research/orchard/harness_np.py    runs it with predators disabled, JSON result per run
survival/research/orchard/sweep.py         configs x seeds in a process pool, summary table
survival/research/orchard/analyze.py       per-seed tables and trajectories of a results directory
survival/research/orchard/debug_*.py       energy ledger, harvest, meal reasons, pose diagnostics
survival/research/orchard/build_cpu_bundle.py  allowlisted tarball for the Runpod CPU pod
survival/research/society/harness.py       shared harness (read-only diagnostics, recorder hook)
survival/research/simple_policies.py       nursery/greedy/speed baselines
survival/vendor/survival-simulator/        pinned upstream engine, unchanged (see survival/SOURCE.md)
survival/debugger/                         replay recorder, catalog and viewer source (no recordings)
survival/results/orchard/                  every run and sweep referenced in the handoff
```

Setup from the repository root (Python 3.11 or 3.12):

```sh
cd survival-simulator/oscar-orchard-research
python3 -m venv survival/.venv
survival/.venv/bin/python -m pip install -r survival/vendor/survival-simulator/requirements.txt
survival/.venv/bin/python survival/research/orchard/harness_np.py --seeds 1 --horizon 300 --label smoke --out /tmp/orchard-smoke
```

The last command should print one JSON line with `survival_seconds` 300 and about 20 agents alive. Paths in the result JSONs are provenance from Oscar's laptop and the pod; their trailing `survival/...` parts map into this folder.

No competition endpoint was called and no score was submitted for this work.

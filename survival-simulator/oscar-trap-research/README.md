# Survival Simulator research handoff

**Latest Lucas trap session:** [progress, limitations and resumption instructions](LUCAS_TRAP_SLOPSESH1_HANDOVER.md). Includes depth-5 / 33-predator arranged tests and the unfinished real-map delivery experiments.

**Read [HANDOFF.md](HANDOFF.md)** for the detailed wall-bait and narrow-gap findings, interactions, failed approaches, conclusions and recommended next steps.

This folder contains the saved Survival research code, Markdown reports, numerical result summaries, archived controller versions, debugger source and unchanged pinned simulator. **Replay files, large replay chunks and the offline viewer with embedded recordings are excluded.** Historical replay links in copied reports refer to recordings retained locally by Oscar.

Snapshot captured 17 September 2026, 15:56–16:00 Europe/Copenhagen. Other experiments were still running; later results are not automatically included. Completed findings and provisional scout/bodyguard/stacking results are distinguished in the handoff.

From the repository root:

```sh
cd survival-simulator/oscar-trap-research
python3 -m venv survival/.venv
survival/.venv/bin/python -m pip install -r survival/vendor/survival-simulator/requirements.txt
python3 verify_findings.py
```

Use Python 3.11 or 3.12. On Windows, use `py -3` and `survival\.venv\Scripts\python.exe`. Commands in copied reports assume this handoff folder as their project root. Absolute paths in historical JSON are provenance; their trailing `survival/...` paths map into this folder.

- [Wall deployment](docs/survival-wall-deployment.md): observation-only maintenance, food, paid births and native-spawning controls.
- [Wall acquisition](survival/research/wall_acquisition.md): routing failures, sacrifice, resting windows and one-generation replacement.
- [Orchard renewal](survival/research/wall_orchard.md): privileged multi-generation feasibility.
- [Narrow-gap refuges](docs/survival-gap-refuges.md): gap geometry, native-map holds, multiple predators and withdrawal controls.
- [Snapshot manifest](SNAPSHOT.json): included source files and original SHA-256 hashes.
- [Validation](VALIDATION.md): checks performed before push.

The debugger source remains available for recording and inspecting future local runs. Existing replay files are not supplied. No competition endpoint was called or score submitted for this handoff.

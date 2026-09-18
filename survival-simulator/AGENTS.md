# Agent Guidelines — survival-simulator

## Folder layout

- `models/` — Code for agent actions: the decision-making logic that determines how the agent behaves in the simulation (policies, movement/evasion logic, etc.).
- `docs/` — Markdown notes about the task (game mechanics, API payload findings, etc.). Put new documentation here; `README.md` and `AGENTS.md` stay in the task root.
- `scripts/` — Helper and tooling scripts, such as visualization, plotting, debugging or analysis. If you are writing a script to visualize or inspect the simulation, put it here, not in `models/`.

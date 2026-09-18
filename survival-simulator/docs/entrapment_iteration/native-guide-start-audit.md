# Guide energy at assignment

Audit of the downloaded baseline native traces at 20:55 UTC: 113 assignments
across seven games with assignments in the downloaded portion. Some games
were still running; this is a mechanism audit, not a complete-game score batch.

At the assignment tick, 36/113 agents were already below 20% maximum energy
and 78/113 below half maximum energy. Frame world state is recorded after
policy evaluation and before the returned action is applied, so the energy
is the assignment-time value. See the JSON for each seed, timestamp and agent.

The native engine caps movement at walking speed below 20% energy
(`src/elements/environment.py`, `move_entity`). The coordinator currently
prioritizes old status before energy when choosing among observing agents.
This can select a walk-capped old guide even when another observer has energy.
A viability-aware coordinator candidate is being evaluated separately from
the full-start-energy isolated delivery benchmark. No success gain is claimed.

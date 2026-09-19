# Policy modules

- `exploration/` builds the shared map and explores unknown areas.
- `survival/` gathers food and manages the population.
- `entrapment/` finds traps, maintains bait, and guides predators.

`core.py` coordinates all three modules.

These production modules were integrated from
`origin/survival-simulator/lucas-experimental` through `53f1a4c7`.
`core.py` enables the guide's observed-motion association in colony play;
the experimental wrapper preserves current bystander avoidance by default.
Fresh teammate sightings inform bystander avoidance; guides are released after
ten seconds of observed holding. Certified corner pockets are a fallback when
no ordinary gap qualifies and remain valid while occupied. The optional
`corner_pockets` feature additionally considers them alongside ordinary gaps.
See [integration notes](../docs/policy_branch_integration.md).
The latest changes and C++ evaluation backend are documented in
[Lucas/fastsim integration](../docs/lucas_fastsim_integration.md).

`experiment_config.py`, `experimental_policy.py` and `experiment_actor.py`
provide optional, isolated tuning of this current policy. `observation_only.py`
and `observed_bounds.py` provide the no-predator Orchard observation adapter.

## Historical reference only

The following are immutable dependencies of `run.py benchmark`, whose original
source hashes and scores are preserved. They are not used by `trapping`,
`orchard`, `serve`, or `tune`:

- `entrapment_policy.py`, `oscar_orchard.py`, `nikolaj/`.
- `entrapment_sites.py`, `observed_trap_sites.py`, `my_guide.py`,
  `guide_pathfinding.py`, `guide_steering.py`, `predator_following.py`.

Keep production changes in the three module directories and `core.py`.
The matching historical recording script is `scripts/entrapment_game.py`;
the supported current recording script is `scripts/trapping_game.py`.

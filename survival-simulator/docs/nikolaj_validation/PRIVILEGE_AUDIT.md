# Privileged-information audit: Nikolaj's best policies

Reviewed commit `6e49081d845dae6f650b4e3d365f383ea4f31ee8` from
`sim-optimization-no-trapping`, including all three `models/best_policies` configs.

**Finding: no privileged-information use found in the reviewed winning policies.**
No competition validation service was contacted and no official attempt was made.
An initially started local HTTP/game check was stopped after the user narrowed
scope to this privilege audit; no score or deployment-success claim is based on it.

## Evidence

- Python `OrchardEvasionPolicy` consumes public agent self-reports, observed
  objects, and elapsed game time. Its Orchard base imports math, its own RNG,
  dataclasses, and the action DTO. It does not import simulator/world state.
- Shared positions, walls, trees and fruit are tracked from observations and
  dead reckoning. Fixed 1600 x 1200 dimensions and other published mechanics are
  assumptions, not hidden per-map coordinates. Anchoring uses observed wall
  endpoints. The policy does not receive a complete undiscovered map.
- The campaign uses a spawned policy worker and an explicit public-field
  whitelist. The world seed, world objects, actual fruit/tree ages, hidden
  predator IDs and hidden agent traits are excluded. Policy randomness defaults
  to a separate seed 0; it is not initialized from the map seed.
- Synthetic audit: for each of the three configs, two isolated workers received
  matching public observations for 30 steps. One also received injected x/y,
  direction, world seed, hidden maximum age, unseen fruit, object ages/energy,
  hidden IDs and world edge coordinates. Actions matched on every step, and
  both workers reported that simulator modules were absent.
- The compiled native policy's include/identifier/public-surface/object-symbol
  checks passed. Manual inspection of `fill_states` and the observation builder
  confirmed that it passes public self-report fields and sensed observations.
  Fruit/predator hidden IDs are not emitted; edge coordinates are relative
  observed endpoints. The native policy's object file has no Python C-API
  references. Shared RNG/math helpers contain algorithms, not engine RNG state.

These are source and accidental-leak checks, not a security proof against
malicious code sharing a process. They support the reviewed policy being
observation-only; they do not certify every future change.

## Separate limitations

The reported 2038.7 mean is over four reused training maps with natural
predators. It is not an independent generalization estimate, and the leading
config's recorded score predates later evasion fixes.

The pushed `agent_server.py` still instantiates `OptimizationPolicy`, so simply
starting that endpoint does not serve the reported Orchard winner. Selecting
`OrchardEvasionPolicy` with `rank1_pod-03.json` is still required before deployment.
Endpoint behavior was left as pushed for this audit.

Reproduce only the synthetic audit (no game):

    python scripts/audit_best_policy_inputs.py

Native source/object checks, after `python fastsim/build_policy.py`:

    python fastsim/check_boundary.py

Evidence: `input-audit.json`, `native-boundary.txt`.

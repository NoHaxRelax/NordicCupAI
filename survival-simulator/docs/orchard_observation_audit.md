# Orchard: observation-only evaluation

The original Orchard controller does not read simulator objects or hidden
runtime fruit/tree ages, agent lifespans, absolute poses, or the biome raster.
However, it assumes a 1600 by 1200 arena and the original evaluation harness
passes its world seed to the policy's random generator. The stricter runner
removes both inputs without editing the shared Orchard policy or native engine.

## Boundary enforced by the runner

- The policy runs in a separate process created with `spawn`.
- Messages contain JSON copies of official `ObservationResponse` fields and
  public simulation time. Extra top-level and nested fields are discarded.
- Fruit and tree observations contain only type, relative distance, and angle.
  Agent observations additionally contain public ID and relative direction;
  edges contain relative endpoint coordinates.
- The worker checks that neither `src.core` nor any `src.elements` module is
  imported, before and after policy decisions. No environment object is sent.
- The policy RNG seed defaults to 0 and is independent of the world seed.
- Map dimensions and wall inset are inferred from compatible perpendicular
  boundary segments in a single agent's observation. Before that evidence is
  available, maps remain relative and assumed absolute bounds are disabled.
- Physics constants, ripening timing, and the declining-tree-count curve remain
  model assumptions. The policy never receives actual global tree counts or
  true fruit/tree birth times to update those assumptions.

This prevents accidental information leakage; it is not an OS security sandbox
against malicious Python code. Source review and input isolation are used
together. Simulator truth is permitted in evaluation output, never as policy
input. Predators are disabled in the evaluator, matching the research scenario.

## Validation

Ten focused tests cover input stripping, official schema matching, real spawned
workers, identical actions with injected hidden fields, and boundary inference
from rotated edges and arenas with different dimensions. Insufficient or
inconsistent geometric evidence leaves the map unanchored.

The boundary adapter also preserves the active pose object during anchoring:
the inherited observation handler holds a reference to it while processing the
rest of the same observation. This prevents stale-frame landmark coordinates.

## Run locally

From the repository root in PowerShell:

```powershell
.\survival-simulator\run.cmd orchard --seed 1 --seconds 3000 --output logs/orchard
```

The runner prints progress every 100 simulated seconds and writes a final JSON
with score, survival duration, policy/runner source hashes, input-field audit,
and observed-boundary evidence. It refuses to overwrite a completed result.
Each seed must use its own process; use a new output directory to rerun a seed.

The historical Orchard scores are not exact expectations for this adapter:
policy randomization is independent of the world seed, anchoring waits for
observed evidence, and platform/set iteration can change trajectories.

## Maintained source locations

The current shared policy is `models/survival/oscar_orchard.py`. It is byte-identical
to the Orchard source imported before the modular branch integration. Input isolation and boundary
inference are in `models/observation_only.py` and `models/observed_bounds.py`;
the evaluator is `scripts/run_orchard.py`. The worker also clears inherited
command-line arguments before loading the policy. The retired vendored engine
was identical to the maintained `src/` engine, which is now used directly.

The module-path integration has not been executed or tested in this update;
the validation above describes the earlier observation-adapter checks.

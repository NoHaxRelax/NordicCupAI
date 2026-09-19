# Results are not comparable across CPU generations (2026-09-19)

A byte-identical binary, given a byte-identical configuration and the same seeds,
produces **different games** on Zen2/Zen3 than on Zen4/Zen5. Any comparison that
places a candidate on one machine and its control on another measures the CPU.

## The measurement

Default `notrap` configuration, seeds 20000-20002, horizon 3000, predators on,
run through `fastsim.fastpolicy.PolicySimulationCore` on all 24 campaign pods:

| scores | pods | CPUs |
|---|---|---|
| `(1153.22, 1792.97, 2629.93)` | 15 | EPYC 9654, 9B45, 9655, 9655P, 9965 (Zen4/Zen5) |
| `(475.92, 1478.96, 1751.80)` | 9 | EPYC 7702P, 7713 (Zen2/Zen3) |

The split is exactly by CPU generation. It is **not** a build difference: pods in
both groups reported the same `_policy*.so` SHA-256 (`71475d379c30`), the same
`models/notrap_config.py` hash, and the same hash of the flattened policy kwargs.
Pods carrying a *different* binary (`bbb45b668182`, built from a later bundle)
still agree with their own CPU group, which further rules the binary out.

The same 500-seed arm run on two pods in the same group is bit-identical
(`dodge1.05` on notrap-cpp-13 and notrap-cpp-22: 0 of 500 seeds differ). So the
simulation is perfectly deterministic *within* a CPU generation.

## Why

glibc selects implementations of `cos`/`sin` (and other libm entry points) at
load time through ifunc dispatch on CPU feature bits, so the same machine code
calls a different routine on AVX-512 hardware than on AVX2. The results differ in
the last ulp. The simulation is chaotic - agents and predators re-plan every tick
from trigonometric quantities - so a 1-ulp difference at t=0 grows into a
different game well before the 30000-tick horizon. A ~677-point score gap on seed
20000 is the amplified form of a last-digit disagreement, not a fault.

## What it invalidated

`sweep_fleet.py` assigns one configuration per pod, so before this was found an
arm and its baseline could land on opposite sides of the split:

- `deflect1` - baseline and every arm on Zen4/Zen5. **Valid.**
- `econ1` - baseline on Zen4; one arm (`breed250`, notrap-cpp-01) on Zen2. That
  arm alone is contaminated.
- `flee1` - baseline on notrap-cpp-17 (**Zen3**), arms spread across both groups.
  **Invalid, discarded and re-run as `flee2`.**

The two-baseline anomaly in the campaign-1 study data (14 pods scoring the default
at 1545.1, 9 at 1651.0) is the same effect. It is not an unexplained defect.

## What to do about it

`sweep_fleet.py` now restricts every sweep to `CONSISTENT_PODS` - the 15
Zen4/Zen5 machines - unless `--any-cpu` is passed. That is the fix: compare
candidates only against controls measured on the same silicon.

Do **not** try to fix this by pinning a `-march` or disabling ifunc. The engine's
determinism guarantee is per-machine and that is sufficient; what matters is that
a comparison never straddles the boundary. Note also that a "held-out validation"
run is only meaningful against a baseline measured on the same CPU group.

Aggregating scores across groups is legitimate only for statistics that do not
compare configurations - e.g. throughput, or the variance of a single
configuration across maps.

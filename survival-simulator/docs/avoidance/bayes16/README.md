# Sixteen avoidance ideas with Bayesian tuning

Frozen experiment on `survival-simulator/orchard-corner-avoidance`.
Oscar's Orchard settings stay fixed; only avoidance-related settings differ.
The engine and policy execute in C++. No trap, oracle, frozen predator, infinite
energy, or undiscovered map information is enabled.

Run: `python scripts/optimize_avoidance16.py --out logs/avoidance/bayes16 --workers 32`

## Protocol fixed before results

- 16 families: sideways escape, radial escape, no gaze tracking, committed heading,
  wall deflection, group sightings, closest-agent targeting, remembered danger,
  predator-aware births, sprint energy reserve, nearest-corner steering,
  sparse-corner steering, cone trigger, three-step pursuit approximation,
  energy-priced escape, orchard-direction escape.
- Each family receives 16 trials on the same eight training seeds (1001–1008).
- First six trials initialize the model; next ten use a Gaussian process with a
  Matern-5/2 kernel and expected improvement. Each family has its own model.
- Select each family's best training configuration, then compare all sixteen and
  the unchanged baseline on 32 different maps (2001–2032).
- Confirm the two validation leaders and baseline on 64 further maps (3001–3064).
- 2,784 complete games in total, horizon 3,000 simulated seconds or extinction.
- Score is the objective. Angle counters do not determine success or selection.
- Both corner variants target ±10 degrees; that is not a guaranteed result.
- Small training sets and sixteen trials make this a bounded search, not proof
  that any family has reached its best possible settings.

The extra local escape search is deliberately approximate: hidden predator
terrain, energy and current target are unavailable. It predicts a conservative
15-unit pursuit, checks recent observed walls, and compares 24 directions at
walking/sprinting speed. Its three-step version repeats the candidate direction;
it is not an exhaustive search of all three-step action sequences. The two
one-step versions price energy or favor the current Orchard movement direction.
Existing policy movement and all engine rules remain unchanged when disabled.

Raw game rows, exact configurations, trial histories, source hashes, software
versions, and completion time are retained. Paired comparisons use matching map
seeds; selection and confirmation results are reported separately.

Runpod: one dedicated 32-vCPU `cpu3c` worker, $0.96/hour plus 10 GB container disk.
Previous research estimate was $5.2511 of a $10 cap, leaving $4.7489. Stop and
remove this experiment's worker after artifacts are downloaded. Other team pods
are outside this experiment and must not be modified.

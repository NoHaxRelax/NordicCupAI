# Crowded trap iteration

Branch: `survival-simulator/lucas-experimental`. Total authorized Runpod budget:
$10 for this research session; spent so far $0 (local experiments only).

## First paired probe

Native 30-preloaded + one newcomer benchmark; 12 maps/encounters per strategy.
All strategies use identical seeds. Full-energy fixture bait, normal-energy
full-start guide, native predator sensing/movement; all other criteria unchanged
from `docs/guide_multi_1000_results.md`. These are development cases, not a
reliability estimate. All ticks remain in `logs/entrapment-iteration/`.

| Strategy | Passed / 12 |
| --- | ---: |
| Existing 55-unit stop, nearest-predator selection | 9 |
| 40-unit stop, nearest-predator selection | 4 |
| Existing stop, observed-motion target association | 10 |
| 40-unit stop, observed-motion target association | 7 |

Do not promote the 40-unit stop: geometric hearing-range arithmetic alone
misses crowd interference. The tracking candidate retains the 55-unit stop.
Its target association uses ordinary observations, no engine IDs or hidden
positions. It can still confuse overlapping predators and loses association
after long occlusions. A paired 100-case evaluation is in progress.

An ordinary inside corner does not exclude a predator's contact radius:
native corner probe caught bait in 16/16 approaches, versus 0/16 for the
15-unit channel control over 30 seconds. This does not reject all compound
corner pockets. Candidate corners still need exclusion and replacement access.

## Reproduction and source control

`trap_corner_probe.py --output NEW_DIRECTORY` saves all native probe ticks.
`guide_batch.py --multi --maps N --workers W --output NEW_DIRECTORY` now archives
its Python/config/viewer source and executes that frozen source. Reusing output
with different source, seeds, or simulation settings is rejected. This avoids
mixing iterations while editing policies during a batch.

The benchmark source paths were repaired after the folder split (`f212fde`).
Gap limits are 10.1–19.9 (`b027919`). The original 1,000-case evaluation remains
unchanged historical evidence, not a measurement of this candidate.

## Remaining work

- Complete the larger paired comparison; inspect retention losses separately
  from delivery failures, with actual replacement-agent access.
- Evaluate compound corner pockets, and rank crowded arrival space as well as
  gap geometry. Plain corners are not automatically safe.
- Test bystander avoidance and live bait handoffs in native games. Avoidance is
  a coordinator wrapper; Oscar's survival module is unchanged.
- No measured 95% delivery reliability is claimed. The earlier 96.5% number is
  map-site availability, not delivery success.
- Account usage remaining is not exposed by the available tools; the 20% stop
  cannot be automatically measured in this session.

# Iteration record

17 September 2026. Local strategy work only; exact simulator commit is recorded
in every result. This folder owns its controller and runner; it does not alter
other research, the vendor, or the debugger.

## Version 1

Implemented a real four-agent orchard colony, visible-fruit allocation, local
tree-density homes, replacement births, energy-based guide selection, outward
leading, observed-stillness release, a temporary worker exclusion zone, and a
return detour. Founders have 150 energy. Controlled orchards supply finite
initial fruit and normal tree replenishment, not energy injections. Additional
predators spawn naturally, plus two recorded encounter schedules in fixtures.

Local coordinates come from odometry, static-object corrections and identified
agent observations. Generated-map policies receive no environment handle.
The separate oracle mode supplies only true agent poses, never predator energy
or resting status.

The smoke test produced a successful remote release in the observation-only
arm, but its predator returned within 250 units of the release-time worker
centroid 8.9 seconds later. Neither colony lost a worker by 100 seconds. This
does not establish a score advantage or sustained exclusion.

The oracle replay exposed a specific failure: a guide reached the top-left
physical boundary while looking backward. It continued to ask for outward
movement and was caught in the corner. More precise localization alone cannot
fix an unsafe route. Version 2 will add forward scans, remembered edges,
boundary-aware outward steering and tighter turn limits near a predator.

Artifacts: `policy_v1.py`, `run_v1.py`, `iteration1-smoke.json`, its three replay
files, and `iteration1-train.json` under the owned results directory. The runner
snapshots retain each algorithm. Historical runners import their matching
versioned policy; this explicit import repair changes their runner hashes from
the original results, but the policy hashes remain the recorded versions.

## Version 2

Added forward scans, remembered observed edges, static-feature consensus rather
than a median over conflicting matches, relative position correction when
identified agents meet, and tighter steering close to a predator. Added paired
oracle-control mode and actual displacement diagnostics. The original corner
failure was avoided in the oracle smoke replay, but attention measurement showed
many supposed guides had not acquired a pursuer. Reduced worker proximity alone
was insufficient: guide time removes individuals from the worker denominator.

## Version 3

Added cautious acquisition, inference of a pursuit from relative predator
heading/motion, remote-disengagement events, and reset of old release/return
waypoints between guide jobs. A failure replay shows two completed release and
return cycles followed by a new encounter in a corner and eventual losses.
Training obstacle fixtures improved in 3/3 pairs (mean score +2.21); forest and
river fixtures remained worse. Shortening displacement to 350 and releasing
without waiting for rest was poor on two of three forest seeds; it was rejected.
These are tuning results, not held-out evidence.

## Version 4, frozen for held-out cases

Reject a guide older than 70, with two visible predators, more than 130 from
its candidate predator, less than 250 energy above the sprint cutoff, within
100 of a recently observed edge, or with a known edge near its first escape
segment. Stop an unproductive acquisition attempt after four seconds. Return
routing, ordinary colony behavior and source physics are unchanged from v3.

The freeze file records hashes, parameters and held-out seeds before their
execution. This is deliberately conservative and does not claim global tuning.
Unseen geometry, imperfect landmark matching, return-route hazards and the
single-active-guide limit remain. No controller reads predator sleep or energy.

## Version 5, observation ordering correction and fresh evaluation

Version 4 generated-map seed 103 produced a 539.9-second control and a
263.9-second banishment run even though neither selected a guide. This falsified
the assumption that the measured difference could be assigned to guide actions.
The mapper broke equal tree-richness ties using discovery order, and observations
arrive through upstream object sets. Tiny geometric differences could also alter
map choices. This is a policy/evaluation defect, not a new engine exploit.

Version 5 sorts observations canonically and rounds incoming numeric DTO values
and outgoing actions to six decimal places. Original engine movement, sensing,
energy and tick timing remain unchanged. Added an order-invariance check using
shuffled actual cached observations. A three-seed generated-map null comparison
checks arms with no selected guide before moving to fresh held-out seeds 201–208
and fixture seeds 21–24. Version 4 held-out results are retained but excluded
from the final strategy comparison. No banishment parameters were changed in v5.

The corrected null check matched all saved game metrics, events and traces on
all three seeds. Final held-out results use only v5. The gate dispatched guides
on only one of eight ordinary generated seeds. A training-only ablation with
the risk gate off caused five guide jobs and one planned release on seed 1,
gaining 2.18 score at a cost of two guides and about 777 movement/turning energy;
seeds 2 and 3 remained inactive ties. This was recorded as an acquisition-limit
diagnostic, not used to retune the frozen final controller.

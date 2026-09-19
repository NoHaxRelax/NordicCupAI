# Observation-only guide energy budget experiment

This isolated experiment starts from `upstream-orchard/source`. It changes no
survival module or simulator mechanic. Two native seeds were run to extinction
for both the frozen upstream coordinator and the candidate, with at most two
local workers.

## Candidate heuristic

For every ordinary observer eligible to guide, the coordinator asks its
observation-derived navigator for a collision-valid route from the estimated
agent pose to the static handoff point. Blocked or non-finite routes are
rejected. For a route of length `d`, it budgets:

- `1.25 * d` for route detours;
- conversion to commanded movement using the agent's current public biome
  modifier;
- native full-sprint energy cost from public walk and sprint speeds;
- 0.5 energy plus 0.5 per 100 route units for turning;
- a post-travel reserve of 20% of that agent's maximum energy plus five.

Only positive-surplus observers are viable. Old age is preferred only within
that set, followed by surplus and current energy. If nobody is viable, no role
is assigned: all observers continue through normal bystander avoidance and the
track retries on later ticks. Assignment events record route, estimated cost,
reserve, and surplus.

The same candidate also defers bait assignment for an agent on a tick where
Oscar explicitly requests reproduction, preventing the bait role from erasing
the planned birth.

Frozen `models/core.py` hashes:

- upstream baseline: `3043d980dbe9378fae2a225be2e30a236d718c9c76f258d7da6469003412c07d`
- candidate: `85e48cfc7cbf2e56485ed04c6a93727d40f6e0c8114eb386825c0869670b0adc`

## Native results

| seed / policy | extinction | score | born | guides / deaths / arrivals | guide starts walk-capped | starts below half energy | held 30s | bait gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 118686522 baseline | 507.8s | 531.5 | 141 | 22 / 19 / 0 | 8 | 15 | 1 | 0.1s |
| 118686522 candidate | 1022.3s | 994.5 | 245 | 2 / 1 / 0 | 0 | 1 | 1 | 83.5s |
| 1883894846 baseline | 530.8s | 554.2 | 102 | 11 / 10 / 4 | 9 | 11 | 4 | 37.3s |
| 1883894846 candidate | 531.0s | 552.1 | 105 | 3 / 3 / 1 | 0 | 3 | 2 | 1.0s |

The mechanism check succeeded: none of five candidate guide starts was already
below the native 20% walk cap, versus 17 of 33 baseline starts. The selected
candidate starts all had positive recorded budget surplus.

The policy outcome is too restrictive. On seed 118686522 it recorded 6,070
track-ticks with no viable observer and made only two assignments while
predators accumulated to ten. On seed 1883894846 it made three assignments and
held two predators for 30 seconds versus four under baseline. Across both seeds
the candidate made 5 assignments versus 33 and produced one delivery arrival
versus four. The longer survival of the first candidate run came with an
83.5-second bait gap and substantially less entrapment activity, so it is not
evidence of a better whole-game policy.

Separate processes diverge because the reproduction-preservation guard changes
population events, so score and lifetime are integrated-policy observations,
not deterministic attribution to guide filtering. The start-energy and
assignment-count differences directly exercise the intended mechanism.

## Decision

Do not promote this hard viability gate. It fixes the documented low-energy
selection error but turns uncertainty in route and terrain cost into prolonged
inaction. A follow-up should retain the energy budget as a ranking signal and
add a bounded response to persistent predator pressure, rather than treating
the conservative full-sprint estimate as an indefinite eligibility gate. The
deferred-bait reproduction guard remains independently compatible with the
tuned orchard.

Exact summaries and per-assignment energies are in `results.json`; full native
traces and source snapshots are retained beside this report.

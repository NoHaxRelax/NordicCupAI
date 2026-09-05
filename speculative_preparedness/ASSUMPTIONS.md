# Assumptions

Every claim this package rests on, with the cost of being wrong and a probe
that falsifies it. Probes are ordered so the cheap, high-impact ones run
first. Budget: all of A1-A5 inside the first fifteen minutes of the
competition.

Confidence is our own honest estimate, not evidence.

## A1 -- The payload shape is real

**Claim:** the observed JSON reflects the actual `/predict` request body.
**Evidence:** one screenshot of a stubbed endpoint returning a frozen
example. Values (`score: 123.4`, `age: 5.2`) never advance, so this is a
documentation placeholder, not a capture.
**Confidence:** medium on shape, zero on values.
**Cost if wrong:** low. `schema.GameState.parse` is tolerant and
`schema.schema_drift` reports the difference on frame one.
**Probe:** log `schema_drift(payload)` on the first live request. Any output
is a finding.

## A2 -- The response format

**Claim:** unknown. Candidates implemented in `actions.py`: per-agent
continuous records, a flat verb list (the 2025 race-car shape), absolute
headings.
**Evidence:** none. The endpoint accepts anything and no-ops it, so it cannot
be probed before the competition opens.
**Confidence:** none.
**Cost if wrong:** total -- a wrong format means every action no-ops and we
score the floor.
**Probe:** the moment the real endpoint responds, send one deliberate
extreme action (`throttle=1.0, turn=0`) for several frames and check whether
any observation bearing shifts. If nothing moves, cycle codecs. This is the
single most valuable minute of the competition; do it first.

## A3 -- Angle convention

**Claim:** `angle` is radians, agent-relative, counter-clockwise-positive,
with 0 straight ahead.
**Evidence:** the sample values are +/-1.57 and -2.57, consistent with
radians in (-pi, pi]. Sign convention is a coin flip.
**Confidence:** high on radians, 50% on sign.
**Cost if wrong:** high and silent -- a sign error makes the policy flee
*toward* predators, and it will look like a bad policy rather than a bug.
**Probe:** after A2 works, turn one direction repeatedly and watch whether a
tracked bearing increases or decreases. `mock.RuleDraw.ccw_positive`
randomises this so the failure is testable offline.

## A4 -- `vision_angle` is a half-cone

**Claim:** `vision_angle: 1.57` means +/-90 degrees, not 90 degrees total.
**Evidence:** none. Both readings are common.
**Confidence:** 50%.
**Cost if wrong:** moderate -- affects only how much unobserved space the
belief layer assumes behind us.
**Probe:** rotate in place and record the maximum absolute bearing ever
observed. It converges to the true half-angle.

## A5 -- `edge.coords` frame

**Claim:** the two points are a line segment. Whether they are world-absolute
or agent-relative is unknown; `[[50,50],[100,100]]` fits either.
**Evidence:** none. If absolute, the agent must know its own world position,
which is not in the payload -- mild evidence for relative.
**Confidence:** 50%.
**Cost if wrong:** moderate. Edge avoidance in `policies._avoidance` treats
the coords as relative; absolute coords would make it steer toward a fixed
corner.
**Probe:** move without turning and check whether the coords change. Absolute
coords stay put; relative coords translate.

## A6 -- Sprinting costs more energy than moving

**Claim:** `sprint_speed > speed` implies a cost, and energy is the survival
resource.
**Evidence:** structural only -- a free speed boost would make `speed`
pointless, and `max_energy: 500` implies a budget worth managing.
**Confidence:** high, but the *magnitude* is unknown and the policy's energy
floors are pure guesses.
**Cost if wrong:** moderate. If sprinting is free, `PolicyConfig
.sprint_energy_floor = 0.0` and we sprint always.
**Probe:** sprint for twenty frames, then idle for twenty, and difference the
energy series. Gives cost per second for both directly.

## A7 -- Actions are executed as commanded

**Claim:** the commanded turn and throttle are applied, so they can serve as
an ego-motion prior in `belief._predict`.
**Evidence:** none. There may be inertia, turn-rate limits, or terrain
effects.
**Confidence:** low.
**Cost if wrong:** moderate. Track dead-reckoning drifts, but the association
step in `belief._associate` corrects it whenever an object is re-observed, so
the failure degrades rather than breaks.
**Probe:** compare predicted against observed bearings for a re-observed
static tree; systematic bias means the prior is wrong.

## A8 -- Score rewards survival

**Claim:** `score` grows with `age`, summed over living agents.
**Evidence:** none beyond the two fields coexisting.
**Confidence:** medium.
**Cost if wrong:** high -- if score rewards foraging, territory, or
reproduction, a pure survival policy optimises the wrong thing entirely.
**Probe:** log `score` against `age` for one episode of the no-op policy. A
straight line means survival; anything else means we are missing a mechanic,
and the observation vocabulary in A9 is where to look for it.

## A9 -- The observation vocabulary is incomplete

**Claim:** `tree`, `predator`, `edge` is not the full set. An energy budget
with no observable food source is not a playable game.
**Evidence:** structural. `energy` decreases with nothing in the schema to
restore it.
**Confidence:** high.
**Cost if wrong:** none -- being wrong here means the game is simpler than
expected.
**Probe:** `schema_drift` accumulates every unseen `type` automatically.
Watch it for the first minutes; a `food` or `water` type appearing changes
the whole objective.

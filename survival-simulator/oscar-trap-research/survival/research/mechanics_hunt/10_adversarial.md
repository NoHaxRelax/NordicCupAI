# Investigator 10: adversarial verification and combinations

Tested 17 September 2026 against the unmodified vendored simulator at commit
`acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. Run:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/10_adversarial.py --trials 400
```

Machine-readable evidence is in
[`10_adversarial.json`](../../results/mechanics_hunt/10_adversarial.json). The
script contains assertions for every headline claim and negative control.

## Verdict

The most useful cross-area result is a contract-compliant emergency reproduction
interaction. Actions and births happen before passive death and predation. Passive
death has no score penalty, while predation subtracts the victim's remaining energy
divided by 100. A threatened parent just above 100 energy can therefore reproduce,
then die from passive drain before the predator phase, leaving a child that may keep
the species alive without a penalty for the parent.

This is a fallback, not a default. In the arranged geometry, walking away was safer
than stationary reproduction. On generated maps, slow terrain made the same apparent
threat less urgent, and reproduction could turn a surviving parent into a dead parent
and eaten child. A policy should reproduce only when it predicts that ordinary escape
will fail, not merely because a predator is close.

## 1. Emergency reproduction before death or predation

The synthetic control places a realistic founder at `(100, 200)` and an active
predator 29 units away. Each controller returns exactly one `ActionRequest` for the
only observed parent. Parent energy and starting geometry are arranged; unrelated
random tree and predator spawning are disabled. Birth, passive drain, predator
movement, collision, sensing and scoring all use the original engine methods.

| Starting energy and action | Species survives tick | Child survives | Mean score effect excluding +0.1 survival time |
| --- | ---: | ---: | ---: |
| 100.00, request birth | 0.0% | 0.0% | −0.999 |
| 100.05, idle | 0.0% | 0.0% | −0.9995 |
| 100.05, stationary birth | 62.5% | 62.5% | −0.2813 |
| 101.00, idle | 0.0% | 0.0% | −1.0090 |
| 101.00, stationary birth | 88.25% | 62.5% | −0.2871 |
| 150.00, idle | 0.0% | 0.0% | −1.4990 |
| 150.00, stationary birth | 88.25% | 62.5% | −0.6252 |

These are 400 deterministic spawn-position seeds per row. The strict energy boundary
is confirmed: energy must be greater than 100 at the birth check. At exactly 100,
the request produces no child.

At 100.05 energy, birth subtracts 100, passive drain kills the parent before the
predator phase, and removing the parent while iterating skips the newborn's first
agent update. A surviving child is still returned, but its first observation list is
empty. The predator is not fooled or disabled: it can sense and eat the child in the
same tick, which happened in 37.5% of these seeds.

### Escape is the preferred control

In this particular open-field geometry, an ordinary 10-unit retreat preserved the
parent in 100% of trials with no predation penalty. At 101 or 150 energy, combining
retreat and birth preserved the parent in every trial and a child in 72.25%, but the
predator sometimes ate the child, creating a mean −0.2078 score effect. Birth is
therefore actively harmful when the parent can simply escape.

### Generated-map spot checks

Seeds 1, 2, 3 and 42 used the upstream `SimulationCore` map generator, terrain and
obstacles. To isolate the ordering interaction, other initial agents were removed,
parent energy was set to 100.05, and one active predator was arranged 29 units away
at the first free tested bearing. These remain fixture-assisted checks, not policy
benchmarks.

- Seeds 1 and 2: idle was fatal with a −0.9995 penalty; birth left one child and no
  predation penalty.
- Seed 3: the parent survived idle in swamp terrain, while birth lost both parent and
  child and incurred −0.75. The slower predator made reproduction unnecessary.
- Seed 42: idle preserved the parent; birth replaced it with one surviving child and
  had no score penalty. It did not improve the one-tick population outcome.

The tactic needs no hidden predator energy or engine handle. Parent energy, biome and
predator observation are ordinary fields. Its reliable use does require a conservative
prediction of next-tick contact from a cached, one-step-old predator observation.

**Practical priority: medium.** It can preserve a last lineage and save roughly one
score point in a narrow emergency, but a false positive sacrifices the parent and may
feed a 75-energy newborn to the predator. Test it inside a real observation-only
controller before treating it as a score improvement.

## 2. Returned observations can contain entities already gone

Two one-action probes confirm that the response is not an atomic snapshot of the
post-step world:

- A fruit overlapping an agent appears in that agent's returned observation even
  though the same agent has already consumed it. It disappears from the following
  response.
- A surviving observer reports another agent's ID even though that other agent is
  killed in the predator phase and is absent from the returned agent-status list.

This is contract-compliant and visible through ordinary observations. The practical
magnitude is one tick. Controllers should treat the live agent-status list as
authoritative for colony membership and should not commit to a fruit for an extra
tick merely because it was present in the last observation. Fruit lacks an ID, so
location/history matching needs a short expiry.

## 3. Collision and duplicate-action combination was falsified

The known synthetic control reproduces a single 40-unit sprint from `x=195` to
`x=235` across a 30-wide wall. This requires the assigned evolved traits
`speed=20, sprint_speed=40` and uses one action.

Two negative controls bound the claim:

- Four duplicate 10-unit walks do **not** tunnel. Collision is checked after each
  request, and the avoidance search deflects the agent to `(195, 160)`. These
  duplicates also violate the documented one-action-per-agent rule.
- A founder's single oversized 40-unit request is capped at sprint speed 20 and does
  not tunnel, ending at `(195, 180)` after collision avoidance.

Endpoint tunnelling is therefore not amplified by merely repeating safe-sized moves.
It remains a trait-and-geometry-dependent, fixture-demonstrated mechanic rather than
a founder tactic.

## 4. Malformed numeric input is not a gameplay lead

The local Pydantic DTO accepts `NaN` as a float, but strict JSON serialization rejects
it. Advancing it into the engine risks contaminating energy or coordinates and breaking
the response. This is malformed, non-standard input outside the documented contract,
not a strategy advantage, and it was intentionally not pursued.

## Coverage and rejected hypotheses

| Hypothesis | Result | Boundary |
| --- | --- | --- |
| Last-action reproduction can preserve a lineage | Supported conditionally | Arranged threat; 62.5% child survival at energy 100.05 |
| Birth prevents the predator from acting on the newborn | Rejected | Newborn can be sensed and eaten in the same tick |
| Reproduce whenever a predator is close | Rejected | Retreat won the open fixture; slow terrain produced harmful false positives |
| Returned observations describe the final world state | Rejected | Consumed fruit and dead agent persisted for one response |
| Duplicate short moves combine into wall tunnelling | Rejected | Per-request collision deflected every short move |
| Founders can obtain the 40-unit tunnel with an oversized request | Rejected | Request capped at sprint speed 20 |
| Local `NaN` acceptance is a usable input exploit | Rejected | Non-standard JSON and likely state/response failure |

## Boundaries

- Proven engine ordering does not prove that an observation-only policy reaches the
  useful emergency state often enough to improve a 3000-second score.
- Synthetic trials use arranged geometry, assigned energy, no food and disabled
  unrelated spawns. Generated-map checks still arrange the encounter and energy.
- No hosted validation, evaluation, submission, network target or modified engine was
  used.
- All recommended behavior follows the documented one-action-per-observed-agent rule.
  Duplicate-action and malformed-input controls are reported separately and are not
  recommended.

# Guide fitness and sprint capture audit, 19 September 2026

Premature predator deaths with sprint energy remain failures, including
non-guides. They are not automatically evidence that a last-tick escape was
physically possible: native sprint traits can mutate below predator speed 15,
terrain applies independently to each creature, walls constrain movement, and
the normal predator observation is one movement behind. Deliberate delivery
holds remain separate.

The recorder now saves every sprint-available capture with role, walking and
sprint traits, the last action, and up to three seconds of observed terrain,
energy, nearest seen predator distance and actions. The viewer exposes failure
events for all roles, not only guides. This is evaluator output only; it never
feeds policy inputs. Recorded biome history does not reveal an unobserved
border or prove the biome at the exact capture position.

## Existing terrain handling

The default guide searches three ticks ahead and checks candidate commands
with river slowdown starting at either of the next two ticks. Current biome
and shared visited biome samples provide the nominal terrain estimate; engine
biome polygons are unavailable to the policy. A known unsafe sampled crossing
loses to a sampled-safe candidate. This is not a guarantee: only four nominal
beam candidates reach the robust check, and danger beyond three ticks, unknown
borders and unsampled predator behavior remain limitations. Bystander avoidance
still uses a shorter local forecast and needs improvement separately.

## Opt-in coordination experiment

`--guide-coordination` enables `models/entrapment/guide_coordinator.py`:

- New guides must have sprint energy and a sprint trait above 15. An
  observed-wall route to delivery must leave energy for sprint plus a short
  escape reserve. The estimate charges walking through river along the whole
  route and earliest senescence. This can reject a guide that would succeed
  on easier terrain; future steering and reacquisition costs can exceed it.
- Among fit observers, prefer a predator already moving compatibly toward the
  agent, then older agents, then energy. Bait, reserved bait, existing guides
  and nursery parents remain protected from selection.
- A tired, blind or not-followed current guide can hand over to a nearer fit
  observer after two consecutive positive chase-compatible samples. Initial
  latch-false, resting and ambiguous multiple-predator observations do not
  establish following. The old guide then receives ordinary avoidance.
- Compatibility is evidence, not a hidden target ID. No advance rendezvous
  dispatch is implemented yet. The Sol `plan_relief` helper remains available
  for that work; only its `forecast_travel` component is connected here.

The experiment is disabled by default pending comparison. Both paired native
games use seed 1883894846, maximum 3000 seconds, and every frame is saved:

- `logs/entrapment-iteration/guide-coordination-control-20260919`
- `logs/entrapment-iteration/guide-coordination-v6-20260919`

The latter replay is served at `http://localhost:9080/`. Manifests record exact
source hashes and flags. These runs use local CPU; no new Runpod cost.

## Completed paired result

Both games ended in extinction. Code: `68b3b25`; the control reproduced the
earlier default score exactly. No handover occurred, so this pair does not
validate the handover path in a real game.

| Measurement | Control | Fit-guide experiment |
| --- | ---: | ---: |
| Score | 784.88 | 754.39 |
| Lifetime (seconds) | 750.4 | 719.5 |
| Delivery arrivals / assignments | 2 / 19 | 2 / 5 |
| Premature sprint-available guide captures | 0 | 0 |
| Premature sprint-available captures, all roles | 3 | 3 |
| Predator deaths, all roles | 20 | 27 |
| Estimated seconds without bait after first arrival | 81.7 | 71.7 |
| Runtime (seconds) | 267.45 | 217.20 |
| Saved frames | 7,504 | 7,195 |

The experiment reduced guide assignments but did not increase deliveries,
reduce sprint-available deaths or improve score. It shifted more predator
deaths onto ordinary evading agents. Do not interpret 2/5 vs 2/19 as a proven
capture-rate improvement. **Keep it disabled by default.**

The three control failures were evading agents with sprint trait 20. One had
recently moved from swamp into desert; two had only swamp observations in the
recorded three-second history. The experiment's three failures were also
evading agents: one had sprint 20 in swamp, two had sprint 11.39; one of those
crossed swamp, grassland and river in its recent history. These observations
identify cases to replay, not a proof that the last action alone caused death.
The all-agent zero-premature-capture benchmark remains unmet.

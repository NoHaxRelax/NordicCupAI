# Predictive harvest admission

Opt-in selector: `sacrifice_mode="predict_contact"`. Existing selector defaults are unchanged.

```python
harvester = Harvester(budget=20000, sacrifice_mode="predict_contact", contact_margin=1.0)
actions = harvester.apply(states, sim_time, base_actions,
                         score=public_score, payout_limit=remaining_transfer_points)
```

Call on every tick so motion history stays current; call `reset()` between games. The public score enables next-tick transfer confirmation and suppression of predicted resting positions. With no score, motion/contact filtering still works but no transfer is claimed as confirmed. `payout_limit` is optional; when supplied, invalid/non-positive values suppress the burst and positive values bound its predicted transfer payout. The surrounding run score guard remains responsible for survival/food points.

The selector uses public observation DTOs and prior actions only. It registers ego motion with static edges, associates nearby predator sightings, rejects stationary/ambiguous tracks, and forecasts the unobserved prior predator move plus its pending move. Forecasts consider capped turning, observed step length, a possible drop to walking speed, a crossing into the target's biome, other observed agents' actions, and remembered visible walls. A one-unit margin is required inside the strict 15-unit contact radius.

Successful-transfer memory is stored in surviving observers' coordinate frames and expires conservatively. It is discarded when observer motion becomes too uncertain. Predator IDs, actual resting state, actual energy, unseen walls/terrain and complete peer positions remain unavailable; the selector does not guarantee every transfer or any minimum survival duration.

`harvester.contact_predictor.stats` exposes rejection counts and `.confirmed` counts public-score confirmations. Each accepted burst records predicted contact distance, observed predator speed, actual burst budget and expected payout in `harvester.log`.

Local C++ comparison, source snapshots and recorded replays:
`survival/results/action-bug/contact-predictor-20260920/`.
The change has not been deployed by this task or tested against the validation API.

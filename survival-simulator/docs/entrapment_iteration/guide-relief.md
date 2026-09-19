# Guide relief forecast

Status: planning helper only, not connected to `core.py` roles or dispatch yet.
It does not currently move or replace guides in the full-game replay.

`models/entrapment/guide_relief.py` is a pure, observation-safe planning helper.
The coordinator supplies the current guide and donor DTO fields plus route
distance/time estimates made by `Navigator` from shared, localized observed
walls and poses. The helper never reads simulator positions, predator energy,
rest state, terrain, or target IDs.

`forecast_travel()` replays the native 0.1-second movement ledger: walking costs
0.05 energy per requested unit, movement beyond walking speed costs 0.5, sprint
is cut off below `max_energy / 5`, biome drain is charged per second, and agents
older than 60 lose `0.01 * age` each tick. It predicts route arrival, sprint
cutoff, death, and arrival energy. The caller should pass a conservative route
time when turns, avoidance, or known pacing make distance/speed optimistic.

`plan_relief()` compares the guide's bait route with every available donor's
route to a caller-selected rendezvous. It forecasts instead of waiting for the
20% cutoff, rejects donors that cannot survive their own trip, and prefers the
candidate with the latest safe dispatch time. The selected reserve remains an
ordinary gatherer until `dispatch_in` expires. The coordinator should then give
it the guide role and route it to `rendezvous`.

The old guide is releasable only when all three normal-observation conditions
are passed as true: the replacement is at the rendezvous, it sees the predator,
and the old guide sees evidence that the predator follows the replacement. This
is deliberately stricter than arrival alone and does not claim access to a
hidden predator target ID.

Integration inputs are explicit `AgentForecastInput` records. Set donor
`route_distance`/`route_time` from a non-mutating navigator probe or a dedicated
route cache; mark blocked, wrong-group, uncertain-pose, bait, nursery, current
guide, or otherwise committed agents `available=False` (or omit them). Pass a
stable observed-frame handover point as `rendezvous`. Call on route/map/energy
changes and when `dispatch_in` crosses zero; do not reserve the donor's actions
before that time.

The forecasts are estimates, not proven bounds or guarantees. Turning energy is
not charged, future gathering during a delayed dispatch is not simulated, and
the conservative aging threshold is 60 rather than an unknown inherited max age.
Unknown terrain ahead, collision
avoidance, newly observed walls, changing guide pace, lost predator contact,
and ambiguous predator attention can lengthen either route. Conservative route
times and handover margins absorb some uncertainty, but the coordinator must
replan as observations arrive. The default guide forecast assumes sprint is
needed until bait; callers may disable that constraint when the remaining route
is safely walkable.

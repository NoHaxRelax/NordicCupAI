# Guide planning and shared traffic forecasts

V1 experiment adds a three-tick beam search (six first-action alternatives),
shared predicted predator paths for non-guides, and six extra seconds of
bait dispatch lead when a replacement is not full. Fruit candidates expand
to 60 units away / 20 from the next route segment / 30 extra path units;
energy and arrival safety are still checked without assuming fruit is eaten.
Replacement diagnostics explain fruit rejection or avoidance on each tick.
Normal survival behavior otherwise remains unchanged.

Only normal DTO sightings, own energy/age/traits, and known walls enter the
policy. V1's chase model uses a 15-unit homing approximation and a 21/33-unit
safety margin; it is not an exact future simulation. The user correctly
requested removal of this extra clearance and use of the game's actual
turning and capture semantics. That is the next iteration.

The first full development-seed run scored 592.23, extinct at 579.9 seconds,
compared with the preceding baseline's 872.19 / 821.3. This combined change
is not a demonstrated improvement. Every frame is saved in
`logs/entrapment-iteration/guide-lookahead-v1-20260919`, viewer port 9077.
Small cyan/salmon dots show the guide/predator forecast in spectator rendering.
The model is approximate; three ticks cover 0.3 seconds only.

Replacements selected fruit for 68 recorded ticks. Other nearby candidates
were rejected 758 times as off-route, 185 for the handoff deadline, 48 for
walls and 5 for excessive detour length; these are tick counts, not unique
fruit counts. No future fruit energy is credited before it is actually eaten.

Focused checks confirmed sprinting around a predator between guide and bait,
the native low-energy walk cap, and a bystander turning away from a forecast
crossing without a direct sighting. The small search took about 38 ms for a
guide decision in the open-map check; the full game took 142.7 wall seconds.

Local comparisons separately disable the search, shared paths, and both,
while retaining the same new bait-food rules. Sol is independently preparing
a guide-relief planner for forecast sprint exhaustion and viable rendezvous.

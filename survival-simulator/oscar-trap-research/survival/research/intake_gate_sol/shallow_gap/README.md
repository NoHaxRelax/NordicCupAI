# Shallow-gap bait / HOLD42.5 experiment

This frozen observation-only candidate places the refuge bait 15 units inside
an observed predator-excluding gap. Every later supplied guide maps its own
copy of the gap and holds centered 42.5 units outside the nearest mouth. The
intent is geometric: a predator stopped by the 11--19 unit mouth should remain
closer to the shallow bait than to a new guide, while the guide's follower is
within the bait's 60-unit hearing range after sacrifice. The tests below measure
those properties; the policy is not given fixture coordinates or dimensions.

## Inputs and fixture limits

The policy receives only JSON-round-tripped native Agent observation DTOs and
public simulation time. Predator IDs, energy, rest state, true positions,
fixture dimensions, and scorer state remain outside the policy. The fixture
arranges two obstacles, one initial bait, and an off-axis guide/predator pair
for each scheduled arrival. Recruitment and travel to those starts are not
demonstrated. Living agents receive unlimited food; predator movement, sensing,
collision, target selection, rest, and capture remain native.

## Criterion

Acquisition requires 20 consecutive ticks both physically near the mouth and,
while active, targeting an agent inside the gap. A successful run requires all
scheduled predators to acquire, no joint or physical loss afterward, all held
throughout the final 30 seconds, the bait alive, and the full horizon complete.
The evaluator separately records physical loss and active target switching.

## Recorded screen (three runs total)

1. `gap=15`, unequal `90x70` faces, offset `-5`, horizontal, four arrivals at
   90-second spacing, seed 4701: **pass**. All 4 acquired, zero losses, bait
   alive, final/tail count 4.
2. The same geometry with 33 arrivals through 3,000 seconds, seed 4702:
   **fail**. All 33 initially acquired, but 28 later had joint losses and 26
   physically escaped; final count was 11 and the final-30-second minimum was
   10. The first target losses began at 456.7 seconds during the sixth guide
   intake, and the first physical losses followed around that guide's capture.
3. Native-like unequal geometry (`gap=14.9669`, faces `81.7861x95.5125`,
   offset `-14.7551`, vertical) with 33 arrivals through 3,000 seconds, seed
   4703: **pass**. All 33 acquired, zero target or physical losses, final/tail
   count 33, and bait alive.

Across all three tests the bait reached depth 15 and survived. Minimum observed
predator-to-bait center distance was 25.14--25.20, above the 15-unit combined
contact radius. This directly supports bait safety for these fixtures.

The opposed full results show that the strategy is geometry/orientation
sensitive. In the failing horizontal fixture, alternating-side guides were
captured at two different phases (`x.4` versus `x.2` seconds). Starting with the
sixth intake, old predators switched during the delayed guide approach and
escaped before the bait could recover them. In the passing vertical fixture,
all guide captures occurred at the same phase (`x.4` seconds), and no switch
was measured. This is evidence from one seed per full geometry, not a general
orientation theorem or a production-ready 33-predator solution.

## Sources and receipts

- `observed_gap_base_v2_snapshot.py` SHA-256:
  `eb61686db26bffedd1248db78bd6c8784881074d9f6eca2b38f9667118038459`
- `observed_shallow_gap_policy.py` SHA-256:
  `ee296e5230dd8dc695dbf5e4b603e8ae2d9096147aa3cc0c2b29145d64de7f2d`
- Combined replay policy receipt:
  `7deb2a52b96c20782ffc1dc6afcd0cd364cfb92877b7cc29e30f18b546b733c7`

Reproduce from `survival-simulator/oscar-trap-research`:

```sh
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_gate_sol/shallow_gap/run.py --predators 4 --interval 90 --seconds 400 --seed 4701 --gap 15 --length 90 --right-length 70 --face-offset -5 --lateral 60 --horizontal
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_gate_sol/shallow_gap/run.py --predators 33 --interval 90 --seconds 3000 --seed 4702 --gap 15 --length 90 --right-length 70 --face-offset -5 --lateral 60 --horizontal
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_gate_sol/shallow_gap/run.py --predators 33 --interval 90 --seconds 3000 --seed 4703 --gap 14.9669 --length 81.7861 --right-length 95.5125 --face-offset -14.7551 --lateral 60 --vertical
```

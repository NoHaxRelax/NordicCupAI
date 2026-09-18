# Observation-only wall funnel: 33-predator capacity

The strongest tested setup is **one 30×100 wall, two permanent bait agents, and
one sacrificial guide**. It acquired all 33 approaching predators and held them
through a 3,000-second simulation, with no escape after acquisition and both
baits alive. Ten fresh 300-second tests also held all 33, including awake starts
and horizontal walls. Six later tests with imperfect initial bait placement
passed after the controller learned the wall and positioned the baits itself.

**This establishes capacity for a prepared approaching group, not autonomous
collection of 33 scattered predators or reliable repeated arrivals.** Independent
waves and automatic guide replacement still failed. Do not advertise the current
controller as an indefinite full-map funnel. The simulation horizon is finite.

Source: the original `codex/survival-trap-handoff-2026-09-17` branch, continued in
the isolated `codex/wall-funneling` worktree. Native simulator code is unchanged.
Guide sacrifice is explicitly allowed; food availability is assumed throughout.

## Recommended placement

Use local coordinates with the rectangle occupying `0 ≤ x ≤ thickness` and
`-length/2 ≤ y ≤ length/2`, predators approaching from negative x.

| Role | Desired position |
| --- | --- |
| Bait 1 | `(thickness + 5.1, -10)` |
| Bait 2 | `(thickness + 5.1, +10)` |
| Guide's final handoff point | `(-5.1, 0)` |
| Guide's collection point | `(-123.1, 0)` |

These are **coordinates inferred from observed wall edges**, not world
coordinates supplied to the controller. The same controller operates after a
90-degree rotation. For a small trap with only one bait, use the far-face midpoint.

The crew first registers relative positions and headings using ordinary Agent
observations. The guide sees the front face; the baits see the far face. The two
observed parallel edges supply thickness, length, endpoints and a local normal.
The controller then moves the protected agents to the desired bait positions.
It has no supplied wall map or known world origin.

The guide goes out to the collection side, reacts to nearby predators, then
returns toward the front-face midpoint. It waits if the observed predator is too
far away and sprints when close. **During delivery it faces away from the
predators**, which avoids inducing the native predator's long-range circling
behavior. The guide is caught; pursuit transfers to the protected baits. The
default controller then leaves those baits stationary and requests no births.

Two baits matter because a predator oscillating along the wall can lose hearing
of a single midpoint bait. It can select the closer of two protected baits as it
moves sideways. This fixed the single-bait tests that consistently missed one
of the 33 predators. Moving extra guides through an occupied trap has the opposite
effect: they can become reachable targets and disrupt containment.

At the closest legal predator position, the cross-wall center distance is at
least `thickness + 15.1`. Predator hearing reaches 60 units; founder hearing only
50. Thickness 35 already puts a directly opposite predator just outside the
founder's hearing. A bait surviving, or not sensing a predator, therefore does
not prove that the predator remains trapped. Evaluation checks both position
and actual target selection, and distinguishes temporary rest from active pursuit.

## Wall-size results

Native random obstacles have independently sampled widths and heights from
30 to 100. Dimensions below are **thickness × length**; transpose them for
horizontal walls. These are tested configurations, not proved maximum capacities.

| Wall | Supported configuration | Evidence and qualification |
| --- | --- | --- |
| 30×70 | 1 bait, 3 predators | 3/3 at 120 s; the 2-bait/33-predator test failed. |
| 30×85 | 2 baits, 33 predators | Passed one 180 s size test; needs broader validation. |
| **30×100** | **2 baits, 33 predators** | **10/10 fresh 300 s tests, one 3,000 s hold, and 6/6 final placement tests.** |
| 32×70 | 1 bait, 3 predators | 3/3 at 120 s; the 2-bait/33-predator test failed. |
| 32×85 | 2 baits, 33 predators | Passed one 180 s size test; needs broader validation. |
| 32×100 | 2 baits, 33 predators | Passed one 180 s size test; needs broader validation. |
| 35×70 | 1 bait, 3 predators | 3/3 at 120 s; the 33-predator test failed. |
| 35×85 | 1 bait, 3 predators | 3/3 at 120 s; two baits failed after acquiring 32. |
| 35×100 | 1 bait, 3 predators | 3/3 at 120 s; the 33-predator test retained 32. |
| Length below 70, or thickness above 35 | Reject in this controller | Short-wall failures and limited hearing margin; no supported observation-only deployment. |

The small-capacity sweep used three approach seeds per geometry, including a
horizontal case. Its 27/27 result covers an approaching group of three, not three
separate arrivals. The larger size screen used one seed per size and should not
be treated as equally strong evidence as the 30×100 validation.

If practical capacity is three, 33 predators require at least 11 separate traps;
if the 33-capacity configuration is available and its approach conditions can be
arranged, one trap suffices. This arithmetic does **not** establish map availability
or safe routing between traps. The handoff's natural-map census found a median
of only three physically usable walls, and interference between several traps
was observed here when a predator was missed.

## What the validation actually did

- The controller receives JSON-serialized native observation DTOs and public
  simulation time only. No positions, predator IDs, rest flags, predator energy,
  fixture seed, wall dimensions or evaluator feedback enter `act`.
- World coordinates and target IDs are used only after actions, by the evaluator
  to measure capture and retention. Early `run.py` probes used privileged control;
  they are separate historical experiments and are not the proposed solution.
- Tests arrange a crew near both wall faces and predators on the approach side.
  The 33-predator validation starts them 190–230 units in front of the wall, with
  lateral offsets within ±15 and heading perturbations within ±0.3 radians.
  Half of the ten fresh cases start awake; half start with native fresh rest.
- Final placement tests perturb bait positions by up to eight units along the
  wall and four units away from it. The controller corrects placement from sensed
  edges. The final scorer uses actual bait positions, avoiding any dependence on
  which agent occupies which endpoint of the bait line.
- All living agents receive a food refill before every tick. Predator energy,
  rest, movement, sensing and contact captures remain native. This is a stronger,
  explicit implementation of the no-food-problem assumption, not demonstrated
  food production. The successful 33-predator runs lose one guide and no baits.
- Terrain is flat forest with physical arena boundaries and the tested wall(s).
  Neighboring random obstacles, trees, global exploration and ordinary stochastic
  predator spawning are not part of these fixtures.
- Acquisition requires two seconds of consecutive containment. A successful
  final hold requires every predator to remain held throughout the last 30 seconds.
  `continuous_all` additionally rejects every loss after acquisition. A resting
  predator must still be geometrically contained; an active one must target a bait.

## Unresolved repeated-arrival problem

One guide delivering a group and then retiring the trap is the supported mode.
Sending additional groups at 20-second intervals failed, with and without new
guides. The observation-based rest gate infers stationarity from successive
positions, fuses both baits' observations as multisets, and rejects missing
targets. It does not read rest flags. Nevertheless, its acquisition and waiting
routes did not preserve the trap in these sequential tests.

A guide orbit intended to gather scattered predators also failed and remains
an optional research control. Leave `--gather` and `--renew-guides` off. The next
task is an observation-only delivery route for a free predator to a fresh or
occupied trap, tested with scattered/unarranged arrivals. More holding-only
tests would not resolve that remaining problem.

A final separate intake test supplied a guide at a prepared approach point with
each later predator, instead of relying on births and a route from behind the
wall. Each guide used only its own observed front edge and predator bearings.
Two of three four-arrival trials held all four; the third held three. All three
33-arrival attempts failed the full-retention criterion: two lost the baits,
and one retained 32. This experimental controller is kept separately in
`observed_policy_experimental.py`; it does not change the validated default.

## Reproduce and inspect

From `survival-simulator/oscar-trap-research`, using a Python environment with
the simulator requirements installed:

```sh
python survival/research/wall_funneling/observed_run.py --width 30 --length 100 --baits 2 --predators 33 --seed 62 --initial-jitter 8 --awake --seconds 180 --native
python survival/research/wall_funneling/observed_run.py --width 30 --length 100 --baits 2 --predators 33 --seed 42 --seconds 3000
python survival/research/wall_funneling/checks.py
python survival/debugger/catalog.py
python survival/debugger/serve.py
```

- [Controller](../survival/research/wall_funneling/observed_policy.py),
  [experiment harness](../survival/research/wall_funneling/observed_run.py),
  [existing debugger adapter](../survival/research/wall_funneling/adapter.py).
- [Small-capacity size sweep](../survival/results/wall_funneling/observed-size-sweep.json).
- [Fresh 33-predator and size validation](../survival/results/wall_funneling/observed-two-bait-validation-v6.json).
- [Final controller placement validation](../survival/results/wall_funneling/observed-placement-validation-v9.json).
- [Native demonstration receipt](../survival/results/wall_funneling/native-demo-v9.json).
- [Native final frame](../survival/results/wall_funneling/native-33-predators-v9.png).
- [3,000-second replay](../survival/results/wall_funneling/replays/observed-w30-l100-n33-sites1-s42-spread15.0-waves1-gate1-h0-a0-555a836d.json.gz).
- [Failed sequential arrivals](../survival/results/wall_funneling/observed-sequential-two-bait-v7.json).
- [Separately supplied guide arrivals](../survival/results/wall_funneling/observed-staged-guides-v10.json).

Every completed simulation, including failures and superseded versions, has a
unique native-state replay. Two native-rendered early privileged demos and the
final observation-only native demo are labeled separately. Archived controller
files preserve recorded source hashes. The 3,000-second run and the ten-seed
validation used v6; the final v9 additionally derives bait placement from sensed
edges and was separately validated. Replay files remain local, excluded by the
handoff's existing Git ignore rules. Other research folders are untouched.

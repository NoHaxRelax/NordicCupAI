# Observation-only sequential intake through a size-selective gap

The strongest result in this folder uses one bait inside a 15-unit passage
between two ordinary native-sized obstacles. Agents have radius 5 and can enter;
predators have radius 10 and stop at the mouth. Unlike the 30x100 wall trap, the
predator-to-bait distance is about 40 units and a newly approaching predator can
select the protected bait directly through the open mouth. No guide passes
through an occupied trap.

The controller maps the two opposing faces from native `Edge` observations,
walks 30 units into the nearest mouth, faces inward, and holds. Its `act` method
receives only JSON-round-tripped native observation DTOs and public simulation
time. Fixture coordinates, obstacle objects, predator rest/energy/targets, and
scorer results never enter the policy. Agent food is explicitly supplied by
resetting energy before each tick.

## Primary evidence

All simulations below called `ReplayRecorder.capture` after every step and have
unique completed replays under `survival/results/intake_geometry_sol/replays/`.

| Policy / case | Result |
| --- | --- |
| v1, seed 301, 3 arrivals every 12 s, 180 s | 3/3 acquired; zero physical or attention loss |
| v1, seed 302, 33 arrivals every 12 s, 900 s | 33/33 acquired; zero physical or attention loss; final and last-30-second minimum 33 |
| **v1, seed 303, 33 arrivals every 90 s, 3,000 s** | **33/33 individually acquired from t=6 through t=2,886; zero physical or attention loss; final and last-30-second minimum 33; bait alive** |
| v2, seed 304, unequal 100/75 faces offset 15, vertical | 3/3 through 180 s |
| v2, seed 305, unequal 90/70 faces offset -5, horizontal | 3/3 through 180 s |
| **v2 native-rendered demo, seed 306, same horizontal unequal geometry, 33 arrivals every 3 s** | **33/33 through 180 s; zero loss; 653 frames rendered by upstream `Environment.draw`** |
| **v2, seed 501, dimensions copied from native-map seed 11 candidate, vertical, 33 arrivals every 90 s, 3,000 s** | **33/33 acquired; zero physical or attention loss; final and last-30-second minimum 33; bait alive** |
| **v2, seed 502, dimensions copied from native-map seed 28 candidate, horizontal, 33 arrivals every 90 s, 3,000 s** | **33/33 acquired; zero physical or attention loss; final and last-30-second minimum 33; bait alive** |

The successful v1 policy hash is
`ffec7e00d2a97560adf4b7730a666663f47730b29ab21b9540b43af162ccd3d2`.
It is preserved at
`survival/research/intake_validation/observed_gap_original_v1.py` by the
independent audit. The generalized v2 hash is
`d1e0cb4ded29d2915c9c56c0c1bf1efa32f351e350c81f5cb0dd3c93c20514fe`.
V2 computes the actual overlap of parallel faces, so it no longer assumes equal
obstacle lengths or aligned endpoints. Rotation here means the native 90-degree
axis rotation; native obstacles are axis-aligned.

The two fresh full-horizon v2 fixtures copy exact dimensions from actual census
candidates while retaining the controlled-map and aligned-approach assumptions:

- Native seed 11, obstacle indices 21/30: vertical passage, gap
  14.9669209025, along-passage face lengths 81.7860682849 and 95.5124660953,
  relative face offset -14.7551377435, overlap 80.7573283518. The reproduced
  simulation uses fresh RNG seed 501.
- Native seed 28, obstacle indices 51/70: horizontal passage, gap
  15.3263118191, along-passage face lengths 86.9616377621 and 79.0561549509,
  relative face offset +8.4697514406, overlap 78.4918863215. The reproduced
  simulation uses fresh RNG seed 502.

These reproduce the measured passage geometry only. They do not retain the rest
of either generated map, its biome, neighboring obstacles, or natural approach.

Independent size checks outside this folder also passed 33 arrivals spaced 90
seconds apart through 3,000 seconds at both screened extremes: gap 11 with
55-unit passage length (seed 411) and gap 19 with 100-unit length (seed 412),
with zero physical or attention loss.

## Acquisition envelope and negative guide pilot

The strong runs arrange each new predator on the open passage axis, 160 units
from the mouth, with lateral jitter up to 6 and heading jitter up to 0.25 radians.
Independent stress tests show acquisition, rather than post-acquisition holding,
is the limit: lateral +/-25 and heading +/-0.5 acquired 32/33 awake and 33/33
resting; lateral +/-60 and heading +/-0.8 acquired 26/33 awake and 22/33 resting.
None of the predators that did acquire were physically lost.

One recorded off-axis guide pilot in this folder is a negative result. A fresh
guide for each of three arrivals mapped the gap from its own observations and
waited 20 units outside the mouth. With lateral +/-60 and heading +/-0.8, only
2/3 predators acquired. All three guides were captured, the two acquired
predators stayed physically held, and the temporary guide targets caused
attention switches. This overlaps the separate guide-protocol work and is not
the recommended controller.

## Map availability and limitations

A pre-existing read-only oracle census of native generated maps found at least
one eligible 11--19 unit passage with 55 units of overlap on 19/40 maps. Counting
unique obstacle pairs gives 23 passages: 10 in seeds 1--20 and 13 in seeds
21--40. Most eligible maps had one passage; this is much more available than a
bank of 33 isolated walls, but it is not universal. The source inventories are
`survival/results/water_deployment/alternative-gap-survey.json` and
`alternative-gap-survey-21-40.json`.

The successful runs still use substantial setup privilege: the obstacles and
mouth are arranged, the bait starts 15 units outside the correct entrance, and
predators are introduced on a clear approach at scheduled public times. The
native-rendered v2 replay uses native dimensions and the original renderer on a
controlled flat map; it is not a generated-map deployment. The work does not
demonstrate site discovery, routing a colony or arbitrary off-axis predators to
the mouth, bait replacement, natural food, ordinary random spawning, or a
competition/full-game score. Random nearby obstacles could block the approach
or offer competing routes and targets. A holder must remain in the refuge.

Physical retention and target attention are scored separately. Physical
containment requires the predator to stay within 75 units of the mouth and on
the excluded side/depth of the passage. Attention additionally requires every
active predator to select the bait; resting predators are accepted only while
physically contained. Acquisition requires two consecutive seconds, and the
reported tail minimum covers the final 30 seconds.

## Reproduce

From `survival-simulator/oscar-trap-research`:

```sh
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_geometry_sol/run.py --predators 33 --interval 90 --seconds 3000 --seed 303 --every 50
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_geometry_sol/run.py --predators 33 --interval 3 --seconds 180 --seed 306 --length 90 --right-length 70 --face-offset=-5 --horizontal --native --every 20
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_geometry_sol/run.py --predators 33 --interval 90 --seconds 3000 --seed 501 --gap 14.966920902528898 --length 81.78606828487959 --right-length 95.51246609533972 --face-offset=-14.75513774349713 --every 50
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_geometry_sol/run.py --predators 33 --interval 90 --seconds 3000 --seed 502 --gap 15.326311819067314 --length 86.96163776214033 --right-length 79.0561549508511 --face-offset=8.469751440614914 --horizontal --every 50
```

The first command now runs generalized v2; the recorded seed-303 receipt is the
preserved v1 run. The current harness also has `--guides` solely to reproduce the
recorded negative pilot.

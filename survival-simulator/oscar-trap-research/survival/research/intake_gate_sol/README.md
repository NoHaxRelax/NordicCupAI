# Observation-only sequential intake gate (Sol)

This bounded experiment produced a **three-predator-per-wall** candidate that
passed the recorded tests below. It did not solve 33 sequential arrivals at one
wall, and it did not test coordination among eleven walls.

## Candidate

`policy_pressure.py` layers a capacity-three contact-pressure fallback over the
sensor gate in `policy.py`. Two protected agents hold the far face of an
observed 30x100 wall at 5.1 clearance and tangent offsets -10/+10. A supplied
guide holds at 90 units from the near face. The bait sensors fuse native
Predator observations as a multiset and require unchanged position and native
relative heading before declaring a sleep window. If the arriving predator
reaches the guide's native 50-unit hearing range just before the window is
confirmed, the guide commits rather than being caught at staging. The wall
retires once three predators have been observed.

The controller receives only JSON-serialized native observation DTOs and
public simulation time. Predator IDs, true positions, energy, rest flags,
fixture geometry, and scorer feedback are not passed to it. The fixture
supplies a guide at a prepared approach point for every arrival and refills all
living agent food each tick. The initial guide begins close enough to both
baits to connect their observed near/far wall faces; later guides start 125
units in front of the near face.

## Success criterion and evidence

A predator is physically held when it is within 60 of a living bait, remains
on the near side at the full wall separation, and stays at least 5 units from
either wall end. Acquisition requires 20 consecutive held ticks. A run passes
only if every scheduled predator is acquired, none ever leaves the physical
region after acquisition, all are held throughout the final 30 seconds, both
baits live to the horizon, and the requested horizon completes. Target choice
is measured separately so a brief target switch is not mislabeled as a
physical escape.

Final candidate results:

- Sparse intake, arrivals at 0/90/180 seconds: 3/3 fresh 360-second runs passed
  (seeds 222-224).
- Long retention: seed 225 passed 3,000 seconds with all three acquired, zero
  physical losses, zero target switches after acquisition, and both baits alive.
- Three additional fresh 3,000-second runs passed (seeds 451-453), again with
  zero physical losses and both baits alive. Seed 452 started every arrival
  awake; the other two used native resting starts. Seed 451 had one brief target
  switch without a physical escape.
- Burst stress, arrivals at 0/12/24 seconds: 3/3 180-second runs passed (seeds
  226-228). These had 1-2 brief target switches but zero physical losses.
- There were 21 total recorded pilots, including all failed intermediate
  policies. The final candidate passed 10/10 of its recorded cases under the
  stated arranged conditions.

The long replay is
`survival/results/intake_gate_sol/replays/intake-gate-sol-v3-n3-i90-s225-a0-588ed5bc.json.gz`.
Every completed run has its own ReplayRecorder file and result JSON.

## Remaining failure

Capacity above three is not established. In the four-arrival experiments, a
stationary guide could be caught one tick before the old mass entered the
observable sleep window. Orbiting temporary holders avoided that contact but
turned the incoming predator laterally; the predator then continued past the
guide's final sprint and wandered away. Repeated misses eventually reached and
killed the protected baits. The capacity-three policy therefore retires the
wall instead of admitting a fourth arrival.

Eleven such traps have arithmetic capacity for 33 predators, but trap
discovery, routing, wall assignment, and simultaneous 3,000-second retention
were not implemented or validated here.

## Reproduce

From `survival-simulator/oscar-trap-research`:

```sh
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_gate_sol/run.py --predators 3 --interval 90 --seconds 360 --seed 222
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_gate_sol/run.py --predators 3 --interval 90 --seconds 3000 --seed 225
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_gate_sol/run.py --predators 3 --interval 12 --seconds 180 --seed 226
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/debugger/catalog.py
```

Current source hashes:

- `policy.py`: `89886dd572e404fbe3003fb0721358958aa61541f52bd6dba9849a817b069230`
- `policy_stationary.py`: `45aeb54dc5b9a3d8127ee1e70bac72553312a6b390a9ab451b4cabb874ae0a66`
- `policy_pressure.py`: `bcb1bb741d50dfd49cbe72636221d54ddb7b11a81f40d1bba0d2f2d0a2e240e7`

The seven overwritten exploratory sources were reconstructed exactly by
replaying this agent's recorded `patch_apply_end` diffs from its Codex rollout
JSONL. Every reconstructed file's SHA-256 matches its replay receipt. They are
retained under `archived_sources/`; the final candidate and its two direct
dependencies are retained at the hashes above.

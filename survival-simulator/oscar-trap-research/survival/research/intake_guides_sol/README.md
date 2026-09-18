# Observation-only repeated guide intake

The successful local protocol is `observed_gap_guide_policy_v4.py`, backed by
the pinned generalized mapper in `observed_gap_base_v2_snapshot.py` and the
fixture/evaluator in `gap_guide_run_v4.py`.

The controller receives only JSON-compatible native agent observation DTOs and
public simulation time. It does not receive fixture coordinates, obstacle
objects, predator IDs, target choices, energy, rest flags, or evaluator state.
The environment supplies food by refilling living agents before each tick.

## V4 protocol

1. The initial bait reconstructs two parallel observed faces separated by an
   11--19 unit gap, computes their actual overlap, enters 30 units through the
   nearest mouth, and remains still.
2. Each later, separately supplied guide reconstructs the same geometry in its
   own local frame and sprints to the passage axis 125 units outside its nearest
   observed mouth.
3. The guide faces inward and waits until its own native observations contain a
   predator behind it (`abs(angle) > pi/2`) within 50 units.
4. It then sprints 50 units along the observed passage axis and holds 75 units
   outside the mouth. The follower captures the guide there. The guide never
   enters the held predators' 60-unit hearing zone. The now-aligned follower
   continues through the open sightline to the protected bait and stops at the
   size-selective mouth.

The successful fixture used a horizontal 15-unit gap between ordinary 90- and
70-unit faces offset by -5 units, leaving 65 units of overlap. Thirty-three
guide/predator pairs were supplied 150 units outside the mouth at alternating
cross-axis offsets of +/-60. Each predator began 65 units behind its guide;
arrivals were 90 seconds apart from t=6 through t=2,886. Food, prepared arrivals,
and the obstacle pair are explicit fixture privileges. Recruitment, travel to
the prepared starts, site discovery, neighboring clutter, arbitrary arrival
directions, and natural-map deployment are not demonstrated.

Policy source SHA-256 is
`7a79888dd5395e5d7dcf66010f0367aeb6452932cc0ea38a7a787c63d824be96`;
the pinned mapper is
`d1e0cb4ded29d2915c9c56c0c1bf1efa32f351e350c81f5cb0dd3c93c20514fe`.
Replay metadata hashes their concatenation as
`c5da1e734597d279f79a4a9455d30ad21b7b63913a91422ad70cd47911098c2d`.

## Evidence

Seed 4330 first passed both lateral signs with four arrivals through 400
seconds: 4/4 acquired, zero target-attention or physical loss, maximum
post-acquisition mouth distance 29.706, and all four guides sacrificed.

Seed 4331 then ran the requested full horizon. All 33 predators acquired and
remained continuously valid through 3,000 seconds: zero attention losses, zero
physical losses, final and final-30-second minimum 33, maximum post-acquisition
mouth distance 33.281, and maximum continuous active target-switch duration
0.0 seconds. The bait survived and all 33 guides were sacrificed.

The exact result and replay are:

- `survival/results/intake_guides_sol/observed-gap-guides-v4-aligned-hold-g15-l90x70-o-5-h1-n33-i90-lat60-s4331-e830bf1d.json`
- `survival/results/intake_guides_sol/replays/observed-gap-guides-v4-aligned-hold-g15-l90x70-o-5-h1-n33-i90-lat60-s4331-e830bf1d.json.gz`

Reproduce from `survival-simulator/oscar-trap-research`:

```sh
/home/Ucals/projects/NordicCupAI/.venv/bin/python survival/research/intake_guides_sol/gap_guide_run_v4.py --predators 33 --interval 90 --seconds 3000 --seed 4331 --gap 15 --length 90 --right-length 70 --face-offset=-5 --horizontal --lateral 60 --approach 150 --follower-gap 65 --heading-jitter .2
```

## Preserved failures

- Rest-window orbiting at the occupied 30x100 wall never acquired all four;
  strict rest transition detection avoided unsafe commits but waiting guides
  were eventually caught.
- A front-side tangential sweep acquired at most 3/4 and killed the protected
  baits in two of three seeds.
- The first gap route acquired 4/4 and 8/8 without physical loss, but its
  33-arrival run lost one old predator during a diagonal guide approach and
  finished with 32.
- V2 centered directly at 75 outside and charged into the unequal gap. It
  acquired 33/33 but later guide sacrifices pulled 13 old predators through
  obstacle-end collision slides; 22 remained at t=3,000.
- V3 held directly at 75 outside. It protected all acquired predators but only
  aligned 2/4 newcomers because residual diagonal heading could miss the bait
  after guide death.

Every completed run has a unique `ReplayRecorder` replay. Exact superseded wall
policy sources are retained in `archive/`; their file hashes match replay
metadata. A final catalog scan found 682 replay files, zero invalid files, and
the seed-4331 replay at duration 3,000 seconds.

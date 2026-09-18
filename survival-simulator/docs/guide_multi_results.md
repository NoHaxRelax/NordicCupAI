# One-map 30 + 3 delivery test

Map seed `424736271`, encounter seed `1335789813`, unchanged current guiding
policy. Run recorded in `logs/guide_lab/multi-424736271-1789733097608953745/`.

**Result: 30/33 held at the end; 0/3 newcomers successfully delivered.**
All original 30 were within 40 units of the bait and could sense it at every
recorded tick, including the final 30-second observation period. Bait survived.

| Newcomer | Spawn time | Guide death time | Killer | Result |
| --- | ---: | ---: | --- | --- |
| 31 | 10.0 s | 28.7 s | Predator 31 | Failed; predator about 100 units from bait at death |
| 32 | 48.7 s | 49.3 s | Predator 32 | Failed; predator about 590 units from bait at death |
| 33 | 69.3 s | 98.1 s | Predator 31 | Failed; earlier loose newcomer killed this guide |

Each attempt continued for 20 seconds after guide death to allow a successful
sacrificial handoff. None met the 10-second continuous hold criterion.
Death itself does not count as failure if the predator is delivered.

## Protocol and limits

- Initial 30 were preplaced in the validated approach lane and allowed to settle
  for 10 seconds. This does **not** demonstrate delivery of the first 30.
- Three encounters sampled sequentially on this map, with full-energy guides
  initially seeing their newcomer, at least 250 units from bait.
- Guides use native energy, aging and observations; all map edges known.
- Bait remains full-energy and unaged; it can still be eaten.
- Native predator energy, resting, sensing and collisions are unchanged. Native
  predators can overlap. Ambient predator spawning disabled to isolate 33.
- Earlier loose predators persist and can affect subsequent attempts.
- Evaluation uses hidden state only to measure retention, never as policy input.
- Recorded 148.1 simulated seconds / 1,482 frames at 0.1-second intervals.
  Local runtime was 104.4 seconds, with no new Runpod resources.
- This is one diagnostic map, not a reliability estimate or whole-game test.

The original group's retention worked here; delivery needs improvement. In
particular, a failed delivery can create an additional threat to the next guide.

## Replay and reproduction

Start the local replay server with:

```sh
.venv/bin/python survival-simulator/scripts/guide_lab.py --serve-only
```

Open <http://127.0.0.1:9056/multi-424736271-1789733097608953745/index.html>.
Buttons jump to each delivery, each guide death, and the final state. The native
render includes a magnified trap inset and predator counts. Every input, action
and evaluation is available in the inspector and compressed trace.

Re-run this experiment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python survival-simulator/scripts/guide_multi.py --seed 424736271 --encounter-seed 1335789813
```

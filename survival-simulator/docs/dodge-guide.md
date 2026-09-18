# Dodge guide: the leash lead with turning-circle dodges instead of sprinting

18 September 2026. Oracle world, unmodified engine at `acfc31a4`. Code: `models/trapper/dodge.py`
(opt-in, nothing in `lure.py`/`manager.py` changed), arena test `scripts/trapper/test_leash_dodge.py`,
game runner `scripts/trapper/run_game_dodge.py`, paired batch `scripts/trapper/batch_dodge.py`.
Background and the exact-model searches: Oscar's research checkout,
`nordic-ai-cup-2026/survival/research/guide_dodge/README.md`. Narrow-gap sites only.

## Idea

The stock leash keeps 40-55 in front of a charging predator by matching its speed, so it sprints
on almost every approach tick (42-46 of 46 ticks in the arena) and needs a guide with 330 energy.
A predator's direct chase turns at most 0.3 rad per tick: at 15 per tick it circles a radius of
50, at 11 a radius of 37, and cannot reach an agent beside it. The dodge guide walks with the
predator inside hearing, lets it close to about 20, then steps into that circle so it overshoots
and has to turn round, and walks on while it does. Exact-model searches show the dodge costs one
20-unit sprint tick against a sprinting predator and nothing against a walking one (below 40
energy it cannot sprint again until it rests).

## What changed in the leash

`DodgeLure(Lure)` replaces the movement choice of the approach and, at a staffed mouth, of the
run down the axis; the hook, the rest handling, the path planner, the run at an empty mouth (the
predator must stay behind a guide that enters the passage) and the enter/flyby endgames are the
stock leash's. Per tick it scores 24 directions x {0, 5, walk, 15, sprint} by

- a three-tick existential safety search against a vectorised clone of the public predator rule
  (collision deflection included, other agents as alternative targets, other predators' next step):
  after the move, does a second step (walk or sprint) and a third (walk) exist that keep every gap
  at or above 16? A third step with the predator still pointed at the guide is discounted 12;
- a clearance reward up to 50 (room after a dodge beats hugging the kill radius);
- progress toward the path waypoint, faded out inside gap 33 (the move that makes it overshoot
  wins there), energy cost weighted 3:1;
- a penalty on sprinting directly away from the predator (that is the stock speed matching; with
  a finite horizon it postpones the dodge for ever), on drifting beyond hearing when the predator
  is not already closing, on slower ground, and on entering a staffed passage.

Three stock lines are patched from source at import time (assertions fail loudly if `lure.py`
changes them): the sprint-budget fallback (118 energy, now max_energy/5 + 12), the rest distance
(42, now 50: a walker cannot dodge a wake-up lunge from 42 next to a mouth) and the truncation of
the last move to the remaining distance to the goal (kept only when entering a passage; it cut
dodge steps to a few units at the flyby point). `DodgeManager` lowers the leash gate to 160 and
uses a lead budget of (energy - 90) / 0.25 instead of (energy - 300) / 0.09.

## Arena (leash arena, random obstacle fields, paired seeds, 60 s)

| Guide | Alone, 500 energy | Second predator loose, 500 | Alone, 200 energy |
| --- | --- | --- | --- |
| stock leash | 24/24, energy mean 201, median 215, p90 310, time to hold median 7.4 s | 19/24, mean 221 | 6/24 |
| dodge | 24/24, mean 145, median 130, p90 233, median 18.4 s | 20/24, mean 149 | 12/24 |

Energy is what the guide spent from the start to the hold, including the hook and the endgame,
which are the same code in both rows. Per approach tick the dodge costs about half; it takes about
2.5 times longer to arrive. At 200 energy both mostly starve mid-lead; the dodge's tows of 20-40 s
cost 6 per second alive and walking. Logs and per-case JSON: `results/trapper/arena-dodge/`.

## Games

| Variant | Games | Mean score | Extinct | Mean predator kills | Held predator-time | Deliveries started | Delivered | Guides lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| society only | 8 | 614.5 | 1 | 10.9 | 0.000 | - | - | - |
| stock trapper (leash) | 8 | 618.5 | 0 | 7.9 | 0.134 | 39 | 14 | 7 |
| dodge guide, stock gate 330 | 8 | 613.7 | 1 | 9.8 | 0.174 | 39 | 13 | 5 |
| dodge guide, gate 160 | 8 | 619.3 | 0 | 7.2 | 0.175 | 41 | 18 | 3 |

Per seed (score; held fraction; delivered/started):

| Seed | society | stock | dodge | dodge gate 160 |
|---:|---|---|---|---|
| 1 | 593.5 | 622.6; 0.27; 2/4 | 625.8; 0.22; 1/6 | 624.2; 0.27; 3/6 |
| 2 | 620.0 | 620.4; 0.30; 3/6 | 604.3; 0.18; 3/5 | 617.2; 0.29; 5/10 |
| 3 | 627.4 | 617.8; 0.11; 3/8 | 625.5; 0.22; 3/10 | 619.0; 0.16; 5/10 |
| 4 | 602.4 | 634.5; 0.00; 0/0 | 625.2; 0.00; 0/0 | 627.5; 0.00; 0/0 |
| 5 | 614.8 | 626.3; 0.03; 0/5 | 620.7; 0.16; 2/5 | 628.2; 0.08; 2/5 |
| 6 | 626.5 | 611.6; 0.00; 0/2 | 625.6; 0.13; 1/4 | 626.6; 0.00; 0/1 |
| 7 | 618.0 | 603.9; 0.21; 3/10 | 606.2; 0.11; 1/4 | 613.0; 0.26; 1/5 |
| 8 | 613.0 | 610.9; 0.15; 3/4 | 576.4; 0.37; 2/5 | 598.7; 0.34; 2/4 |

One 600-second game takes 150-500 s of wall time; the society is not seed-reproducible (the engine iterates Python sets), so single seeds are noise and even the eight-seed means are only indicative. All 32 games are oracle-world, unrecorded batch runs (`results/trapper/*-dodge-v1-*.json`, merged in `results/trapper/batches/dodge-v1-merged.json`); one recorded dodge game (seed 2, gate 160) is in Survival Lab as `game-trapper-seed2-demo-dodge-lowgate-*`. The first batch attempt aborted after 23 games because the stock `lure.py` was being rewritten by another session at that moment and the import-time patch check refused the transient file; the five missing games were rerun afterwards.

## Limits

- Oracle world only: exact positions and predator energy (through `pred_next_speed`), like the
  stock leash's development runs. The margin of 16 against a kill radius of 15 leaves nothing for
  estimator noise.
- Slower deliveries mean longer exposure to the rest of the map; guide losses in games come from
  pincers and crowded mouths, as for the stock leash.
- `_leash` and `_assign` are patched copies frozen against today's `lure.py`/`manager.py`;
  a change there raises at import.

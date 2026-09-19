# Per-object expert systems

One deterministic expert per object class, built only from the training-split
sprite bank. Experts generate candidates; they never rule candidates out on
appearance. Only physical gates remain (size at fixed altitude, visibility);
every other measure is recorded on the candidate for the verifier trained later. Every branch reports its own score so they can be
compared: size gate, proposer, masked correlation, colour, rim or competitor
margin, edge handling, and a SIFT comparison branch that is never fused.

## Bank

```sh
python -m drone.experts.build_bank --output data/drone/expert-bank-20260918-v1
```

Source: `data/drone/sprite-bank-reviewed-20260918` (Oscar-reviewed outlines),
cut from `grid-comparison-20260918-v1/256`, splits from `grid384-20260918-v1`.
Reference frames are training data by contract; validation sprites must come
from training-split tracks; tracks and frames in `label_exclusions.json` are
dropped. Current build: 67 sprites, 15 classes (no helicopter), 57 reference
and 10 validation views. Sprites are BGRA cutouts; the expert composites each
over its own mean foreground colour, so no source background enters any feature.

## Hangar

`drone/experts/hangar.py`. Dark connected components, size and aspect gates,
heading from the minimum-area rectangle, masked gray/high-pass correlation
against the posed sprite on visible pixels only, interior chroma, rim contrast.
Edge-cut components get a coarse heading sweep and a slide along the cut axis.
Output box is the posed mask extent plus the organiser offset from the bank.

Status 18 Sep 2026: unit tests pass on synthetic placements of the training
sprites (rotation, level-1 scale, half-visible at the top edge, rim rejection,
size rejection). Smoke run on the training hangar's own reference tiles only.
**No dev-split evaluation has been run; Oscar decides when.**

```sh
python -m unittest drone.experts.test_hangar -v
python -m drone.experts.evaluate --class hangar --split dev --output artifacts/<new-dir>
```

The evaluator scores complete and edge-cut targets separately per branch,
counts proposals, and measures false alarms on verified-empty training tiles.
Unmatched proposals on positive crops are listed, not counted, because the
validation labels are partial.

## Condor

`drone/experts/condor.py` with the X part model in `parts.py` and the shared
heading-sweep proposer in `common.py`. Branches: proposer, part model over an
affine pose family (heading free over 360 degrees, scale, one-axis stretch,
shear), low-chroma check, size-applicable competitor margin, full-pixel
correlation and SIFT kept separate. Floors are recall-first; a later verifier
prunes. Explain mode returns every candidate with every branch score.

Status 18 Sep 2026, training tiles only (`artifacts/drone-experts-condor-smoke-20260918-v3`):
part model 41/42 with 77 proposals; pixel branch 12/42; SIFT 1/42 with 276
proposals (measured before the gates were relaxed). All 13 expert tests pass. No dev-split run, no empty-tile run, never the organiser API.

## Overnight 19 Sep: all classes, verifier, live detector

- `generic.py` + `specs.py`: one shared expert skeleton (heading-sweep correlation proposer,
  optional colour-blob proposer, fine masked correlation, colour agreement, competitor
  correlation, size gate only) with per-class distinctive measurements recorded on every
  candidate: red markings (small plane), green panel rectangle (jammer), square roof and pale
  slab (small tower), rotor lines through a hub (helicopter), camo texture and barrel lines
  (vehicles), dark/bright fractions. `registry.py` builds any class.
- `add_helicopter.py`: the bank has no reviewed helicopter outline; this adds the v4 library
  sprite from reference frame 0 with `review_status=claude-auto`. Replace when reviewed.
- `crops.py`, `train_verifier.py`, `verifier.py`: harvest candidate crops from training-tile
  runs (positives by label overlap, background from empty tiles), fine-tune a pretrained
  ResNet-18 for presence + class, and score candidates at runtime.
- `live_detector.py`: `DRONE_DETECTOR=drone.experts.live_detector:factory` for the team
  endpoint; all experts in a thread pool, verifier optional, organizer-style boxes.
- `report_all.py`: one table over all class runs.

Compute runs on Runpod pod `ypuawkayl3px8t` under `/workspace/experts`; helper scripts and
logs are in `artifacts/drone-experts-overnight-20260919/`. `drone/portal.py` refuses any
route that is not status, verify or validate.

## Status, 19 Sep 2026 09:30 UTC (expert session)

- Best local proxy: sides-only L1 sweep (`DRONE_L1_WAYPOINTS=2`), caps 12, gates v9, verifier v1, 95 sprites: 0.334
  local, 0.246 on the validation API. Team best remains run O (0.511). Oscar's bar is 0.9.
- The organizer delivers frames on a fixed 333 ms clock and skips frames while a request is open: 60-75% of frames
  arrive through the Runpod HTTP proxy, 249/249 through a cloudflared quick tunnel started on the pod. Serve every
  endpoint through a pod-side tunnel.
- The local validation labels are incomplete pseudo-labels: replaying them as answers scores 0.329 on the API.
  Local proxy scores are relative only. Single-class API replays: `artifacts/drone-experts-overnight-20260919/`
  (`single_class_runs.sh`, `single-class-runs.log`).
- Per-object diagnosis on real frames (`eval_real.py`) and synthetic composites (`eval_synthetic.py`): the four
  zero classes (large_tower, small_tower, mine_roller, medium_launcher) and helicopter fail on appearance/pose of
  single-instance training sprites; box convention, candidate cap, scale sweep, mirrored bank, gate removal, heading
  step 30 and delivered-pixel proposers do not fix them. Next step is a data decision (validation-scene sprites or a
  learned detector), see `artifacts/drone-experts-overnight-20260919/PER-OBJECT.md` and `RESULTS.md`.
- API runs are paused (Oscar) unless a change can plausibly score above 0.6. All pods stopped (disks kept).

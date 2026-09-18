# Live drone endpoint

`api.py` serves the organizer protocol. `example.py` now runs the real
pipeline: a detector on the delivered view, the perspective tracker in
`tracking/`, and the upper-band camera sweep. Every response covers the whole
source frame, including objects the camera is not looking at.

## What happens per sequence

1. Frames 0 and 1 are full-frame overviews. Their pixel motion calibrates the
   scene geometry; their detections seed the first tracks.
2. The camera then cycles **L1 left, L0, L1 right, L0** across the upper band
   (`DRONE_VERTICAL_FRACTION=0`). Objects enter at the top, are seen at L1 on
   one side, and are refreshed on every overview.
3. Each complete detection becomes a track that is forecast on every frame
   until the object leaves the source frame. Seen again whole, it is refreshed.
   An image-based motion clock recognizes frozen and doubled capture steps.
4. Box placement follows the organizer convention: extents blended with a
   per-class prior, edge-cut detections reported for the current frame,
   objects entering at the top forecast from their visible part, clipping at
   pixel index 3839/2159. See the box placement report in the preparation
   project (`docs/drone-box-placement.md`).

The camera moves at most once per frame and every move is derived from the
constraints in the request, so commands are never refused.

## Detector contract

`detectors.py` documents the callable. In short: boxes in delivered-image
pixels, the 16 competition class names, confidences in [0, 1], organizer-style
extents (the asset footprint, not the tight silhouette), and edge-touching
boxes kept. `DRONE_DETECTOR=package.module:factory` plugs in any detector;
`DRONE_DETECTOR=ultralytics` with `DRONE_WEIGHTS` loads a local checkpoint.

## Run

```sh
pip install -r requirements.txt            # transport, tracker, scorer
pip install -r requirements-detector.txt   # plus the matching torch build
DRONE_WEIGHTS=/path/to/last.pt DRONE_DEVICE=cuda:0 python api.py
```

Latency decides recall: the evaluator emits a frame every 333 ms and only
ever sends the newest one, and a frame we never receive scores nothing. Keep
detector plus tracking well under 333 ms on the deployment machine. The
tracker itself takes about 10 ms per frame after a one-off calibration of
about 200 ms. `DRONE_DETECT_EVERY=2` halves detector cost if needed; the
tracker fills the gaps.

## Test locally

```sh
python run_local_eval.py --detector oracle                 # plumbing and policy: organizer boxes as detections
python run_local_eval.py --weights /path/to/last.pt --device cuda:0
python run_local_eval.py --weights /path/to/last.pt --device cuda:0 --realtime   # with the 3 fps clock
python -m unittest tracking.test_tracker tracking.test_placement tracking.test_revisit
```

`run_local_eval.py` starts the server on port 9153, runs `local_evaluator.py`
against it and stops the server. Per-frame diagnostics land in `logs/`. The
oracle detector reads the local scene's annotations and only exists to check
the transport, camera policy and tracker; it cannot run against the
evaluation service.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DRONE_DETECTOR` | `ultralytics` if `DRONE_WEIGHTS` else `none` | `ultralytics`, `none`, `oracle`, or `module:factory` |
| `DRONE_WEIGHTS`, `DRONE_DEVICE`, `DRONE_IMGSZ`, `DRONE_CONF` | –, `cpu`, `960`, `0.25` | Ultralytics checkpoint and inference settings |
| `DRONE_DETECT_EVERY` | `1` | Run the detector on every k-th frame |
| `DRONE_EXTENT_POLICY` | `blend` | `detector`, `blend` or `prior` extents |
| `DRONE_EMIT_PARTIALS`, `DRONE_ENTRY_TRACKS`, `DRONE_CLIP_LAST_INDEX` | `1` | Placement rules |
| `DRONE_BIRTH_CONFIDENCE`, `DRONE_UPDATE_CONFIDENCE` | `0.6`, `0.4` | Track creation and refresh thresholds |
| `DRONE_VERTICAL_FRACTION` | `0` | Band of the L1 crops, 0 is the top |
| `DRONE_OVERVIEW_BETWEEN_SIDES` | `1` | L0 between the L1 sides; `0` uses L1 centre instead |
| `DRONE_OBSERVE_MOTION` | `1` | Image-based motion clock |
| `DRONE_LOG_DIR` | unset | Per-sequence JSONL diagnostics |
| `DRONE_HOST`, `DRONE_PORT` | `0.0.0.0`, `9053` | Server binding |

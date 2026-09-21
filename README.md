# Drone Flyby: the evaluation endpoint

Team **Powered by Smørrebrød**, Technical University of Denmark. Nordic AI Cup 2026, use case `drone-flyby`.

This branch contains the `/predict` endpoint that served our evaluation attempt on 20 September 2026, the model
weights it loaded, and the scripts the models were trained with. No hosted model or API is in the request path.

## Run it

    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    ./run_endpoint.sh                        # port 19130, needs a CUDA GPU (served on one NVIDIA L40S, ~130 ms per frame)
    curl -s http://localhost:19130/api       # {"checkpoint":"02","ready":true}

The endpoint is then at `http://<host>:19130/predict` and follows the organisers' template (`endpoint/dtos.py`).
`DEVICE=cpu ./run_endpoint.sh` also works, but is far slower than the 333 ms frame interval. `run_endpoint.sh`
holds the configuration (environment variables); `endpoint/serve_candidate.py` holds the remaining defaults.
Use one process per run: tracks are kept per `sequence_id`.

`weights/both_m1280.pt` is not used for prediction; `api_checkpoint02.py` checks its hash at start-up, so it has to be present.

## How it works

The organisers send one view per frame (a crop of the 3840 x 2160 source frame at level L0, L1 or L2) and score
boxes for the whole source frame, so the endpoint remembers what it is not looking at.

1. **Calibrate** (`tracking/motion.py`, `fast_motion.py`): the first two full-frame views give the scene motion, one
   time-parametrised homography for the ground under the drone.
2. **Clock** (`tracking/workflow.py`): dropped or repeated frames are recognised from the overlap between consecutive
   views, so the motion step stays right.
3. **Camera**: after calibration the camera stays at L1 and sweeps left, centre, right across the upper band of the
   frame, where objects enter. L2 zoom looks exist in the code and are switched off (`DRONE_REVISIT_EVERY=0`).
4. **Detect** (`sweep-ens/ensemble_detector.py`): three YOLO models see every view and their boxes are pooled with
   class-wise NMS. Two YOLO26m models trained on 256 px tiles (the view is upsampled x4 at L0, x2 at L1) and one
   YOLO26l trained on 1280 px full views.
5. **Place** (`tracking/placement.py`): boxes are put in the organisers' convention (integer, clipped at the frame edge,
   objects cut by the view edge completed); two launcher classes blend the detector's size with a per-class size prior.
6. **Track** (`tracking/revisit.py`): tracks live in source-frame coordinates and are moved by the motion model every
   frame; a detection of 0.4 or more starts a track, a whole sighting of 0.3 or more refreshes it, repeated silent looks
   at its predicted place retire it.
7. **Answer** every live track for the whole frame, whether or not the current view shows it.

## Models

| file | model | training |
| --- | --- | --- |
| `weights/T1-E-SV-s7.pt` | YOLO26m, 256 px tiles | `training/launch_final2.sh T1-E-SV-s7 7 E1` with `SCALE=0.6` |
| `weights/T2-E-s7.pt` | YOLO26m, 256 px tiles | `training/launch_final2.sh T2-E-s7 7 E2` |
| `weights/fly36-l.pt` | YOLO26l, 1280 px views | `training/run_views.sh fly36-l yolo26l.pt ...`, 36,000 generated views, 5 epochs |

All three start from the public YOLO26 checkpoints. The training images are synthetic: object sprites cut from the
organisers' frames, rendered or pasted onto terrain and background imagery and cut into tiles or full views; the two
tile models' mix also contains tiles cut from the validation frames. `launch_final2.sh` builds the dataset
(`build_ds_final4.py`) and trains (`explore_final.sh`: AdamW, 60-epoch schedule, the epoch-40 checkpoint is the model).
The generated datasets are large and not in this branch; they are available on request.

## Files

| path | what it is |
| --- | --- |
| `endpoint/` | the server (`api_checkpoint02.py`, `serve_candidate.py`, `example.py`, `detectors.py`), the tracker (`tracking/`), the detector (`sweep-ens/`), the organisers' template files |
| `weights/` | the model files |
| `config/classpolicy-ml-sl.json` | the per-class box-size setting |
| `run_endpoint.sh`, `requirements.txt` | start the endpoint from a checkout |
| `training/` | the training scripts |

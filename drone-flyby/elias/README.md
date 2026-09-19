# elias/: a detector that generalises, and the pipeline settings that go with it

Everything in this folder was built on the night of 18 to 19 September 2026. The full account, with every number
and how it was measured, is `../research/02-night-report.md`. This file is the operating manual.

## Serve it

Oscar's endpoint is unchanged apart from three optional settings that default to his behaviour. Weights are in
`elias/release/` (Git LFS).

```sh
pip install -r requirements.txt -r requirements-detector.txt
export DRONE_DETECTOR=ultralytics DRONE_WEIGHTS=elias/release/both_m1280.pt DRONE_DEVICE=cuda:0 DRONE_IMGSZ=1280
export DRONE_OVERVIEW_BETWEEN_SIDES=0     # L1 left, centre, right, centre; no L0 overviews
export DRONE_MISS_RULE=seen               # tracking/revisit.py: a miss counts only if the view gave the object enough pixels
export DRONE_CONF=0.05 DRONE_BIRTH_CONFIDENCE=0.25 DRONE_UPDATE_CONFIDENCE=0.15
python api.py                              # port 9053, submit http(s)://<host>/predict
```

Latency: about 85 ms per frame on an RTX 5070 laptop GPU when nothing else uses it (the portal then delivers all
249 frames), about 200 ms when the GPU is shared (10 to 13 frames lost). Use a machine that does nothing else.

## Check it before trusting it

```sh
bash elias/run_harness.sh elias/release/both_m1280.pt mytag --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15 --imgsz 1280
python elias/portal.py status                                  # the team's validation history (read-only)
IMGSZ=1280 bash elias/serve_portal.sh <weights> mytag validate "0:83"    # ONE concealed portal validation run
```

`serve_portal.sh` starts the server and a Cloudflare quick tunnel, waits until the URL answers from outside, warms
the model, queues one validation run and prints its score. The fourth argument is `DRONE_ANSWER_WINDOWS`: the
pipeline runs over the whole sequence but only emits answers for those frame indices, so the public board does
not move. Disjoint windows add up to the full score within about 0.015 (three thirds: `0:83`, `83:166`,
`166:100000`). `portal.py` refuses to queue while another validation of the team is running, and it knows nothing
about the one-shot final attempt.

The local harness scores against the team's pseudo-labels. For a model trained on validation-derived data that is
circular (it said 0.81 where the portal said 0.50). Trust it for unseen-scene models only.

## Rebuild the model

```sh
python elias/data/sprites.py ...                # cut sprites (GrabCut) into elias/sprites; see its docstring
python elias/data/synth_yolo.py --out DATA --n 12000 --sprite-scenes helsinki validation \
    --background-scenes helsinki validation --real-val-scenes helsinki validation \
    --extra-bg <UCMerced_LandUse/Images> --extra-bg-prob 0.4
yolo detect train model=yolo26m.pt data=DATA/data.yaml imgsz=1280 epochs=5 batch=16 \
    fliplr=0.5 flipud=0 degrees=0 scale=0.05 translate=0.05 mosaic=0.5 close_mosaic=5 hsv_h=0.01 hsv_s=0.4 hsv_v=0.3
```

`elias/pod_final.sh` does exactly this on a RunPod GPU in about 25 minutes. What matters, in order of measured
effect on a scene the model never saw: a medium backbone (0.316 to 0.423), training at 1280 (to 0.528), extra
aerial backgrounds (+0.055 on the small model), wide and INDEPENDENT object and terrain exposure (+0.04),
rotation that respects the lean of tall objects (+0.06), hard-edged pasting (soft blending costs 0.05), and a
short schedule (unseen-scene accuracy peaks after 40 to 60 thousand samples and then declines).

## Files

| file | what it is |
|---|---|
| `SPEC.md` | the 64x64 scale-preserving window and the dataset container used by the classifier experiments |
| `data/real_crops.py`, `data/synth.py`, `data/sprites.py`, `data/tilt_study.py` | real windows, synthetic windows, sprite cutting, the camera-tilt study |
| `data/synth_yolo.py` | synthetic FULL VIEWS in YOLO format: the training data of the deployed model |
| `data/validation_exclude.json`, `data/validation_hidden.json` | pseudo-label errors and unlabelled real objects that must never enter training |
| `train.py` | window classifier and its ablations (scale input, experts, test-time averaging) |
| `detector.py`, `views.py` | the window classifier as a sliding detector (it does not work: 45 false positives per view), view-level evaluation, verifier re-ranking |
| `ensemble.py` | several checkpoints behind the detector hook (`DRONE_DETECTOR=elias.ensemble:build`) |
| `policy_sim.py` | camera-policy simulator calibrated to the real harness (mean error 0.019) |
| `find_unlabelled.py` | how the unlabelled small planes in the validation flight were found |
| `run_harness.sh`, `serve_portal.sh`, `portal.py` | local end-to-end run, concealed portal validation, portal status |
| `pod_*.sh` | what ran on the RunPod GPUs |

#!/usr/bin/env python3
"""Replay the checkpoint-02 pipeline over the reconstructed validation flight, with or without the verifier.

Runs on the pod, from a copy of experiments/pipeline300-20260919/checkpoint-02/endpoint that also holds
verifier_hook.py and train_verifier.py. Settings are checkpoint 02's serve_candidate defaults; the detector hook,
weights and log dir come from the arguments. Frames are rendered causally like local_evaluator does (only the
requested view reaches the endpoint). Per-frame diagnostics go to <out>/<sequence>.jsonl through DRONE_LOG_DIR.

    python3 replay_compare.py --endpoint /workspace/verifier/cp02/endpoint --weights /workspace/verifier/cp02/both_m1280.pt \
        --frames /root/data/reconstructed-validation --out /workspace/verifier/replays/baseline
    python3 replay_compare.py ... --verifier /workspace/verifier/runs/m1-convnext/model.pt,/workspace/verifier/runs/m2-smallcnn/model.pt \
        --gate 0.5 --out /workspace/verifier/replays/verified
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--verifier', default='', help='comma list of verifier model.pt files; empty = baseline')
    ap.add_argument('--gate', type=float, default=0.0, help='p(object) gate for the four small classes (0 = re-rank only)')
    ap.add_argument('--floor', type=float, default=0.1)
    ap.add_argument('--first', type=int, default=5)
    ap.add_argument('--last', type=int, default=249)
    ap.add_argument('--extra', default='', help='JSON of extra DRONE_* overrides')
    ap.add_argument('--gate-json', default='', help='per-class p(object) gates, overrides --gate')
    ap.add_argument('--gate-padded', type=float, default=0.8)
    ap.add_argument('--save-captures', action='store_true', help='write the delivered views to <out>/captures/<seq>-<frame>.png')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    endpoint = Path(a.endpoint).resolve()
    sys.path.insert(0, str(endpoint))
    os.chdir(endpoint)
    settings = dict(DRONE_DETECTOR='ultralytics', DRONE_WEIGHTS=a.weights, DRONE_FAST_CALIBRATION='1', DRONE_DEVICE='cuda:0',
                    DRONE_IMGSZ='1280', DRONE_HALF='1', DRONE_CONF='0.05', DRONE_BIRTH_CONFIDENCE='0.25', DRONE_UPDATE_CONFIDENCE='0.15',
                    DRONE_CV_THREADS='4', DRONE_CONFIDENCE_MEMORY='1', DRONE_CONFIDENCE_MODE='peak', DRONE_CONFIDENCE_DECAY='0.75',
                    DRONE_STRONG_CONFIDENCE='0.5', DRONE_OVERVIEW_BETWEEN_SIDES='0', DRONE_MISS_RULE='seen', DRONE_REVISIT_EVERY='8',
                    DRONE_EXTENT_POLICY='blend', DRONE_EMIT_PARTIALS='1', DRONE_ENTRY_TRACKS='1', DRONE_LOG_DIR=str(out),
                    DRONE_ANSWER_WINDOWS='', DRONE_ANSWER_CLASSES='')
    if a.verifier:
        settings.update(DRONE_DETECTOR='verifier_hook:build', VERIFIER_BASE='ultralytics', VERIFIER_MODELS=a.verifier,
                        VERIFIER_GATE=a.gate_json or json.dumps({c: a.gate for c in ('small_launcher', 'medium_launcher', 'ta-ta', 'jammer')}),
                        VERIFIER_GATE_FLOOR=str(a.floor), VERIFIER_GATE_PADDED=str(a.gate_padded), VERIFIER_TTA='1')
    if a.extra:
        settings.update(json.loads(a.extra))
    os.environ.update(settings)
    (out / 'settings.json').write_text(json.dumps(settings, indent=1))

    import cv2
    from local_evaluator import Camera, build_request, render_view
    from dtos import DroneFlybyPredictRequestDto
    import example
    camera = Camera(); started = time.time(); seq = out.name
    for frame in range(a.first, a.last + 1):
        im = cv2.imread(str(Path(a.frames) / f'frame_{frame:06d}.png'))
        assert im is not None, frame
        encoded = render_view(im, camera)
        if a.save_captures:
            (out / 'captures').mkdir(exist_ok=True)
            (out / 'captures' / f'{seq}-{frame:06d}.png').write_bytes(base64.b64decode(encoded))
        req = build_request(frame, frame - 1, camera, encoded, None)
        req['sequence_id'] = seq; req['request_id'] = f'{seq}-{frame}'
        response = example.predict(DroneFlybyPredictRequestDto.model_validate(req))
        if response.requested_view:
            camera.apply(**response.requested_view.model_dump())
        if frame % 25 == 0:
            print(f'frame {frame}/{a.last}, elapsed {time.time()-started:.0f}s', flush=True)
    det = getattr(example, 'DETECTOR', None) or getattr(example, 'detector', None)
    stats = getattr(det, 'stats', None)
    (out / 'complete.json').write_text(json.dumps({'frames': a.last - a.first + 1, 'elapsed_seconds': time.time() - started,
                                                   'verifier': a.verifier, 'gate': a.gate, 'stats': stats}, indent=1))
    print('completed', json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()

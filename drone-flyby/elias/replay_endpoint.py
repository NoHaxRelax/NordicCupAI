"""A /predict endpoint that replays FIXED boxes per frame index and never moves the camera.

Used to ask the validation portal one question about a set of candidate boxes: emitted under several class
names at once, the score is nonzero only if the boxes hit a class that is present in the ground truth (boxes of
absent classes are ignored by the scorer, boxes of present classes that miss are cheap false positives). No
detector, no GPU, no camera command; the portal keeps sending L0 overviews.

    DRONE_REPLAY=elias/out/replay_candidates.json DRONE_PORT=9053 python elias/replay_endpoint.py

The JSON maps frame index (string) to a list of {"object_id", "bbox": [x1, y1, x2, y2] in source px, "confidence"}.
VALIDATION only: this file knows the validation queue through elias/portal.py and nothing else.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto   # noqa: E402

PLAN = {int(k): v for k, v in json.loads(Path(os.environ['DRONE_REPLAY']).read_text()).items()}
app = FastAPI()
SEEN = []


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict(request: DroneFlybyPredictRequestDto):
    W, H = request.original_width, request.original_height
    rows = []
    for r in PLAN.get(request.frame_index, [])[:500]:
        x1, y1, x2, y2 = r['bbox']
        rows.append({'object_id': r['object_id'], 'confidence': float(r['confidence']),
                     'bbox': [max(0., x1/W), max(0., y1/H), min(1., x2/W), min(1., y2/H)]})
    SEEN.append(request.frame_index)
    return {'request_id': request.request_id, 'frame': request.frame, 'annotations': rows, 'requested_view': None}


@app.get('/api')
def hello():
    return {'service': 'drone-flyby-replay', 'frames_seen': len(SEEN), 'plan_frames': len(PLAN)}


@app.get('/')
def index():
    return 'replay endpoint running'


if __name__ == '__main__':
    uvicorn.run(app, host=os.environ.get('DRONE_HOST', '0.0.0.0'), port=int(os.environ.get('DRONE_PORT', '9053')))

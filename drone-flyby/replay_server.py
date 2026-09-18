"""Serve the per-frame responses of a recorded endpoint run, instantly.

An offline replay is deterministic: for every frame the diagnostics log holds
the annotations and the camera command the endpoint produced. Serving those
back verbatim gives the evaluator exactly the same view sequence with no
latency, so a recorded configuration can be scored against the organizer's
validation service without the 3333 ms budget costing frames.

    python replay_server.py logs/O-l2top-blend-rev3 --port 9310
    python local_evaluator.py --url http://127.0.0.1:9310/predict --scene validation

Frames without a recorded response answer with no annotations and no camera
command. Only the recorded sequence's frame numbers are meaningful; do not
serve a recording of one scene to another.
"""
import argparse
import glob
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from dtos import (DroneFlybyPredictionDto, DroneFlybyPredictRequestDto,
                  DroneFlybyPredictResponseDto, RequestedViewDto)
from utils import validate_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_recording(directory):
    rows = {}
    for path in sorted(glob.glob(str(Path(directory)/'*.jsonl'))):
        for line in open(path):
            row = json.loads(line)
            rows[int(row['frame'])] = row
    if not rows:
        raise SystemExit(f'No diagnostics rows under {directory}')
    return rows


def build_app(recording):
    app = FastAPI()

    @app.post('/predict', response_model=DroneFlybyPredictResponseDto)
    def predict(request: DroneFlybyPredictRequestDto):
        row = recording.get(request.frame)
        annotations, requested = [], None
        if row is not None:
            annotations = [DroneFlybyPredictionDto(object_id=a['object_id'], bbox=list(a['bbox']), confidence=float(a['confidence']))
                           for a in row.get('response') or []]
            view = row.get('requested_view')
            if view:
                requested = RequestedViewDto(**view)
        response = DroneFlybyPredictResponseDto(request_id=request.request_id, frame=request.frame,
                                                annotations=annotations, requested_view=requested)
        validate_response(response)
        logger.info('frame %s: %d recorded annotations, camera %s', request.frame, len(annotations), view)
        return response

    @app.get('/')
    def index():
        return 'Replay endpoint is running!'

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', help='Directory with the run diagnostics JSONL (DRONE_LOG_DIR of that run)')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9310)
    args = parser.parse_args()
    recording = load_recording(args.recording)
    logger.info('Loaded %d recorded frames from %s', len(recording), args.recording)
    uvicorn.run(build_app(recording), host=args.host, port=args.port)


if __name__ == '__main__':
    main()

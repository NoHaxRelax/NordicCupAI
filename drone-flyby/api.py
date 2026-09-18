"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import os
import time

import uvicorn
from fastapi import FastAPI

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from example import predict
from utils import validate_response

HOST = os.environ.get('DRONE_HOST', '0.0.0.0')
PORT = int(os.environ.get('DRONE_PORT', '9053'))
ROUTE = os.environ.get('DRONE_ROUTE', '').strip('/')  # optional secret path prefix for a public tunnel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if ROUTE:
    # The evaluation service uses the submitted URL verbatim, so serve the
    # same handlers under the prefix as well.
    app.post(f'/{ROUTE}/predict', response_model=DroneFlybyPredictResponseDto)(predict_endpoint)
    app.get(f'/{ROUTE}/api')(hello)
    app.get(f'/{ROUTE}/')(index)


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)

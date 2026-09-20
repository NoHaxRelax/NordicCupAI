"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9054/predict``
rather than just the host.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

# Logging first: example.py runs the model warm-up at import and its lines are lost otherwise.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto  # noqa: E402
from example import predict  # noqa: E402
from utils import validate_response  # noqa: E402

HOST = '0.0.0.0'
PORT = 9054

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    """Answer every question about one conversation."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently score every
    # question about this conversation wrong.
    validate_response(response, expected_count=len(request.questions))

    return response


@app.get('/api')
def hello():
    import example
    import model
    return {
        'service': 'medical-appointment-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        # live configuration and counters, read by the pre-flight (research/committee-2026-09-17/02)
        'predict_deadline_s': example.PREDICT_DEADLINE,
        'timed_out_conversations': example.timed_out,
        'model': model.status(),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)

"""Hosted adapter for frozen fast1m at8721a3d; append-only timing evidence."""
import json
import time
import random
import os
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, Body, Request
from telemetry import Telemetry
from fastapi.responses import JSONResponse
from night_policy import NightPolicy
from harvest import Harvester

CONFIG = json.loads(Path(__file__).with_name('pred_best.json').read_text())
PARAMS = dict(enabled=True, budget=1000000, max_harvests=100,
              sacrifice_mode='predict_contact', cooldown=0, contact_margin=1.0)
LOG = Path(os.environ.get('FAST1M_LOG', '/var/log/nordiccup-fast1m-endpoint.jsonl'))
RANDOM_TAIL = False
rng = random.Random(20260920)
policy = NightPolicy(CONFIG)
harvester = Harvester(**PARAMS)
lock = Lock()
api = FastAPI()
state = dict(strategy='fast1m', source_commit='8721a3d', requests=0, game=0,
             sim_time=None, score=None, agents=None, burst_attempts=0,
             confirmed_transfers=0, last_callback_unix=None,
             last_burst=None, previous_callback_gap_ms=None)

telemetry = Telemetry(LOG)

def write(row):
    telemetry.emit(row)

@api.get('/health')
@api.get('/seed-live-mode144/status')
@api.get('/seed-live-eval-20260920/status')
@api.get('/')
def health():
    return dict(state, parameters=dict(PARAMS, min_free=harvester.min_free), random_tail=RANDOM_TAIL, logging=telemetry.status())

@api.post('/seed-live-mode144/predict')
@api.post('/seed-live-eval-20260920/predict')
@api.post('/predict')
def predict(request: Request, step: dict = Body(...)):
    global policy, harvester
    with lock:
        started = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        t = float(step.get('sim_time', 0))
        states = step.get('agent_status', [])
        if state['sim_time'] is None or t < state['sim_time']:
            policy = NightPolicy(CONFIG)
            harvester = Harvester(**PARAMS)
            state['game'] += 1
        state.update(requests=state['requests']+1, sim_time=t, score=step.get('score'),
                     agents=len(states), last_callback_unix=time.time())
        before = harvester.harvests
        actions = ([{'agent_id': int(s['agent_id']),
                     'move_distance': rng.uniform(0, float(s.get('sprint_speed', 10))),
                     'move_direction': rng.uniform(-3.141592653589793, 3.141592653589793),
                     'turn_angle': rng.uniform(-3.141592653589793, 3.141592653589793),
                     'spawn_agent': False} for s in states] if RANDOM_TAIL
                   else policy(states, t) if states else [])
        policy_end = time.perf_counter_ns()
        if not RANDOM_TAIL and states and step.get('game_status') != 'game_over':
            actions = harvester.apply(states, t, actions, score=step.get('score'))
        harvest_end = time.perf_counter_ns()
        burst = harvester.log[-1] if harvester.harvests > before else None
        if burst:
            state['last_burst'] = burst
        cp = harvester.contact_predictor
        state.update(burst_attempts=harvester.harvests,
                     confirmed_transfers=cp.confirmed if cp else 0)
        response = JSONResponse({'actions': actions})
        ended = time.perf_counter_ns()
        write(dict(event='response_ready', request_id=request.state.log_id, seq=state['requests'], game=state['game'],
                   unix=time.time(), sim_time=t, score=state['score'], agents=len(states),
                   status=step.get('game_status'), burst=burst,
                   burst_attempts=harvester.harvests,
                   confirmed_transfers=state['confirmed_transfers'],
                   actions=len(actions), bytes=len(response.body),
                   policy_ms=(policy_end-started)/1e6,
                   harvest_ms=(harvest_end-policy_end)/1e6,
                   handler_serialization_ms=(ended-started)/1e6,
                   thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6))
        return response

class Timing:
    def __init__(self, inner):
        self.inner = inner
        self.last_done = None
        self.sequence = 0
        self.instance = str(time.time_ns())

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] not in ('/predict', '/seed-live-mode144/predict', '/seed-live-eval-20260920/predict'):
            return await self.inner(scope, receive, send)
        start = time.perf_counter_ns()
        self.sequence += 1
        request_id = self.instance + ':' + str(self.sequence)
        scope.setdefault('state', {})['log_id'] = request_id
        gap = (start-self.last_done)/1e6 if self.last_done else None
        state['previous_callback_gap_ms'] = gap
        chunks = []
        incoming = outgoing = 0
        status = None
        received_at = first_send = last_send = None
        send_wait = 0
        too_large = complete = disconnected = False
        error = None

        async def observed_receive():
            nonlocal incoming, received_at, too_large, disconnected
            message = await receive()
            if message['type'] == 'http.request':
                data = message.get('body', b'')
                incoming += len(data)
                if incoming > telemetry.max_request_bytes:
                    chunks.clear()
                    too_large = True
                elif not too_large:
                    chunks.append(data)  # Retain existing bytes; no parse/copy here.
                if not message.get('more_body', False):
                    received_at = time.perf_counter_ns()
                    if too_large:
                        telemetry.oversized += 1
                    else:
                        telemetry.emit(dict(event='request_received', request_id=request_id,
                                            unix=time.time(), bytes=incoming), tuple(chunks))
                    chunks.clear()
            elif message['type'] == 'http.disconnect':
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal status, outgoing, first_send, last_send, send_wait, complete
            if message['type'] == 'http.response.start':
                status = message['status']
                first_send = time.perf_counter_ns()
            elif message['type'] == 'http.response.body':
                outgoing += len(message.get('body', b''))  # Never log/copy action payload.
            before = time.perf_counter_ns()
            await send(message)
            after = time.perf_counter_ns()
            send_wait += after-before
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                last_send = after
                complete = True

        try:
            await self.inner(scope, observed_receive, observed_send)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end = time.perf_counter_ns()
            write(dict(event='http_complete', request_id=request_id, unix=time.time(),
                       asgi_ms=(end-start)/1e6, previous_callback_gap_ms=gap,
                       receive_ms=(received_at-start)/1e6 if received_at else None,
                       response_start_ms=(first_send-start)/1e6 if first_send else None,
                       response_sent_ms=(last_send-start)/1e6 if last_send else None,
                       send_wait_ms=send_wait/1e6, request_bytes=incoming,
                       response_bytes=outgoing, http_status=status, completed=complete,
                       disconnected=disconnected, error=error, request_oversized=too_large))
            self.last_done = end

app=Timing(api)

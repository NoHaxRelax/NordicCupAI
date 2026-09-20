"""Hosted adapter for frozen fast1m at8721a3d; append-only timing evidence."""
import json
import time
import random
import os
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, Body
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

def write(row):
    with LOG.open('a') as f:
        f.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')

@api.get('/health')
@api.get('/seed-live-mode144/status')
@api.get('/seed-live-eval-20260920/status')
@api.get('/')
def health():
    return dict(state, parameters=dict(PARAMS, min_free=harvester.min_free), random_tail=RANDOM_TAIL)

@api.post('/seed-live-mode144/predict')
@api.post('/seed-live-eval-20260920/predict')
@api.post('/predict')
def predict(step: dict = Body(...)):
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
        write(dict(event='response_ready', seq=state['requests'], game=state['game'],
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
        self.inner=inner
        self.last_done=None

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] not in ('/predict', '/seed-live-mode144/predict', '/seed-live-eval-20260920/predict'):
            return await self.inner(scope, receive, send)
        start=time.perf_counter_ns()
        gap=(start-self.last_done)/1e6 if self.last_done else None
        state['previous_callback_gap_ms']=gap
        await self.inner(scope, receive, send)
        end=time.perf_counter_ns()
        write(dict(event='http_complete', seq=state['requests'], unix=time.time(),
                   asgi_ms=(end-start)/1e6, previous_callback_gap_ms=gap))
        self.last_done=end

app=Timing(api)

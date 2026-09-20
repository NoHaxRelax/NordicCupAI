"""The frozen fast1m winner, with public-input run-health telemetry."""
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
WINNER=ROOT/'docs/burst-score/local-winner.json'
SOURCE=ROOT/'oscar-overnight-cpp/nightsim/serve'
winner=json.loads(WINNER.read_text())
os.environ['NIGHT_CONFIG']=str(SOURCE/'pred_best.json')
assert json.loads((SOURCE/'pred_best.json').read_text())==winner['policy_config']
os.environ['NIGHT_HARVEST']=json.dumps({k:v for k,v in winner['harvester_config'].items() if k!='enabled'})
for key in ['NIGHT_SCORE_GUARD','NIGHT_SCORE_CEILING','NIGHT_SCORE_RESERVE','NIGHT_TRANSFER_SCORE_MARGIN','NIGHT_NOOP_BURST']:
    os.environ[key]='0'
sys.path.insert(0,str(SOURCE))
import night_agent_server as base
from fastapi import Body,FastAPI

app=FastAPI(title='fast1m burst policy with run health')
base.LATENCY_LOG=Path(os.environ.get('BURST1M_LATENCY_LOG','/tmp/burst1m-latency.jsonl'))
EVENT_LOG=Path(os.environ.get('BURST1M_HEALTH_LOG','/tmp/burst1m-health.jsonl'))
serial=threading.Lock()
health={'strategy':'fast1m','status':'awaiting_game','requests':0,'errors':0,'source_commit':'8721a3d',
        'budget':1000000,'cooldown':0,'hosted_viability':'unverified'}

@app.get('/health')
@app.get('/')
def status():
    current=dict(health)
    current['seconds_since_callback']=time.time()-current['last_callback_unix'] if 'last_callback_unix' in current else None
    return current

@app.post('/predict')
def predict(step:dict=Body(...)):
    global health
    with serial:
        now=time.time();start=time.perf_counter()
        t=step.get('sim_time');score=step.get('score')
        reset=t is not None and health.get('sim_time') is not None and t<health['sim_time']
        prev=None if reset else health.get('score')
        health={**health,'status':'processing','last_callback_unix':now,'sim_time':t,'score':score,
                'score_delta':score-prev if isinstance(score,(int,float)) and isinstance(prev,(int,float)) else None,
                'agents':len(step.get('agent_status',[])),'requests':health['requests']+1}
        try:
            response=base.predict(step)
        except Exception as exc:
            health={**health,'status':'error','errors':health['errors']+1,'last_error':type(exc).__name__}
            raise
        else:
            health={**health,'status':'game_over' if step.get('game_status')=='game_over' else 'active',
                    'response_ms':(time.perf_counter()-start)*1000,'response_bytes':len(response.body),
                    'bursts_sent':base.harvester.harvests,'confirmed_transfers':base.confirmed_transfers}
            EVENT_LOG.parent.mkdir(parents=True,exist_ok=True)
            with EVENT_LOG.open('a') as f:f.write(json.dumps(health)+'\n')
            return response

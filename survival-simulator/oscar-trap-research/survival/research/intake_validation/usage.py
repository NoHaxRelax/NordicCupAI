"""Read only this task's account-limit telemetry; never print session contents."""
from pathlib import Path
import json
import os

SESSIONS=['01a0afb2-b67f-74d3-b2d7-fe4f87193a13','01a0afea-97b6-77b0-b969-7274849a17b5','01a0afea-1d3a-73a1-a78d-c13ab712c26a','01a0afea-5836-7f03-97e8-72b68855e80f','01a0b002-4dab-7311-946b-2c8ebe91d096','01a0b00f-f411-78e0-8fb1-f2bc0df6eb49']
STOP=Path('/tmp/predator-intake-stop')
STOP_REMAINING=60


def status():
    results=[]
    for session in SESSIONS:
        paths=list(Path('/home/Ucals/.codex/sessions').rglob('*'+session+'*'))
        if not paths:continue
        p=paths[0]
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2_000_000))
            lines=f.read().splitlines()
        for line in reversed(lines):
            if b'"rate_limits"' not in line:continue
            try:row=json.loads(line)
            except ValueError:continue
            payload=row.get('payload',{})
            if payload.get('type')!='token_count':continue
            limits=payload.get('rate_limits')
            if not limits:continue
            windows={key:100-value['used_percent'] for key in ['primary','secondary']
                     if (value:=limits.get(key)) and value.get('used_percent') is not None}
            if windows:results.append(dict(available=True,timestamp=row.get('timestamp'),remaining_percent=windows))
            break
    if not results:return {'available':False}
    result=max(results,key=lambda r:r['timestamp'])
    result['stop_remaining']=STOP_REMAINING
    if min(result['remaining_percent'].values())<=STOP_REMAINING:
        STOP.write_text(json.dumps(result)+'\n')
        result['stop']=True
    return result


if __name__=='__main__':print(json.dumps(status()))

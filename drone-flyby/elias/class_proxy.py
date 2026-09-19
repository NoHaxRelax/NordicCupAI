"""A class-filtering proxy in front of ANY team endpoint, so a foreign server can be measured one class at a time.

    PROXY_TARGET=http://host:port PROXY_CLASSES=helicopter PROXY_PORT=9071 python elias/class_proxy.py

Every /predict request is forwarded unchanged (the camera command comes back unchanged too); only annotations of
the listed classes are returned, so the portal score x 13 is that class's AP for the target server. GET / and
GET /api are passed through so readiness checks see the target's use-case name. Nothing is stored.
"""
from __future__ import annotations

import os

import requests
import uvicorn
from fastapi import FastAPI, Request, Response

TARGET = os.environ['PROXY_TARGET'].rstrip('/')
CLASSES = {c.strip() for c in os.environ.get('PROXY_CLASSES', '').split(',') if c.strip()}
app = FastAPI()
session = requests.Session()


@app.get('/')
@app.get('/api')
async def passthrough(request: Request):
    r = session.get(TARGET+request.url.path, timeout=10)
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get('content-type', 'application/json'))


@app.post('/predict')
async def predict(request: Request):
    body = await request.body()
    r = session.post(TARGET+'/predict', data=body, headers={'content-type': 'application/json'}, timeout=3.2)
    if r.status_code != 200:
        return Response(content=r.content, status_code=r.status_code, media_type='application/json')
    out = r.json()
    if CLASSES:
        out['annotations'] = [a for a in out.get('annotations', []) if a.get('object_id') in CLASSES]
    return out


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PROXY_PORT', '9071')), log_level='warning')

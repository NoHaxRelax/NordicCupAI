"""Streaming replay diagnostic; never weakens or replaces the shadow safety gate."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastsim import SimulationCore
from models.seed_shadow.replay import step, public_frame, close
import numpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--journal',type=Path,required=True)
p.add_argument('--candidates',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
candidates=json.loads(a.candidates.read_text())
if len(candidates)!=1:raise SystemExit('Diagnostic requires exactly one search candidate')

def dynamic(body):
    body=copy.deepcopy(body)
    for agent in body['agents']:
        agent['observations']=[o for o in agent['observations'] if o['type']!='Edge']
    return body

def attributes(body):
    return {**body,'agents':[{k:v for k,v in agent.items() if k!='observations'} for agent in body['agents']]}

def difference(x,y,path=''):
    if close(x,y):return None
    if isinstance(x,dict) and isinstance(y,dict) and x.keys()==y.keys():
        for key in x:
            d=difference(x[key],y[key],path+'.'+key)
            if d:return d
    elif isinstance(x,(list,tuple)) and isinstance(y,(list,tuple)):
        if len(x)!=len(y):return dict(path=path,actual_length=len(x),recorded_length=len(y),actual_items=x[:20] if path.endswith('.observations') else None,recorded_items=y[:20] if path.endswith('.observations') else None)
        for i,(left,right) in enumerate(zip(x,y)):
            d=difference(left,right,path+f'[{i}]')
            if d:return d
    return dict(path=path,actual=x,recorded=y)

start=time.perf_counter();sim=SimulationCore(seed=candidates[0])
counts=dict(full_mismatches=0,dynamic_mismatches=0,attribute_mismatches=0)
first={};frames=0
with gzip.open(a.journal,'rt') as f:
    for line in f:
        frame=json.loads(line);frames+=1
        actual=public_frame(step(sim,frame['actions']));recorded=public_frame(frame['public'])
        for name,x,y in [('full',actual,recorded),('dynamic',dynamic(actual),dynamic(recorded)),('attribute',attributes(actual),attributes(recorded))]:
            if not close(x,y):
                counts[name+'_mismatches']+=1
                if name not in first:first[name]=dict(frame=frames,difference=difference(x,y))
result=dict(scope='Diagnostic only; excluding Edge observations is NOT acceptance of a shadow or proof of hidden-state parity',
            seed_candidate=candidates[0],frames=frames,**counts,first=first,
            python=platform.python_version(),numpy=numpy.__version__,seconds=time.perf_counter()-start,
            journal_sha256=hashlib.sha256(a.journal.read_bytes()).hexdigest())
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

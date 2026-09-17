"""Recorded fixture runner for observation-only front tangential intake.

Later guide/predator pairs and their prepared approach positions are supplied by
the fixture.  Recruitment and routing to those starts are not demonstrated.
The controller receives only JSON-round-tripped native observation DTOs and
public time.  Food replenishment is the same disclosed fixture assumption as
the wall-funnel validation.
"""
from __future__ import annotations

import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE.parents[0]/"wall_funneling"/"observed_run.py"
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location("wall_funneling_observed_runner_tangent",SOURCE)
runner=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(runner)
pspec=importlib.util.spec_from_file_location("intake_tangent_policy",HERE/"observed_tangent_policy.py")
policy=importlib.util.module_from_spec(pspec);assert pspec.loader is not None;pspec.loader.exec_module(policy)
POLICY_HASH=hashlib.sha256((HERE/"observed_tangent_policy.py").read_bytes()).hexdigest()
OUT=ROOT/"results"/"intake_guides_sol"
BaseRecorder=runner.ReplayRecorder

class Recorder(BaseRecorder):
    def __init__(self,*args,**kwargs):
        kwargs.update(policy="observation-only-front-tangential-intake-v1",
                      notes=__doc__,policy_sha256=POLICY_HASH)
        super().__init__(*args,**kwargs)

def run(**kwargs):
    runner.OUT=OUT;runner.StagedGuideFunnel=policy.ObservedFunnel
    runner.EXPERIMENTAL_HASH=POLICY_HASH;runner.ReplayRecorder=Recorder
    kwargs["staged_guides"]=True
    kwargs.setdefault("width",30);kwargs.setdefault("length",100)
    kwargs.setdefault("baits",2);kwargs.setdefault("awake",True)
    return runner.run(**kwargs)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predators",type=int,default=4);p.add_argument("--waves",type=int)
    p.add_argument("--interval",type=float,default=90.);p.add_argument("--seconds",type=float,default=400.)
    p.add_argument("--seed",type=int,default=4201);p.add_argument("--spread",type=float,default=15.)
    p.add_argument("--depth-spread",type=float,default=40.);p.add_argument("--horizontal",action="store_true")
    p.add_argument("--native",action="store_true")
    a=vars(p.parse_args());a["waves"]=a["waves"] or a["predators"]
    r=run(**a);print(json.dumps({k:v for k,v in r.items() if k not in ("trace","events")},indent=2))

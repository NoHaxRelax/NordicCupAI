"""Mandatory, immutable per-run recording and result receipts.
No engine/RNG mutations. Every game step must call recorder.capture, including
steps not retained by sampling. Existing recordings and metrics stay untouched.
"""
import hashlib,json,os,re,uuid
from datetime import datetime,timezone
from pathlib import Path
from recorder import ReplayRecorder

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/dedicated_sprinter'

def unique_id(label):
    stem=re.sub(r'[^a-zA-Z0-9_-]+','-',label).strip('-')[:130]
    return f'{stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}'

class RunRecording:
    def __init__(self,env,*,policy,module,seed,scenario,parameters,horizon,label=None,
                 reproduction=None,native=None,every=None):
        version=module.replace('controller','v5') if module=='controller' else module
        title=label or f'Sprinter {scenario} {policy} {version} seed {seed}'
        self.run_id=unique_id(f'{scenario}-{policy}-{version}-seed{seed}')
        self.path=OUT/'replays'/(self.run_id+'.json.gz')
        self.reproduction=reproduction
        self.parameters=parameters
        if native is None:native=bool(label) and reproduction is None
        if every is None:every=int(os.environ.get('DEDICATED_REPLAY_EVERY','10' if scenario=='generated' else '5'))
        if reproduction:title='New reproduction | '+title
        if scenario=='generated':
            limits='Original generated map, normal spawns, food and birth costs. Observation-only policy. Diagnostic hidden state is not a policy input.'
        else:
            limits='Privileged arranged geometry and initial energy; finite prepared food/trees if configured. No new tree or predator spawns. Actions use only cached observations. Stationary workers when present.'
        if reproduction:limits+=' New local reproduction of saved parameters; original missing frames are not recovered. Outcomes may differ.'
        source=Path(__file__).parent/(module+'.py')
        if policy=='adaptive':source=ROOT/'research/sprinter_decoys.py'
        if policy=='nursery':source=ROOT/'research/simple_policies.py'
        self.recorder=ReplayRecorder(env,title=title,policy=f'{policy} / {version}',seed=seed,
            every=every,scenario=scenario,notes=limits,policy_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_render=native,native_width=960)
        self.recorder.meta.update(run_id=self.run_id,parameters=parameters,requested_horizon=horizon,
            reproduction=reproduction,recording_kind='new reproduction' if reproduction else 'new simulation run')
        self.recorder.capture(force=True)
    def capture(self,*args,**kwargs):return self.recorder.capture(*args,**kwargs)
    def finish(self,result,reason):
        summary=self.recorder.save(self.path,reason=reason)
        result.update(run_id=self.run_id,replay=str(self.path.relative_to(ROOT)),
                      replay_kind='new reproduction' if self.reproduction else 'original recording of this new run')
        if self.reproduction:result['reproduction_of']=self.reproduction
        receipt=OUT/'runs'/(self.run_id+'.json')
        receipt.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(source_commit=self.recorder.meta['source_commit'],meta=self.recorder.meta,
                     replay_summary=summary,result=result)
        temporary=receipt.parent/('.'+receipt.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)
            os.link(temporary,receipt)
        finally:
            temporary.unlink(missing_ok=True)
        return self.path

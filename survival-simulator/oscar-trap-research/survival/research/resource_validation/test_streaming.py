"""Compare disk-backed and original capture on the exact same native steps."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from pathlib import Path
import sys,json,gzip,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'debugger'),str(ROOT/'vendor/survival-simulator')]
from recorder import ReplayRecorder as Original
from streaming_recorder import ReplayRecorder as Streaming
from src.core import SimulationCore
from uuid import uuid4

env=SimulationCore(starting_agents=0,starting_predators=0,seed=18181).env
args=dict(title='Streaming recorder equivalence test',policy='recording-validation',seed=18181,
          native_render=True,native_width=320,every=1,
          scenario='zero actors; same native environment observed by both recorders')
a=Original(env,**args);b=Streaming(env,**args)
a.capture();b.capture()
for tick in range(10):
 env.non_agent_step(.1)
 a.capture();b.capture()
out=ROOT/'results/resource_validation/replays';out.mkdir(parents=True,exist_ok=True)
tag=uuid4().hex[:8];left=out/f'original-{tag}.json.gz';right=out/f'streaming-{tag}.json.gz'
a.save(left,reason='equivalence test complete');b.save(right,reason='equivalence test complete')
with gzip.open(left,'rt') as f:x=json.load(f)
with gzip.open(right,'rt') as f:y=json.load(f)
for key in ('world','frames','events','summary'):assert x[key]==y[key],key
assert len(y['frames'])==11 and all(f.get('native_image') for f in y['frames'])
assert not b.frames.path.exists()
report={'passed':True,'same_native_steps':10,'frames':11,'exact_match':['world','frames','events','summary'],
        'recordings':[str(p.relative_to(ROOT)) for p in (left,right)],
        'streaming_source_sha256':hashlib.sha256((ROOT/'debugger/streaming_recorder.py').read_bytes()).hexdigest()}
(ROOT/'results/resource_validation/equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

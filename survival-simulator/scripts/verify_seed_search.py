"""Verify candidates using only the recorded public transcript and own actions."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import argparse,gzip,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.seed_shadow.replay import ShadowJournal
from fastsim import SimulationCore
p=argparse.ArgumentParser()
p.add_argument('--journal',type=pathlib.Path,required=True)
p.add_argument('--search',type=pathlib.Path,required=True)
p.add_argument('--samples',type=pathlib.Path,required=True)
args=p.parse_args()
process_start=time.monotonic()
manifest=json.loads((args.search/'manifest.json').read_text())
progress=json.loads((args.search/'progress.json').read_text())
if not progress['complete'] or progress['checked'] != 2**32 or manifest['start']!=0 or manifest['end']!=2**32:
    raise SystemExit('This verifier requires a complete uint32 search')
if hashlib.sha256(args.samples.read_bytes()).hexdigest()!=manifest['samples_sha256']:
    raise SystemExit('Public terrain sample provenance mismatch')
journal=ShadowJournal(deadline_seconds=max(0.,600-progress['seconds']))
journal.begin_search()
with gzip.open(args.journal,'rt') as f:
    for line in f:
        frame=json.loads(line)
        journal.record(frame['actions'],frame['public'])
expected=''.join(f'{x} {y} {label}\n' for x,y,label in journal.terrain.rows())
if expected!=args.samples.read_text():
    raise SystemExit('Terrain samples do not come from this public transcript')
candidates=json.loads((args.search/'candidates.json').read_text())
start=time.monotonic();seed=journal.recover(candidates,SimulationCore,complete_search=True)
result=dict(status=journal.status,recovered_seed=seed,terrain_candidates=len(candidates),
    replay_frames=len(journal.frames),replay_seconds=time.monotonic()-start,
    search_seconds=progress['seconds'],total_compute_seconds=progress['seconds']+time.monotonic()-process_start,
    full_uint32_search=True,provenance='public observations and own actions only',
    caveat='Public consistency is not proof that all unobserved dynamic state is exact.')
(args.search/'recovery.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)

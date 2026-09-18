"""Stress the depth boundary with33 direct sequential arrivals, every native frame."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import run
STOP=Path('/tmp/predator-intake-stop')

def job(config):
    if STOP.exists():return {'stopped':True}
    r=run.run(predators=33,seconds=120,interval=.1,native=True,every=1,lateral_spread=15,heading_jitter=.3,length=90,right_length=70,face_offset=-5,horizontal=True,**config)
    print({k:r.get(k) for k in ['depth','gap','seed','seconds','bait_alive','acquired','final_joint']},flush=True)
    return r

if __name__=='__main__':
    cases=[dict(depth=d,gap=g,seed=seed) for seed in (631,632) for g in (11,15,19) for d in (4.5,4.9,5)]
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(job,cases))
    (run.OUT/'crowd-frontier-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

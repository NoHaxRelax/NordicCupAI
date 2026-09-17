"""Native every-frame safety frontier; positive depth inside, negative outside."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import run
STOP=Path('/tmp/predator-intake-stop')

def job(config):
    if STOP.exists():return {'stopped':True}
    r=run.run(predators=1,seconds=60,interval=12,native=True,every=1,lateral_spread=0,heading_jitter=0,length=90,right_length=70,face_offset=-5,horizontal=True,**config)
    print({k:r.get(k) for k in ['depth','gap','seed','seconds','bait_alive']},flush=True)
    return r

if __name__=='__main__':
    cases=[dict(depth=d,gap=g,seed=600+int(g)) for g in (11,15,19) for d in (-5,0,2.5,4,4.5,4.9)]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(job,cases))
    (run.OUT/'frontier-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

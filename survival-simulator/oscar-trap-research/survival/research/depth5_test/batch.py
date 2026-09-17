from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import run

def job(config):
    r=run.run(predators=1,seconds=60,interval=12,native=True,every=1,lateral_spread=0,heading_jitter=0,**config)
    print({k:r.get(k) for k in ['depth','gap','seed','seconds','bait_alive','acquired']},flush=True)
    return r

if __name__=='__main__':
    cases=[dict(depth=d,gap=g,seed=600+int(g),length=90,right_length=70,face_offset=-5,horizontal=True) for g in (11,15,19) for d in (5,30)]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(job,cases))
    (run.OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

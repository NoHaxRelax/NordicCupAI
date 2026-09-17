"""Frozen gap v1, independent approach stress; every case recorded."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys
from usage import STOP
sys.path.insert(0,str(Path(__file__).parent/'gap_v1'))
import run

def job(config):
    if STOP.exists():return {'stopped':True}
    return run.run(predators=33,interval=12,seconds=900,**config)

if __name__=='__main__':
    configs=[dict(seed=401,lateral_spread=25,heading_jitter=.5),dict(seed=402,lateral_spread=60,heading_jitter=.8),dict(seed=403,lateral_spread=25,heading_jitter=.5,awake=False),dict(seed=404,lateral_spread=60,heading_jitter=.8,awake=False)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','arrivals','acquired','joint_losses','final_joint','joint_success']},flush=True)
    (run.OUT/'stress-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

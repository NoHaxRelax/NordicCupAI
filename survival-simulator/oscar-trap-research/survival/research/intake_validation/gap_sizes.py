"""Frozen gapv1 size extremes with full-game sequential admission."""
from gap_stress import job, run
from concurrent.futures import ProcessPoolExecutor
from usage import STOP
import json

def size_job(config):
    if STOP.exists():return {'stopped':True}
    return run.run(predators=33,interval=90,seconds=3000,**config)

if __name__=='__main__':
    configs=[dict(seed=411,gap=11,length=55),dict(seed=412,gap=19,length=100)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(size_job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','gap','length','seconds','acquired','joint_losses','joint_success']},flush=True)
    (run.OUT/'size-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

"""Independent frozen guided-gap heldout games."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json
from usage import STOP
sys.path.insert(0,str(Path(__file__).parent/'guided_gap_v1'))
import gap_guide_run as run

def job(config):
    if STOP.exists():return {'stopped':True}
    return run.run(predators=33,interval=90,seconds=3000,**config)

if __name__=='__main__':
    configs=[dict(seed=441,lateral=90,approach=180,follower_gap=50),dict(seed=442,lateral=30,approach=150,follower_gap=80)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','acquired','physical_losses','final_joint','physical_success','guides_alive']},flush=True)
    (run.OUT/'stress-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

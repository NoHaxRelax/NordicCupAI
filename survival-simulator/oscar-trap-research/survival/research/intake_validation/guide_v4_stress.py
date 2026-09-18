"""Independent long-lead HOLD full games, frozen observation-only policy."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json
from usage import STOP
sys.path.insert(0,str(Path(__file__).parent/'guided_gap_v4'))
import gap_guide_run_v4 as run

def job(config):
    if STOP.exists():return {'stopped':True}
    return run.run(predators=33,interval=90,seconds=3000,**config)

if __name__=='__main__':
    configs=[dict(seed=481,gap=15.326311819067314,length=86.96163776214033,right_length=79.0561549508511,face_offset=8.469751440614914,horizontal=True,lateral=60,approach=150,follower_gap=65,native=True,every=100),dict(seed=482,gap=11,length=55,right_length=65,face_offset=-5,horizontal=False,lateral=60,approach=150,follower_gap=50)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','acquired','physical_losses','final_joint','physical_success','guides_alive']},flush=True)
    (run.OUT/'stress-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

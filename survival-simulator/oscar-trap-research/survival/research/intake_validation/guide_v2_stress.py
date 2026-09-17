"""Independent axial-v2 full games on native-example gap dimensions."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json
from usage import STOP
sys.path.insert(0,str(Path(__file__).parent/'guided_gap_v2'))
import gap_guide_run_v2 as run

def job(config):
    if STOP.exists():return {'stopped':True}
    return run.run(predators=33,interval=90,seconds=3000,**config)

if __name__=='__main__':
    configs=[dict(seed=461,gap=14.966920902528898,length=81.78606828487959,right_length=95.51246609533972,face_offset=-14.75513774349713,horizontal=False,lateral=90,approach=180,follower_gap=50,native=True,every=100),dict(seed=462,gap=15.326311819067314,length=86.96163776214033,right_length=79.0561549508511,face_offset=8.469751440614914,horizontal=True,lateral=60,approach=150,follower_gap=65)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','acquired','physical_losses','final_joint','physical_success','guides_alive','max_post_acquisition_mouth_distance']},flush=True)
    (run.OUT/'stress-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

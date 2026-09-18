"""Independent frozen shallow-bait full games at gap width extremes."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json
from usage import STOP
sys.path.insert(0,str(Path(__file__).parent/'shallow_gap'))
import run

def job(config):
    if STOP.exists():return {'stopped':True}
    return run.run(predators=33,interval=90,seconds=3000,**config)

if __name__=='__main__':
    configs=[dict(seed=491,gap=11,length=55,right_length=65,face_offset=-5,horizontal=False,lateral=60,approach=150,follower_gap=65),dict(seed=492,gap=19,length=100,right_length=80,face_offset=10,horizontal=True,lateral=60,approach=150,follower_gap=65)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','acquired','physical_losses','final_joint','physical_success','bait_alive']},flush=True)
    (run.OUT/'stress-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

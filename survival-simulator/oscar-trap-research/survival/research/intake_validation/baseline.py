"""Independent, recorded sparse-arrival baseline for the existing controller."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from usage import STOP

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/wall_funneling'))
import observed_run
OUT=ROOT/'results/intake_validation'


def job(seed):
    if STOP.exists():return {'stopped':True,'seed':seed}
    observed_run.OUT=OUT
    OUT.mkdir(parents=True,exist_ok=True)
    return observed_run.run(predators=33,baits=2,waves=33,interval=90,
        seconds=3000,seed=seed,staged_guides=True)


if __name__=='__main__':
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,[201,202,203]):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','arrivals','final_held','all_held','holders_alive']},flush=True)
    (OUT/'baseline-sparse.json').write_text(json.dumps(rows,indent=2)+'\n')

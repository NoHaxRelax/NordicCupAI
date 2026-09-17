"""Rotate proven fixtures to check that geometry claims do not hide world-axis dependence."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,importlib.util
from usage import STOP
HERE=Path(__file__).parent

def job(case):
    if STOP.exists():return {'stopped':True}
    kind,config=case
    directory=HERE/('guided_gap_v4' if kind=='v4' else 'shallow_gap')
    sys.path.insert(0,str(directory))
    path=directory/('gap_guide_run_v4.py' if kind=='v4' else 'run.py')
    spec=importlib.util.spec_from_file_location('orientation_run_'+kind,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run(predators=33,interval=90,seconds=3000,**config)

if __name__=='__main__':
    cases=[('v4',dict(seed=485,horizontal=False)),('shallow',dict(seed=495,gap=14.9669,length=81.7861,right_length=95.5125,face_offset=-14.7551,horizontal=True))]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,cases):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','acquired','physical_losses','final_joint','physical_success']},flush=True)
    (HERE.parents[1]/'results/intake_validation/orientation-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

"""Fresh-seed repetitions of the first successful33-predator guided protocol."""
from guide_v4_stress import job,run
from concurrent.futures import ProcessPoolExecutor
import json
if __name__=='__main__':
    configs=[dict(seed=483),dict(seed=484)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(job,configs):
            rows.append(row)
            print({k:row.get(k) for k in ['seed','seconds','acquired','physical_losses','final_joint','physical_success','guides_alive']},flush=True)
    (run.OUT/'repeat-summary.json').write_text(json.dumps(rows,indent=2)+'\n')

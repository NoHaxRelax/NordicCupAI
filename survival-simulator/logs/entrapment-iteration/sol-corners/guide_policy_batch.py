"""Two-worker fixed 20-encounter batch for the generic corner override."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, hashlib, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import guide_multi_corner_override as override

def worker(job):
    index,seed,root=job;out=Path(root)/f'case-{index:02d}'
    out.mkdir(parents=True,exist_ok=False);folder=override.run_one(seed,out,True)
    summary=json.loads((folder/'summary.json').read_text())
    return dict(index=index,encounter_seed=seed,folder=str(folder.relative_to(Path(root))),
                outcome=summary['outcome'],initial_min_held=summary['initial_min_held'],
                final_hold_min=summary['final_hold_min'],deliveries=summary['deliveries'],
                replacement=summary.get('replacement'),rear_final=summary['replacement_side_final_period'],
                policy_sha256=summary['policy_sha256'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.output.exists():p.error('output exists')
    a.output.mkdir(parents=True)
    manifest=dict(map_seed=override.MAP_SEED,encounter_seeds=override.ENCOUNTER_SEEDS,
                  site=override.site_contract(),workers=a.workers,
                  override_sha256=hashlib.sha256(Path(override.__file__).read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,(i,s,str(a.output))) for i,s in enumerate(override.ENCOUNTER_SEEDS)]
        for future in as_completed(futures):
            row=future.result();rows.append(row);print(row,flush=True)
    rows.sort(key=lambda x:x['index']);(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),passes=sum(x['outcome']=='delivery_pass' for x in rows),
        initial_retention=sum(x['initial_min_held']==30 for x in rows),
        replacement_arrived=sum(x.get('replacement',{}).get('arrived',False) for x in rows),
        replacement_alive=sum(x.get('replacement',{}).get('alive_at_end',False) for x in rows)),indent=2))
if __name__=='__main__':main()

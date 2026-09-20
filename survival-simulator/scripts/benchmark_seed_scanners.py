"""Compare scanners on identical inputs; timings do not prove full recovery."""
import argparse,json,subprocess,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--binary',action='append',required=True);p.add_argument('--count',type=int,default=4194304);p.add_argument('--rounds',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[];reference=None
for rep in range(a.rounds):
    for binary in a.binary:
        proc=subprocess.run([binary,a.samples,'0',str(a.count)],capture_output=True,text=True,check=True)
        if reference is None:reference=proc.stdout
        if proc.stdout!=reference:raise RuntimeError('Candidate sets differ')
        row=json.loads(proc.stderr);row.update(binary=Path(binary).name,repeat=rep);rows.append(row);print(json.dumps(row),flush=True)
result=dict(scope='Same-input bounded throughput; not full-domain recovery',candidate_sets_match=True,rows=rows,median_seconds={Path(b).name:statistics.median(r['seconds'] for r in rows if r['binary']==Path(b).name) for b in a.binary})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')

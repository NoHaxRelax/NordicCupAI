"""Read-only compact status collection through the shared GPU helper."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess
import sys


REMOTE = r'''
import csv,json,os,subprocess
from pathlib import Path
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return None
work=Path(os.environ['NORDIC_WORK']);report={'profile':work.name,'jobs':[]}
for run in sorted((work/'runs').glob('drone-*')):
    q=run/'experiments';state=read(q/'status.json') or {}
    job={'job':run.name,'receipt':read(run/'result.json'),'queue':state,'experiments':[]}
    preflight=read(run/'preflight.json')
    job['preflight_ready']=preflight.get('ready') if preflight else None
    for path in sorted(q.glob('*/runtime.json')):
        exp=path.parent;runtime=read(path) or {};progress=read(exp/'progress.json') or {}
        result=read(exp/'result.json');pid=runtime.get('pid')
        cmd=Path('/proc')/str(pid)/'cmdline'
        command=cmd.read_bytes().decode().replace('\0',' ') if cmd.exists() else ''
        row={'experiment':exp.name,'training_process_alive':'train.py' in command,
             'progress':{k:v for k,v in progress.items() if k!='dev'},
             'failure':read(exp/'failure.json'),'completed':bool(result)}
        dev=(result or progress).get('dev',{})
        row['dev']={z:{k:v for k,v in m.items() if k not in ['confusion_matrix','per_class','per_class_ap50']} for z,m in dev.items()}
        csvpath=exp/'fit/results.csv'
        if csvpath.exists():
            values=list(csv.DictReader(csvpath.read_text().splitlines()))
            row['latest_detector_metrics']=values[-1] if values else None
        row['checkpoints']=[{'file':str(p.relative_to(work)),'bytes':p.stat().st_size}
                            for p in [exp/'best.pt',exp/'last.pt',exp/'fit/weights/best.pt',exp/'fit/weights/last.pt'] if p.exists()]
        job['experiments'].append(row)
    report['jobs'].append(job)
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print(json.dumps(report))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--brief',action='store_true',help='Save full evidence but print only active work')
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    def collect(profile):
        r=subprocess.run([sys.executable,str(root/'compute/gpu.py'),'exec',profile,'--','python','-c',REMOTE],
                         capture_output=True,text=True,timeout=90)
        if r.returncode:return dict(profile=profile,error=r.stderr[-2000:])
        return json.loads(r.stdout)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(collect,['a100-a','a100-b','a100-c']))
    data=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),profiles=results)
    if a.output.exists():raise SystemExit('Use a new timestamped output path')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    if a.brief:
        compact=[]
        for profile in results:
            jobs=[]
            for job in profile.get('jobs',[]):
                if job.get('receipt'):continue
                queue=job.get('queue') or {}
                jobs.append(dict(job=job['job'],state=queue.get('state'),current=queue.get('current'),
                    failed=queue.get('failed'),experiments=[dict(experiment=e['experiment'],progress=e.get('progress'),failure=e.get('failure'))
                    for e in job.get('experiments',[]) if not e.get('completed')]))
            compact.append(dict(profile=profile['profile'],gpu=profile.get('gpu'),error=profile.get('error'),active=jobs))
        print(json.dumps(dict(checked_at=data['checked_at'],profiles=compact),indent=2))
    else:print(json.dumps(data,indent=2))


if __name__=='__main__':main()

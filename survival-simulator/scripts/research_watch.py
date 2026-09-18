"""Read quota metadata, task compute estimate, and hourly cross-branch changes.

Account/session contents are never copied into the repository. The state file
belongs under ignored logs. Remote checks inspect all origin branches.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def quota():
    candidates=[]
    paths=sorted((Path.home()/'.codex/sessions').rglob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in paths[:12]:
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-2_000_000))
            for line in handle:
                if b'"rate_limits"' not in line:continue
                try:row=json.loads(line)
                except (ValueError,UnicodeDecodeError):continue
                limits=row.get('payload',{}).get('rate_limits')
                if isinstance(limits,dict) and limits.get('primary'):
                    candidates.append(dict(timestamp=row.get('timestamp'),primary=limits['primary']))
    return max(candidates,key=lambda r:r['timestamp']) if candidates else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-remotes',action='store_true')
    parser.add_argument('--state',type=Path,default=ROOT/'logs/entrapment-iteration/research-watch.json')
    args=parser.parse_args();now=datetime.now(timezone.utc)
    previous=json.loads(args.state.read_text()) if args.state.exists() else {}
    result=dict(timestamp=now.isoformat(),quota=quota())
    budget=json.loads((ROOT/'docs/entrapment_iteration/runpod-budget.json').read_text())
    estimate=0.
    for pod in budget['created_pods']:
        start=datetime.fromisoformat(pod['created_at_utc'].replace('Z','+00:00'))
        end=datetime.fromisoformat(pod['terminated_at_utc'].replace('Z','+00:00')) if pod.get('terminated_at_utc') else now
        estimate+=max(0,(end-start).total_seconds())/3600*pod['hourly_usd']
    result['runpod_estimate_usd']=round(estimate,3)
    result['runpod_cap_usd']=budget['budget_usd']
    if args.check_remotes:
        subprocess.run(['git','fetch','origin'],cwd=ROOT,check=True)
        refs=dict(line.split() for line in git('for-each-ref','--format=%(refname:short) %(objectname)','refs/remotes/origin').splitlines())
        changes=[]
        for ref,oid in refs.items():
            old=previous.get('refs',{}).get(ref)
            if old==oid:continue
            files=git('diff','--name-only',old,oid).splitlines() if old else git('ls-tree','-r','--name-only',oid).splitlines()
            relevant=[p for p in files if p.endswith('.py') and any(s in p.lower() for s in ('survival','orchard','population'))]
            if relevant:changes.append(dict(ref=ref,old=old,new=oid,files=relevant))
        result['remote_changes']=changes;result['refs']=refs;result['remote_checked_at']=now.isoformat()
    else:
        for key in ('refs','remote_checked_at'):
            if key in previous:result[key]=previous[key]
    args.state.parent.mkdir(parents=True,exist_ok=True)
    args.state.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='refs'},indent=2))


if __name__=='__main__':main()

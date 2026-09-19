"""Start one prepared study after the controller's fleet guard acknowledges it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--guard-journal',type=Path,required=True)
    parser.add_argument('--runner',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    if (root/'launch-receipt.json').exists() or (root/'state.json').exists():
        raise ValueError('Study already launched')
    pod=json.loads((root/'pod.json').read_text())
    container=dict(part.split(b'=',1) for part in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in part)
    actual=os.environ.get('RUNPOD_POD_ID') or container.get(b'RUNPOD_POD_ID',b'').decode()
    if actual!=pod['id']:raise ValueError('Wrong Pod for this study')
    heartbeat=args.guard_journal.with_suffix('.heartbeat.json')
    for _ in range(40):
        if heartbeat.exists():
            guard=json.loads(heartbeat.read_text())
            if time.time()-guard['updated_at']<35 and actual in guard['checked']:break
        time.sleep(1)
    else:raise ValueError('Fleet shutdown guard did not acknowledge this Pod')
    script=root/'launch.sh'
    script.write_text('#!/bin/bash\nset -eu\n'+"trap 'touch "+str(root/'launcher.finished')+"' EXIT\n"
        +'export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n'
        +'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
        +'/workspace/predator-search-venv/bin/python -B -u '+str(args.runner)+' run --plan '+str(root/'plan.json')+' --pod '+str(root/'pod.json')+'\n')
    with (root/'console.log').open('ab') as log:
        process=subprocess.Popen(['/bin/bash',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pod_id=actual,launcher_pid=process.pid,started_at=time.time(),deadline_epoch=pod['deadline_epoch'],guard='controller fleet guard')
    (root/'launch-receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()

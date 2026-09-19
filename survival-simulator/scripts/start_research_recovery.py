"""Start the prepared continuation only after auth, checks and budget guard pass."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

OUT=Path('/workspace/research-resume-20260919')
SOURCE=Path('/workspace/research-recovery-source')
PYTHON='/workspace/predator-search-venv/bin/python'
POD='a0aruuldun08t9'
CREATED='2026-09-19T06:08:06.38Z'


def read(path):return json.loads(path.read_text())


def main():
    environment=dict(part.split(b'=',1) for part in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in part)
    actual=os.environ.get('RUNPOD_POD_ID') or environment.get(b'RUNPOD_POD_ID',b'').decode()
    if actual != POD:raise ValueError('Wrong Pod for this recovery')
    if (OUT/'launch-receipt.json').exists():raise ValueError('Recovery already launched; use its existing supervisor')
    if (OUT/'STOP').exists() or (OUT/'launcher.finished').exists():raise ValueError('Explicit stop or prior completion requires operator review')
    if not read(OUT/'recovery-preflight.json')['complete']:raise ValueError('Trusted checks have not passed')
    helper=Path('/root/.local/bin/codex-code-mode-host')
    if not helper.is_file() or not os.access(helper,os.X_OK):
        raise ValueError('Matching Codex code-mode helper is required before launching reviews')
    auth=Path('/root/.codex/auth.json')
    if not auth.exists() or read(auth).get('auth_mode')!='chatgpt':raise ValueError('ChatGPT subscription login is required')
    auth.chmod(0o600)
    login=subprocess.run(['/root/.local/bin/codex','login','status'],capture_output=True,text=True)
    if login.returncode or 'ChatGPT' not in login.stdout+login.stderr:raise ValueError('ChatGPT login could not be verified')
    key=Path('/root/.config/nordiccup/runpod-api-key')
    if not key.exists() or key.stat().st_size < 20:raise ValueError('Private shutdown key is required')
    key.chmod(0o600)
    config=read(OUT/'config.json')
    deadline=datetime.fromisoformat(CREATED.replace('Z','+00:00')).timestamp()+24*3600
    if deadline-time.time() < config['schedule']['final_reserve_seconds']+3600:
        raise ValueError('Insufficient funded runtime remains for development and final assessment')
    guard=dict(pod_id=POD,created_at=CREATED,deadline_epoch=deadline,
        finished_file=str(OUT/'launcher.finished'),budget=dict(total_usd=40,prior_spend_usd=10,reserve_usd=5,hourly_rate_usd=.985))
    (OUT/'shutdown.json').write_text(json.dumps(guard,indent=2))
    guard_command=['python3','/workspace/recovery-ops/runpod_shutdown_guard.py','--config',str(OUT/'shutdown.json'),
        '--key-file',str(key),'--journal',str(OUT/'shutdown.jsonl')]
    subprocess.run([*guard_command,'--check'],check=True,capture_output=True)
    with (OUT/'shutdown.console.log').open('ab') as output:
        watchdog=subprocess.Popen(guard_command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(30):
        if watchdog.poll() is not None:raise RuntimeError('Shutdown guard exited before launch')
        journal=OUT/'shutdown.jsonl'
        if journal.exists() and any(json.loads(line).get('event')=='armed' for line in journal.read_text().splitlines()):break
        time.sleep(1)
    else:raise RuntimeError('Shutdown guard did not arm')
    launcher=OUT/'launch.sh'
    launcher.write_text('#!/bin/bash\nset -uo pipefail\n'
        +"trap 'touch "+str(OUT/'launcher.finished')+"' EXIT\n"
        +'cd '+str(SOURCE)+' || exit 1\n'
        +'export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n'
        +'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
        +PYTHON+' -B -u scripts/research_resilient.py --out '+str(OUT)+'\n')
    with (OUT/'console.log').open('ab') as output:
        process=subprocess.Popen(['/bin/bash',str(launcher)],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pod_id=POD,guard_pid=watchdog.pid,launcher_pid=process.pid,
        deadline_epoch=deadline,started_at=time.time(),major_generations=8,
        chatgpt_login_verified=True,model='gpt-6-astra',reasoning='xhigh')
    (OUT/'launch-receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()

"""Bound one supervisor subprocess, even if its parent disappears."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    folder = args.request.parent
    process = None
    result = {'status': 'error', 'returncode': None}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        environment = dict(os.environ)
        environment.update(request.get('env', {}))
        if not request.get('agent', False):
            for key in list(environment):
                if key.endswith(('_API_KEY', '_ACCESS_TOKEN')):
                    environment.pop(key, None)
        with (folder/'console.log').open('wb') as log:
            stdin = open(request['stdin'], 'rb') if request.get('stdin') else open(os.devnull, 'rb')
            try:
                process = subprocess.Popen(request['command'], cwd=request['cwd'], env=environment,
                    stdin=stdin, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=os.name != 'nt')
            finally:
                stdin.close()
            interrupted_at = None
            while process.poll() is None:
                heartbeat = folder/'control'/'heartbeat'
                cancelled = ((folder/'control'/'STOP').exists() or stopping
                             or not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime > 90)
                expired = time.time() >= request['deadline']
                if cancelled or expired:
                    (folder/'control'/'STOP').touch()
                    if interrupted_at is None:
                        interrupted_at = time.time()
                    if time.time()-interrupted_at > request.get('grace_seconds', 45):
                        break
                time.sleep(.25)
            if process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            result = dict(status='interrupted' if interrupted_at else 'complete', returncode=process.returncode)
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None and os.name != 'nt':
            # Reap descendants too, including children left after a parent's exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process is not None and process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        temporary = folder/'process-result.json.tmp'
        temporary.write_text(json.dumps(result), encoding='utf-8')
        temporary.replace(folder/'process-result.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Execute one explicitly prepared validation probe and save bounded receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--active-plan', type=Path, default=ROOT / 'artifacts/drone-api-tests/active-plan.json')
    parser.add_argument('--url-file', type=Path, default=Path('/private/tmp/nordic-ai-cup-secrets/endpoint-url'))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    name = plan['name']
    active = args.active_plan
    folder = ROOT / 'artifacts/drone-api-tests/score-probes' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'queue.json').exists():
        raise SystemExit('Probe already queued; inspect its existing receipt rather than submitting twice')
    active.write_text(json.dumps(plan))
    (folder/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')

    def portal(action, filename, uuid=None):
        command = [sys.executable, str(ROOT/'drone/portal.py'), action,
                   '--url-file', str(args.url_file),
                   '--output', str(folder/filename)]
        if uuid:
            command.extend(['--uuid', uuid])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'{action} failed: {result.stdout[-250:]}')
        return json.loads((folder/filename).read_text())

    queued = portal('validate', 'queue.json')
    uuid = queued['queued_attempt_uuid']
    print(json.dumps({'probe': name, 'uuid': uuid, 'status': queued['status']}), flush=True)
    for _ in range(90):
        time.sleep(15)
        status = portal('queue', 'progress.json', uuid)
        if status['status'] in ('completed', 'finished', 'done'):
            result = portal('result', 'result.json', uuid)
            print(json.dumps({'probe': name, 'result': result}), flush=True)
            pause = ROOT / 'artifacts/drone-api-tests/pause-after-probe'
            if pause.exists():
                print('Paused between completed probes for local maintenance', flush=True)
                while pause.exists():
                    time.sleep(1)
            return
        if status['status'] not in ('queued', 'in_progress', 'running'):
            print(json.dumps({'probe': name, 'unexpected_queue_status': status}), flush=True)
            return
    raise SystemExit('Attempt still pending; inspect existing queue UUID before any further submission')


if __name__ == '__main__':
    main()

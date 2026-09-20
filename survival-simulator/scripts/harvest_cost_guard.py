"""Terminate only this task's recorded temporary pods at its deadline.

Run separately from exploit_lab (which intentionally forbids networking).
Credentials stay in the user's existing Runpod configuration and are not logged.
"""
import argparse
import json
import os
from pathlib import Path
import time
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    import runpod
    config = tomllib.loads((Path.home()/'.runpod/config.toml').read_text())
    runpod.api_key = os.environ.get('RUNPOD_API_KEY') or config.get('apikey') or config.get('api_key') or config.get('apiKey')
    if not runpod.api_key:
        raise RuntimeError('Runpod credential not found; cost guard not armed')
    runpod.get_pods()  # verify auth without logging account data
    print('Cost guard armed for manifest-owned pods only.', flush=True)
    while True:
        state = json.loads(manifest.read_text())
        if state.get('completed') or time.time() >= state['deadline_epoch']:
            live = {p['id'] for p in runpod.get_pods()}
            receipt = []
            for pod in state['created_pods']:
                pid = pod['id']
                if pid in live:
                    runpod.terminate_pod(pid)
                receipt.append(pid)
            manifest.with_suffix('.guard-cleanup.json').write_text(json.dumps(dict(terminated=receipt, at=time.time()), indent=2))
            print('Cost guard cleanup complete.', flush=True)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()

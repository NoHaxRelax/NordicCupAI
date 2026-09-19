"""Watch the existing Runpod campaign over SSH; never starts or stops compute."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    if not 5 <= args.interval <= 300:
        parser.error('--interval must be between 5 and 300 seconds')
    launch = Path(__file__).resolve().parents[1]/'runs/runpod-launch-20260918'
    state = json.loads((launch/'remote-state.json').read_text(encoding='utf-8-sig'))
    campaign = state['campaign']
    if not re.fullmatch(r'/workspace/[a-zA-Z0-9_/-]+', campaign):
        parser.error('Unexpected campaign path')
    cmd = ['ssh', '-n', '-i', str(launch/'id_ed25519_user'), '-p', str(state['ssh_port']),
        '-o', 'UserKnownHostsFile='+str(launch/'known_hosts'), '-o', 'StrictHostKeyChecking=yes',
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10',
        '-o', 'ServerAliveCountMax=2', 'root@'+state['ssh_host'],
        'python3 /workspace/research-monitor.py --out '+campaign]
    latest = launch/'monitor-latest.txt'
    try:
        while True:
            try:
                result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
            except subprocess.TimeoutExpired:
                print('Connection timed out. Check Pod status in Runpod.')
                return 1
            if result.returncode:
                print(result.stderr.strip())
                print('Connection ended. Check Pod status in Runpod.')
                print('Last successful view: '+str(latest))
                return result.returncode
            if not args.once:
                print('\033[2J\033[H', end='')
            print(result.stdout, end='')
            latest.write_text(result.stdout, encoding='utf-8')
            if args.once:
                return 0
            print(f'\nRefreshes every {args.interval}s. Ctrl+C closes this view; research keeps running.', flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nMonitoring closed; the research campaign is unaffected.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())

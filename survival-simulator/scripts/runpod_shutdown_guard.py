"""Stop one explicitly identified Runpod Pod at completion or a fixed deadline.

Runs independently of the research supervisor and SSH session. Credentials stay
outside the research tree. This is a Pod-side watchdog, not a provider scheduler:
loss of the container or API connectivity still requires external intervention.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

API = 'https://api.runpod.io/v2'
TERMINAL = {'complete', 'inconclusive', 'failed', 'paused'}


def validate_deadline(config):
    created = dt.datetime.fromisoformat(config['created_at'].replace('Z', '+00:00')).timestamp()
    deadline = float(config['deadline_epoch'])
    hours = (deadline-created)/3600
    if not math.isfinite(hours) or not 0 < hours <= 48:
        raise ValueError('Deadline must be within 48 hours of Pod creation')
    if hours > 12:
        budget = config.get('budget', {})
        required = ('total_usd','prior_spend_usd','reserve_usd','hourly_rate_usd')
        if any(type(budget.get(k)) not in (int,float) or not math.isfinite(budget[k]) or budget[k] < 0 for k in required):
            raise ValueError('Extended runtime requires a complete financial guard')
        if budget['hourly_rate_usd'] <= 0 or (budget['prior_spend_usd']+budget['reserve_usd']
                +hours*budget['hourly_rate_usd'] > budget['total_usd']):
            raise ValueError('Shutdown deadline exceeds the authorized dollar limit')
    return deadline


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def shutdown_reason(now, deadline, campaign=None, finished=None):
    if now >= deadline:
        return 'hard_deadline'
    if finished and Path(finished).exists():
        return 'launcher_finished'
    if campaign:
        try:
            state = load(Path(campaign)/'state.json')
        except (OSError, ValueError):
            return None
        if state.get('status') in TERMINAL:
            return 'campaign_' + state['status']
    return None


class PodClient:
    def __init__(self, pod_id, created_at, key_file):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', pod_id):
            raise ValueError('Invalid Pod ID')
        self.pod_id, self.created_at = pod_id, created_at
        self.key_file = Path(key_file)

    def request(self, action=None):
        key = self.key_file.read_text(encoding='utf-8-sig').strip()
        if not key or '\n' in key or '\r' in key:
            raise ValueError('Missing or malformed credential')
        body = json.dumps({'action': action}).encode() if action else None
        request = urllib.request.Request(API+'/pods/'+self.pod_id+('/action' if action else ''),
            data=body, headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json',
                                'User-Agent': 'NordicCup-Research/1.0'})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def check(self):
        pod = self.request()
        if pod.get('id') != self.pod_id or pod.get('createdAt') != self.created_at:
            raise ValueError('Pod identity does not match this launch')
        return pod

    def stop(self):
        pod = self.check()
        if pod.get('status') in {'EXITED', 'ERROR'}:
            return pod['status']
        if 'stop' not in pod.get('actions', []):
            raise ValueError('Pod does not currently allow stop')
        result = self.request('stop')
        return result.get('status', 'unknown')


def emit(path, **fields):
    fields['utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    line = json.dumps(fields)
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Read-only credential and identity check')
    args = parser.parse_args()
    config = load(args.config)
    deadline = validate_deadline(config)
    client = PodClient(config['pod_id'], config['created_at'], args.key_file)
    pod = client.check()
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    emit(args.journal, event='checked', pod_id=config['pod_id'], status=pod['status'],
         allowed_actions=pod.get('actions', []), deadline_epoch=deadline, read_only=args.check)
    if args.check:
        return
    emit(args.journal, event='armed', pid=os.getpid(), deadline_epoch=deadline)
    while True:
        reason = shutdown_reason(time.time(), deadline, config.get('campaign'), config.get('finished_file'))
        if reason:
            emit(args.journal, event='stop_requested', reason=reason)
            try:
                status = client.stop()
                emit(args.journal, event='stop_response', status=status)
                if status in {'EXITED', 'ERROR'}:
                    return
            except Exception as exc:
                # No HTTP body, key, request headers or arbitrary exception text in logs.
                emit(args.journal, event='stop_error', error_type=type(exc).__name__,
                     http_status=exc.code if isinstance(exc, urllib.error.HTTPError) else None)
            time.sleep(30)
        else:
            time.sleep(min(10, max(0.1, deadline-time.time())))


if __name__ == '__main__':
    main()

"""Guard the authorized no-trapping search fleet; keep credentials on the controller.

Separate from scripts/runpod_fleet_guard.py, which encodes the trapping
campaign's own $40/four-Pod authorization and is deliberately left untouched so
both campaigns keep independent, verifiable ceilings. Shares only the Runpod
client and journal helpers.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

from runpod_shutdown_guard import PodClient, emit

TOTAL_USD = 80.
RESERVE_USD = 8.
MAX_PODS = 12
MAX_POD_HOURS = 12.
MAX_HOURLY_USD = 1.50


def verify(config):
    if config['total_usd'] != TOTAL_USD or config['reserve_usd'] != RESERVE_USD:
        raise ValueError('Unexpected authorized budget')
    primary = config['primary']
    pods = [primary, *config.get('workers', [])]
    if len(pods) > MAX_PODS or len({p['id'] for p in pods}) != len(pods):
        raise ValueError('Invalid fleet membership')
    reserved = 0.
    for pod in pods:
        created = datetime.fromisoformat(pod['createdAt'].replace('Z', '+00:00')).timestamp()
        hours = (pod['deadline_epoch']-created)/3600
        if not 0 < hours <= MAX_POD_HOURS or not 0 < pod['hourly_rate_usd'] <= MAX_HOURLY_USD:
            raise ValueError('Invalid Pod lifetime/rate')
        reserved += hours*pod['hourly_rate_usd']
    ceiling = config['prior_spend_usd']+config['reserve_usd']+reserved
    if ceiling > TOTAL_USD+1e-5:
        raise ValueError(f'Fleet reservation {ceiling:.2f} exceeds ${TOTAL_USD:.0f}')
    return ceiling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--deadline-only', action='store_true')
    args = parser.parse_args()
    checked, stopped, armed = set(), set(), False
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    while True:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        ceiling = verify(config)
        primary = config['primary']
        workers = config.get('workers', [])
        for pod in [primary, *workers]:
            if pod['id'] in checked:
                continue
            try:
                live = PodClient(pod['id'], pod['createdAt'], args.key_file).check()
                checked.add(pod['id'])
                emit(args.journal, event='checked', pod_id=pod['id'], status=live['status'])
            except Exception as exc:
                emit(args.journal, event='check_error', pod_id=pod['id'], error_type=type(exc).__name__)
        if not armed and primary['id'] in checked:
            emit(args.journal, event='armed', pid=os.getpid(), ceiling_usd=ceiling,
                 pods=len(workers)+1, deadline_only=args.deadline_only)
            armed = True
        now = time.time()
        primary_done = not args.deadline_only and Path(config['primary_finished_file']).exists()
        primary_due = primary_done or now >= primary['deadline_epoch']
        for pod in [*workers, primary]:
            if pod['id'] in stopped:
                continue
            finished = (not args.deadline_only and pod.get('finished_file')
                        and Path(pod['finished_file']).exists())
            due = primary_due or now >= pod['deadline_epoch'] or finished
            # Keep the controller alive until every billed worker has stopped.
            if pod['id'] == primary['id'] and any(w['id'] not in stopped for w in workers):
                continue
            if not due:
                continue
            try:
                status = PodClient(pod['id'], pod['createdAt'], args.key_file).stop()
                emit(args.journal, event='stop_response', pod_id=pod['id'], status=status)
                if status in ('EXITED', 'ERROR'):
                    stopped.add(pod['id'])
            except Exception as exc:
                emit(args.journal, event='stop_error', pod_id=pod['id'], error_type=type(exc).__name__)
        if primary['id'] in stopped:
            return
        heartbeat = args.journal.with_suffix('.heartbeat.json')
        temporary = heartbeat.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_at=time.time(), ceiling_usd=ceiling,
                                             checked=sorted(checked), stopped=sorted(stopped))))
        temporary.replace(heartbeat)
        time.sleep(10)


if __name__ == '__main__':
    main()

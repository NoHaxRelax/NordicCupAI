"""Stop the four-Pod fleet on completion, sustained failure, or a 72h failsafe.

The former $40 target is informational following the user's expanded spending
authorization. No credential is placed on simulation worker Pods.
"""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import time

from research_dispatch import read, write
from runpod_shutdown_guard import PodClient, emit


def verify(config):
    pods = config['pods']
    if config.get('authorization') not in ('continue-four-pods-through-generation-8', 'scale-research-fleet-through-generation-8'):
        raise ValueError('Missing expanded spending authorization')
    maximum = config.get('max_pods', 4)
    if maximum not in (4, 8, 12) or not 4 <= len(pods) <= maximum or len({p['id'] for p in pods}) != len(pods):
        raise ValueError('Invalid authorized fleet membership')
    start = datetime.fromisoformat(pods[0]['createdAt'].replace('Z', '+00:00')).timestamp()
    hours = (config['deadline_epoch']-start)/3600
    if not math.isfinite(hours) or not 0 < hours <= 72 or config['primary_id'] != pods[0]['id']:
        raise ValueError('Invalid emergency runtime or primary Pod')
    if any(not 0 < p['hourly_rate_usd'] <= 1 for p in pods):
        raise ValueError('Unexpected existing Pod price')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--journal', type=Path, required=True)
    p.add_argument('--deadline-only', action='store_true')
    args = p.parse_args()
    config = read(args.config)
    verify(config)
    checked, stopped = set(), set()
    shutdown_reason = None
    idle_since = None
    stale_since = None
    while True:
        config = read(args.config)
        verify(config)
        now = time.time()
        for pod in config['pods']:
            if pod['id'] not in checked:
                try:
                    value = PodClient(pod['id'], pod['createdAt'], args.key_file).check()
                    checked.add(pod['id'])
                    emit(args.journal, event='checked', pod_id=pod['id'], status=value['status'])
                except Exception as exc:
                    emit(args.journal, event='check_error', pod_id=pod['id'], error_type=type(exc).__name__)
        if len(checked) == len(config['pods']) and not (args.journal.with_suffix('.armed.json')).exists():
            emit(args.journal, event='armed', pid=os.getpid(), deadline_only=args.deadline_only,
                 monetary_cap_removed=True, deadline_epoch=config['deadline_epoch'])
            write(args.journal.with_suffix('.armed.json'), dict(pid=os.getpid(), at=now))
        if now >= config['deadline_epoch']:
            shutdown_reason = '72-hour infrastructure failsafe'
        if not args.deadline_only:
            campaign, queue = Path(config['campaign']), Path(config['queue'])
            state = read(campaign/'state.json', {})
            if (Path(config['finished_file']).exists() or state.get('stage') == 'done'
                    or (queue/'STOP').exists()):
                shutdown_reason = 'campaign complete or explicitly stopped'
            unhealthy = (not (campaign/'state.json').exists()
                         or now-(campaign/'state.json').stat().st_mtime > 180
                         or state.get('status') in ('failed', 'paused'))
            stale_since = (stale_since or now) if unhealthy else None
            if stale_since and now-stale_since >= 600:
                shutdown_reason = 'coordinator unhealthy for ten minutes'
            workers = [read(queue/'workers'/f'{pod["id"]}.json', {}) for pod in config['pods']]
            # Startup allowance; low CPU alone never cancels long surviving games.
            fresh = [w for w in workers if now-w.get('updated_at', 0) < 60]
            active = sum(w.get('busy_slots', 0) for w in fresh)
            all_ready = len(fresh) == len(config['pods'])
            idle_since = (idle_since or now) if all_ready and active == 0 else None
            if idle_since and now-idle_since >= 1800:
                shutdown_reason = 'no simulation work for thirty minutes'
        if shutdown_reason:
            if not args.deadline_only:
                Path(config['queue'], 'STOP').touch()
            for pod in [*config['pods'][1:], config['pods'][0]]:
                if pod['id'] in stopped:
                    continue
                if pod['id'] == config['primary_id'] and len(stopped) < len(config['pods'])-1:
                    continue
                try:
                    status = PodClient(pod['id'], pod['createdAt'], args.key_file).stop()
                    emit(args.journal, event='stop_response', pod_id=pod['id'], status=status, reason=shutdown_reason)
                    if status in ('EXITED', 'ERROR'):
                        stopped.add(pod['id'])
                except Exception as exc:
                    emit(args.journal, event='stop_error', pod_id=pod['id'], error_type=type(exc).__name__)
        write(args.journal.with_suffix('.heartbeat.json'), dict(updated_at=time.time(), checked=sorted(checked),
            stopped=sorted(stopped), pid=os.getpid(), shutdown_reason=shutdown_reason))
        if len(stopped) == len(config['pods']):
            return
        time.sleep(10)


if __name__ == '__main__':
    main()

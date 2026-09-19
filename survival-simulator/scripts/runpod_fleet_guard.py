"""Guard the authorized four-Pod research budget; keep credentials on the controller."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from runpod_shutdown_guard import PodClient, emit


def verify(config):
    if config['total_usd'] != 40 or config['reserve_usd'] != 5:
        raise ValueError('Unexpected authorized budget')
    primary = config['primary']
    created = datetime.fromisoformat(primary['createdAt'].replace('Z', '+00:00')).timestamp()
    hours = (primary['deadline_epoch']-created)/3600
    if not 0 < hours <= 24 or config['auxiliary_allowance_usd'] != 8:
        raise ValueError('Invalid fleet budget limits')
    ceiling = config['prior_spend_usd']+config['reserve_usd']+hours*config['primary_hourly_rate_usd']+8
    if ceiling > 40.00001:
        raise ValueError('Fleet reservation exceeds $40')
    workers = config.get('workers', [])
    if len(workers) > 3 or len({p['id'] for p in [primary, *workers]}) != len(workers)+1:
        raise ValueError('Invalid four-Pod membership')
    auxiliary_cost = 0
    for pod in workers:
        start = datetime.fromisoformat(pod['createdAt'].replace('Z', '+00:00')).timestamp()
        duration = (pod['deadline_epoch']-start)/3600
        if not 0 < duration <= 2.7 or not 0 < pod['hourly_rate_usd'] <= .97:
            raise ValueError('Invalid worker lifetime/rate')
        auxiliary_cost += duration*pod['hourly_rate_usd']
    if auxiliary_cost > 8:
        raise ValueError('Auxiliary reservation exceeds $8')
    return ceiling


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--deadline-only',action='store_true')
    args=parser.parse_args()
    checked=set()
    stopped=set()
    armed=False
    args.journal.parent.mkdir(parents=True,exist_ok=True)
    while True:
        config=json.loads(args.config.read_text(encoding='utf-8-sig'))
        ceiling=verify(config)
        primary=config['primary'];workers=config.get('workers',[])
        for pod in [primary,*workers]:
            if pod['id'] not in checked:
                try:
                    client=PodClient(pod['id'],pod['createdAt'],args.key_file)
                    live=client.check()
                    checked.add(pod['id'])
                    emit(args.journal,event='checked',pod_id=pod['id'],status=live['status'])
                except Exception as exc:
                    emit(args.journal,event='check_error',pod_id=pod['id'],error_type=type(exc).__name__)
        if not armed and primary['id'] in checked:
            import os
            emit(args.journal,event='armed',pid=os.getpid(),ceiling_usd=ceiling,deadline_only=args.deadline_only)
            armed=True
        now=time.time()
        primary_done=(not args.deadline_only and Path(config['primary_finished_file']).exists())
        primary_due=primary_done or now >= primary['deadline_epoch']
        for pod in [*workers,primary]:
            if pod['id'] in stopped:continue
            is_primary=pod['id']==primary['id']
            finished=(not args.deadline_only and pod.get('finished_file') and Path(pod['finished_file']).exists())
            due=primary_due or now>=pod['deadline_epoch'] or finished
            # Keep the controller alive until all billed workers have stopped.
            if is_primary and any(w['id'] not in stopped for w in workers):continue
            if not due:continue
            try:
                status=PodClient(pod['id'],pod['createdAt'],args.key_file).stop()
                emit(args.journal,event='stop_response',pod_id=pod['id'],status=status)
                if status in ('EXITED','ERROR'):stopped.add(pod['id'])
            except Exception as exc:
                emit(args.journal,event='stop_error',pod_id=pod['id'],error_type=type(exc).__name__)
        if primary['id'] in stopped:return
        heartbeat=args.journal.with_suffix('.heartbeat.json')
        temporary=heartbeat.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_at=time.time(),checked=sorted(checked),stopped=sorted(stopped))))
        temporary.replace(heartbeat)
        time.sleep(10)


if __name__=='__main__':main()

"""External shared-volume simulation queue. Frozen simulation code is never edited.

One Linux flock per case prevents duplicate cache writers, including after a
worker restart. Queue consumers run each candidate in a fresh interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_job(job, settings):
    campaign = Path(settings['campaign']).resolve()
    source = Path(job['source']).resolve()
    if not source.is_relative_to(campaign/'snapshots') or source.name != 'code':
        raise ValueError('Only frozen campaign snapshots may execute')
    for field in ('cache', 'control'):
        path = Path(job[field]).resolve()
        if not any(path.is_relative_to(Path(root).resolve()) for root in settings['output_roots']):
            raise ValueError('Output escapes registered research roots')
    request = job['request']
    cfg = read(campaign/'config.json')
    evaluation = cfg['evaluation']
    seed = request['seed']
    if seed in evaluation.get('retired_holdout_seeds', []):
        raise ValueError('Retired holdout is sealed')
    if seed in evaluation['holdout_seeds']:
        selection = read(campaign/'final-selection.json', {})
        refs = [selection.get('original', {}), selection.get('selected', {})]
        if (request.get('engine') != 'python' or seed not in selection.get('holdout_seeds', [])
                or not any(ref.get('snapshot') == source.parent.name
                    and ref.get('config') == request.get('config')
                    and ref.get('baseline', False) == request.get('baseline', False) for ref in refs)
                or not Path(job['control']).resolve().is_relative_to(campaign/'steps/final-holdout')):
            raise ValueError('Holdout requires frozen final selection and final evaluation')
    if request.get('seconds') != 3000 or not 0 < job['timeout'] <= 28800:
        raise ValueError('Unexpected horizon or wall-time bound')


def publish(queue, request, cache, control, timeout, source, priority=10):
    queue = Path(queue)
    identity = dict(case_id=request['case_id'], cache=str(Path(cache).resolve()),
                    control=str(Path(control).resolve()), source=str(Path(source).resolve()))
    jid = digest(identity)
    if (queue/'results'/f'{jid}.json').exists():
        return jid
    path = queue/'jobs'/f'{priority:03d}-{jid}.json'
    job = dict(**identity, request=request, timeout=timeout, priority=priority,
               id=jid, created_at=time.time())
    validate_job(job, read(queue/'config.json'))
    if not path.exists():
        # Atomic publication: never expose a partially written request.
        write(path, job)
    else:
        previous = read(path)
        if any(previous[k] != job[k] for k in (*identity, 'request')):
            raise ValueError('Queue identity collision')
    return jid


def remote_case(queue, request, cache, control, timeout, *, runtime_root, priority=10):
    queue, control = Path(queue), Path(control)
    jid = publish(queue, request, cache, control, timeout, runtime_root, priority)
    while True:
        result = read(queue/'results'/f'{jid}.json', {})
        if result:
            if result.get('error'):
                raise RuntimeError(result['error'])
            return read(result['result'])
        if (control/'STOP').exists() or (queue/'STOP').exists():
            return dict(case_id=request['case_id'], seed=request['seed'], mode=request['mode'], status='interrupted')
        time.sleep(.5)


def execute_job(queue, job_path):
    queue = Path(queue)
    job = read(job_path)
    if job['priority'] >= 20 and hasattr(os, 'nice'):
        os.nice(8)  # Background work yields CPU immediately, including all descendants.
    settings = read(queue/'config.json')
    validate_job(job, settings)
    source = Path(job['source'])
    sys.path.insert(0, str(source))
    from scripts.research_support import assert_snapshot, digest as source_digest, research_case
    from scripts.optimize_policy import run_case, versions
    manifest = read(source.parent/'manifest.json')
    if source_digest(manifest) != source.parent.name:
        raise ValueError('Snapshot address mismatch')
    assert_snapshot(source, manifest)
    request = job['request']
    expected = research_case(source.parent.name, versions(), request['config'], request['baseline'],
        request['seed'], request['policy_seed'], request['diagnostics'], request['engine'],
        infrastructure_retries=request.get('infrastructure_retries', 0))
    if expected != request:
        raise ValueError('Worker environment or case identity differs from frozen request')
    import fcntl
    cache = Path(job['cache'])
    folder = cache/'cases'/request['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    # A second logical job (for another comparison) may share this cache case.
    with (folder/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = run_case(request, cache, Path(job['control']), job['timeout'], runtime_root=source)
    assert_snapshot(source, manifest)
    result_path = queue/'case-results'/f'{job["id"]}.json'
    write(result_path, result)
    write(queue/'results'/f'{job["id"]}.json', dict(result=str(result_path), finished_at=time.time()))


def legacy_cases(owned_pids):
    count = 0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command = (proc/'cmdline').read_bytes().split(b'\0')
            if any(part.endswith(b'/experiment_case.py') for part in command):
                # Case parent is the dispatcher job interpreter when queue-owned.
                parent = int((proc/'stat').read_text().rsplit(')', 1)[1].split()[1])
                if parent not in owned_pids:
                    count += 1
        except (OSError, ValueError, IndexError):
            pass
    return count


def cpu_usage():
    path = Path('/sys/fs/cgroup/cpu.stat')
    if path.exists():
        values = dict(line.split() for line in path.read_text().splitlines())
        return int(values['usage_usec']) / 1e6
    for mount in ('cpuacct', 'cpu,cpuacct', 'cpu'):
        path = Path('/sys/fs/cgroup')/mount/'cpuacct.usage'
        if path.exists():
            return int(path.read_text()) / 1e9
    raise RuntimeError('Container CPU accounting is unavailable')


def admission(priority, slots, legacy, active, urgent):
    normal = max(0, slots-legacy-active)
    if priority > 10:
        return normal
    # A small urgent burst avoids waiting for long background games. Lower-nice
    # background processes yield CPU; none are killed or assigned worse scores.
    return max(normal, min(max(0, 8-urgent), max(0, slots+8-legacy-active)))


def worker(queue, worker_id, slots):
    import fcntl
    queue = Path(queue)
    active = {}
    stopping = False
    last_cpu, last_tick = cpu_usage(), time.monotonic()
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping or active:
        settings = read(queue/'config.json')
        campaign = Path(settings['campaign'])
        state = read(campaign/'state.json', {})
        if ((queue/'STOP').exists() or state.get('stage') == 'done'
                or time.time() >= settings['deadline_epoch']):
            stopping = True
        for jid, item in list(active.items()):
            process, lock, job = item
            code = process.poll()
            if code is None:
                continue
            if not (queue/'results'/f'{jid}.json').exists():
                write(queue/'results'/f'{jid}.json', dict(error=f'Dispatch executor exited {code}; inspect job log',
                      finished_at=time.time()))
            lock.close()
            del active[jid]
        legacy = legacy_cases({item[0].pid for item in active.values()})
        capacity = max(0, slots - legacy - len(active))
        pending = 0
        for job_path in sorted((queue/'jobs').glob('*.json')):
            jid = job_path.stem.split('-', 1)[1]
            if (queue/'results'/f'{jid}.json').exists():
                try:
                    job_path.replace(queue/'archive'/job_path.name)
                except FileNotFoundError:
                    pass  # Another worker already archived this immutable job.
                continue
            pending += 1
            priority = int(job_path.name.split('-', 1)[0])
            urgent = sum(item[2]['priority'] <= 10 for item in active.values())
            capacity = admission(priority, slots, legacy, len(active), urgent)
            if not capacity or stopping:
                continue
            lockpath = queue/'locks'/f'{jid}.lock'
            lockpath.parent.mkdir(exist_ok=True)
            lock = lockpath.open('a+')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            if (queue/'results'/f'{jid}.json').exists():
                lock.close()
                continue
            try:
                job = read(job_path)
            except FileNotFoundError:
                lock.close()
                continue
            control = Path(job['control'])
            if (control/'STOP').exists():
                result_path = queue/'case-results'/f'{jid}.json'
                write(result_path, dict(case_id=job['case_id'], seed=job['request']['seed'],
                    mode=job['request']['mode'], status='interrupted'))
                write(queue/'results'/f'{jid}.json', dict(result=str(result_path), finished_at=time.time()))
                lock.close()
                continue
            heartbeat = control/'heartbeat'
            if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime > 120:
                lock.close()
                continue
            with (queue/'logs'/f'{jid}.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    'case', '--queue', str(queue), '--job', str(job_path)],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(lock.fileno(),),
                    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            active[jid] = (process, lock, job)
            capacity -= 1
        tick, usage = time.monotonic(), cpu_usage()
        used_cores = max(0, (usage-last_cpu)/max(.001, tick-last_tick))
        last_cpu, last_tick = usage, tick
        write(queue/'workers'/f'{worker_id}.json', dict(id=worker_id, pid=os.getpid(),
            updated_at=time.time(), slots=slots, active=len(active), legacy_active=legacy,
            busy_slots=len(active)+legacy, cpu_cores_used=used_cores,
            cpu_percent=100*used_cores/settings['vcpus_per_pod'], stopping=stopping,
            jobs=list(active), pending_or_active=pending))
        if stopping and not active:
            break
        time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['worker', 'case'])
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--job', type=Path)
    p.add_argument('--worker-id')
    p.add_argument('--slots', type=int, default=28)
    args = p.parse_args()
    for name in ('jobs', 'archive', 'results', 'case-results', 'locks', 'logs', 'workers'):
        (args.queue/name).mkdir(parents=True, exist_ok=True)
    if args.mode == 'case':
        execute_job(args.queue, args.job)
    else:
        worker(args.queue, args.worker_id, args.slots)


if __name__ == '__main__':
    main()

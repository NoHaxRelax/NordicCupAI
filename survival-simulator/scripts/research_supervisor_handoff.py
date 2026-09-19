"""Replace only a coordinator while its independently bounded search continues.

Only valid during a running research_process wrapper, never during an inline
comparison or final evaluation. The existing shutdown guard remains armed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

OUT = Path('/workspace/research-night-1')
SOURCE = Path('/workspace/NordicCupAI/survival-simulator')
RUNTIME = Path('/workspace/research-runtime/research_loop.py')
CONTROLS = OUT / 'operator-controls/max-generations-8.json'
PYTHON = '/workspace/predator-search-venv/bin/python'
LAUNCHER = Path('/workspace/research-night-1-launcher.sh')
FINISHED = Path('/workspace/research-night-1.finished')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def proc(pid):
    path = Path('/proc') / str(pid)
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, parent=int(stat[1]), state=stat[0], start=stat[19],
                args=(path / 'cmdline').read_bytes().split(b'\0'))


def wrapper_for(request):
    for path in Path('/proc').glob('[0-9]*'):
        try:
            info = proc(int(path.name))
            if str(request).encode() in info['args'] and any(a.endswith(b'/research_process.py') for a in info['args']):
                return info['pid']
        except (OSError, ValueError):
            pass
    raise ValueError('Independent bounded wrapper is missing')


def safe_step(state):
    if state['status'] != 'running' or state['stage'] in ('final', 'done'):
        raise ValueError('Campaign is not actively developing')
    if FINISHED.exists() or (OUT / 'STOP').exists() or (OUT / 'final-selection.json').exists():
        raise ValueError('Campaign has a stop/finalization marker')
    pending = [(k, s) for k, s in state['steps'].items() if s['status'] in ('pending', 'running')]
    if len(pending) != 1:
        raise ValueError('Handoff requires exactly one running wrapped step')
    key, step = pending[0]
    folder = OUT / 'steps' / key
    if (step['status'] != 'running' or not key.endswith(('.search', '.agent'))
            or not (folder / 'process-request.json').exists() or (folder / 'process-result.json').exists()
            or 'evaluation_control' in step or step['deadline'] - time.time() < 90):
        raise ValueError('Handoff requires an independently wrapped search/review with time remaining')
    heartbeat = folder / 'control/heartbeat'
    if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 30:
        raise ValueError('Search heartbeat is stale')
    return key, step, wrapper_for(folder / 'process-request.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    parser.add_argument('--guard-pid', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    old, guard = proc(args.supervisor_pid), proc(args.guard_pid)
    parent = proc(old['parent'])
    if not any(a.endswith(b'scripts/research_loop.py') for a in old['args']) or str(OUT).encode() not in old['args']:
        raise ValueError('Unexpected old coordinator identity')
    if str(LAUNCHER).encode() not in parent['args'] or not any(a.endswith(b'runpod_shutdown_guard.py') for a in guard['args']):
        raise ValueError('Unexpected launcher or guard identity')
    command = [PYTHON, '-B', '-u', str(RUNTIME), 'run', '--out', str(OUT),
               '--source', str(SOURCE), '--controls', str(CONTROLS)]
    environment = dict(item.split(b'=', 1) for item in
                       (Path('/proc') / str(old['pid']) / 'environ').read_bytes().split(b'\0') if b'=' in item)
    environment = {os.fsdecode(k): os.fsdecode(v) for k, v in environment.items()}
    check = list(command)
    check[4] = 'check'
    result = subprocess.run(check, env=environment, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise ValueError('Runtime preflight failed: ' + result.stderr[-2000:])
    if not 0 <= args.wait_seconds <= 900:
        raise ValueError('Handoff wait must be bounded to at most 15 minutes')
    wait_until = time.monotonic() + args.wait_seconds
    while True:
        state = read(OUT / 'state.json')
        try:
            key, step, wrapper = safe_step(state)
            break
        except ValueError as exc:
            if (time.monotonic() >= wait_until or state['status'] != 'running'
                    or state['stage'] in ('final', 'done') or proc(old['pid'])['start'] != old['start']):
                raise
            write(OUT / 'operator-controls/handoff-pending.json', dict(status='waiting',
                reason=str(exc), target_major_generations=8,
                checked_at=datetime.now(timezone.utc).isoformat()))
            time.sleep(5)
    report = dict(old_supervisor=old['pid'], old_launcher=parent['pid'], guard=args.guard_pid,
                  active_step=key, wrapper_pid=wrapper, deadline=step['deadline'], preflight=json.loads(result.stdout))
    if not args.apply:
        print(json.dumps(report, indent=2))
        return
    protected_paths = [OUT / name for name in ('config.json', 'protocol.json', 'research_agent_prompt.md')]
    protected_paths += list((OUT / 'snapshots').glob('*/manifest.json'))
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
    before = hashes()
    original_launcher = LAUNCHER.read_text()
    replacement = '#!/usr/bin/env bash\nset -uo pipefail\ntrap ' + shlex.quote('touch ' + str(FINISHED)) + ' EXIT\n'
    replacement += 'cd ' + shlex.quote(str(SOURCE)) + ' || exit 1\n' + shlex.join(command) + '\n'
    prepared = RUNTIME.parent / 'launcher.sh'
    prepared.write_text(replacement)
    suspended, replaced = [], False
    try:
        for info in (parent, old):
            if proc(info['pid'])['start'] != info['start']:
                raise ValueError('Process identity changed')
            os.kill(info['pid'], signal.SIGSTOP)
            suspended.append(info['pid'])
        until = time.monotonic() + 3
        while proc(old['pid'])['state'] != 'T' and time.monotonic() < until:
            time.sleep(.01)
        if proc(old['pid'])['state'] != 'T':
            raise ValueError('Coordinator did not suspend')
        frozen = read(OUT / 'state.json')
        final_key, final_step, final_wrapper = safe_step(frozen)
        if (final_key, final_wrapper) != (key, wrapper):
            raise ValueError('Active step changed before handoff')
        report.update(state_before=frozen, protected_hashes=before,
                      handoff_at=datetime.now(timezone.utc).isoformat())
        write(OUT / 'operator-controls/handoff-before.json', report)
        (OUT / 'operator-controls/launcher-before.sh').write_text(original_launcher)
        # Do not send signals to the independently launched wrapper or simulations.
        # SIGKILL prevents the old shell's EXIT trap from emitting a false finish.
        os.kill(parent['pid'], signal.SIGKILL)
        os.kill(old['pid'], signal.SIGKILL)
        replaced = True
        with Path('/workspace/research-night-1.console.log').open('ab') as log:
            launch = subprocess.Popen(['/bin/bash', str(prepared)], cwd=SOURCE, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        until = time.monotonic() + 40
        receipt = {}
        while time.monotonic() < until:
            path = OUT / 'runtime-controls-applied.json'
            if path.exists():
                receipt = read(path)
                if receipt.get('supervisor_pid') != old['pid']:
                    break
            if launch.poll() is not None:
                raise RuntimeError('Replacement launcher exited; inspect campaign console')
            time.sleep(.25)
        if not receipt or receipt.get('supervisor_pid') == old['pid']:
            raise RuntimeError('Replacement did not acknowledge its runtime controls')
        proc(receipt['supervisor_pid'])
        proc(wrapper)
        proc(args.guard_pid)
        if FINISHED.exists() or before != hashes():
            raise RuntimeError('Handoff integrity check failed')
        current = read(OUT / 'state.json')
        if (current['steps'][key]['deadline'] != step['deadline']
                or current['best'] != frozen['best'] or current['started_at'] != frozen['started_at']):
            raise RuntimeError('Handoff changed protected campaign state')
        # The familiar launcher remains the correct entry point for a later resume.
        pending = LAUNCHER.with_suffix('.pending.sh')
        pending.write_text(replacement)
        pending.replace(LAUNCHER)
        result = dict(applied=True, supervisor_pid=receipt['supervisor_pid'], launcher_pid=launch.pid,
            previous_supervisor_pid=old['pid'], previous_launcher_pid=parent['pid'],
            active_step=key, wrapper_pid=wrapper, wrapper_preserved=True, deadline_preserved=True,
            best_preserved=True, started_at_preserved=True, frozen_protocol_preserved=True,
            shutdown_guard_pid=args.guard_pid, effective_major_generations=receipt['schedule']['major_generations'],
            applied_at=receipt['applied_at'])
        write(OUT / 'operator-controls/handoff-result.json', result)
        print(json.dumps(result, indent=2))
    finally:
        if not replaced:
            for pid in reversed(suspended):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    main()

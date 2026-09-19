"""Audited schedule-only overrides around an unchanged frozen supervisor.

Deploy outside the frozen source tree. Evaluation protocols, candidate sources,
seed panels, deadlines, financial limits and holdout selection remain unchanged.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def effective_config(base, control):
    if control.get('version') != 1 or set(control.get('changes', {})) != {'schedule.major_generations'}:
        raise ValueError('Only the major-generation ceiling may be amended')
    delta = control['changes']['schedule.major_generations']
    if (set(delta) != {'from', 'to'} or type(delta['from']) is not int
            or type(delta['to']) is not int or not delta['from'] < delta['to'] <= 32
            or delta['from'] != base['schedule']['major_generations']):
        raise ValueError('Invalid or stale generation amendment')
    result = copy.deepcopy(base)
    result['schedule']['major_generations'] = delta['to']
    return result


def verify(out, control_path, source):
    control = read(control_path)
    if control['campaign'] != str(out) or control['source'] != str(source):
        raise ValueError('Control belongs to a different campaign/source')
    if control['protocol_sha256'] != sha(out / 'protocol.json'):
        raise ValueError('Frozen protocol changed')
    if control['runtime_sha256'] != sha(Path(__file__)):
        raise ValueError('Runtime control implementation changed')
    base = read(out / 'config.json')
    config = effective_config(base, control)
    if read(out / 'protocol.json')['config'] != base:
        raise ValueError('Frozen configuration changed')
    return control, config


def prepare(out, control_path, source, maximum):
    state, base = read(out / 'state.json'), read(out / 'config.json')
    if state['status'] != 'running' or state['stage'] in ('final', 'done') or (out / 'final-selection.json').exists():
        raise ValueError('Only an active development campaign may be amended')
    control = dict(version=1, id='max-generations-8-20260919-01', campaign=str(out), source=str(source),
        requested_at=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly requested increasing the maximum to eight major generations.',
        protocol_sha256=sha(out / 'protocol.json'), runtime_sha256=sha(Path(__file__)),
        changes={'schedule.major_generations': {'from': base['schedule']['major_generations'], 'to': maximum}})
    effective_config(base, control)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    if control_path.exists():
        existing, _ = verify(out, control_path, source)
        if existing['changes'] != control['changes']:
            raise ValueError('An issued amendment is immutable')
    else:
        with control_path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(control, indent=2))
    return verify(out, control_path, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--max-generations', type=int, default=8)
    args = parser.parse_args()
    out, source, controls = args.out.resolve(), args.source.resolve(), args.controls.resolve()
    if args.command == 'prepare':
        control, config = prepare(out, controls, source, args.max_generations)
    else:
        control, config = verify(out, controls, source)
    sys.path.insert(0, str(source))
    from scripts.research_loop import Supervisor
    from scripts.research_support import validate_config, manifest, digest, assert_snapshot, editable
    from scripts.optimize_policy import study_lock, versions
    validate_config(config)
    # Check the same frozen launch/evaluation sources before any handoff.
    state = read(out / 'state.json')
    snapshot = out / 'snapshots' / state['initial_snapshot']
    expected = read(snapshot / 'manifest.json')
    if digest(expected) != state['initial_snapshot']:
        raise ValueError('Invalid initial source identity')
    assert_snapshot(snapshot / 'code', expected)
    actual = manifest(source)
    for name, checksum in expected.items():
        if not editable(name) and actual.get(name) != checksum:
            raise ValueError(f'Trusted launch source changed: {name}')
    if versions() != read(out / 'protocol.json')['environment']:
        raise ValueError('Runtime environment changed')
    if args.command != 'run':
        print(json.dumps(dict(verified=True, amendment=control['id'], schedule=config['schedule'],
                              budget=config['budget'], agent_max_calls=config['agent']['max_calls'])))
        return
    with study_lock(out / 'supervisor.lock'):
        if (out / 'STOP').exists():
            raise ValueError('An explicit stop must not be cleared by a runtime amendment')
        supervisor = Supervisor(out)
        supervisor.config = effective_config(supervisor.config, control)
        receipt = dict(id=control['id'], controls_file=str(controls), controls_sha256=sha(controls),
            runtime_file=str(Path(__file__).resolve()), runtime_sha256=sha(Path(__file__)),
            applied_at=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
            schedule={'major_generations': supervisor.config['schedule']['major_generations']},
            frozen_protocol_sha256=sha(out / 'protocol.json'),
            financial_limits_unchanged=True, evaluator_unchanged=True, deadlines_unchanged=True)
        supervisor.state['runtime_controls'] = receipt
        supervisor.event(control['id'], 'authorized generation-limit amendment',
                         dict(changes=control['changes'], receipt=receipt,
                              note='Only the generation ceiling changes. Budget, deadlines, adaptive stopping, agent-call cap and final holdout reserve remain in force.'))
        write(out / 'runtime-controls-applied.json', receipt)
        supervisor.run()


if __name__ == '__main__':
    main()

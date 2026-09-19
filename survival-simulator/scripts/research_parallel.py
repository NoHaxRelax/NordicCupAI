"""Bounded independent development studies alongside an immutable campaign.

Deploy this operational helper outside the frozen source tree. Each Pod owns one
study/cache. It never writes campaign state, promotes a candidate or reads holdout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def size(root):
    from scripts.research_io import directory_bytes
    return directory_bytes(root)


def definitions():
    return [dict(name='guide-delivery', title='Guide delivery and trap geometry',
        paths=['guide.safe_distance', 'guide.contact_buffer', 'guide.bait_buffer',
               'guide.lost_wait_ticks', 'guide.reacquire_arrival_distance',
               'guide.delivery_arrival_distance', 'navigator.stall_seconds',
               'safety.guide_geometry_interval'],
        variants=[dict(label='Refresh guide geometry', overrides={'features.refresh_guide_geometry': True}),
                  dict(label='Corner pockets', overrides={'features.corner_pockets': True}),
                  dict(label='Geometry + pockets', overrides={'features.refresh_guide_geometry': True, 'features.corner_pockets': True})]),
      dict(name='bait-continuity', title='Bait continuity and safer replacement',
        paths=['safety.emergency_parent_max_age', 'safety.emergency_reserve',
               'safety.emergency_target_fraction', 'safety.emergency_min_young',
               'trapping_orchard.breed_reserve', 'trapping_orchard.cap_mult',
               'bystander.trap_clearance', 'bystander.contact_clearance'],
        variants=[dict(label='Earlier bounded replacements', overrides={'features.emergency_reproduction': True,
            'safety.emergency_parent_max_age': 70., 'safety.emergency_target_fraction': .25, 'safety.emergency_reserve': 50.}),
          dict(label='Replacement + exclusion', overrides={'features.emergency_reproduction': True,
            'safety.emergency_parent_max_age': 70., 'safety.emergency_target_fraction': .25,
            'safety.emergency_reserve': 50., 'features.trap_exclusion': True}),
          dict(label='Trap exclusion', overrides={'features.trap_exclusion': True})]),
      dict(name='capture-allocation', title='Capture versus feeding-worker allocation',
        paths=['safety.minimum_workers', 'safety.guide_fraction', 'safety.guide_timeout',
               'safety.guide_lost_seconds', 'safety.guide_cooldown',
               'safety.guide_geometry_interval', 'guide.safe_distance', 'guide.bait_buffer'],
        variants=[dict(label='Patient guide allocation', overrides={'features.role_budget': True,
            'safety.guide_timeout': 70., 'safety.guide_lost_seconds': 8.}),
          dict(label='Geometry refresh', overrides={'features.refresh_guide_geometry': True}),
          dict(label='Patient allocation + geometry', overrides={'features.role_budget': True,
            'features.refresh_guide_geometry': True, 'safety.guide_timeout': 70., 'safety.guide_lost_seconds': 8.})])]


def prepare(campaign, out):
    if (out / 'fleet.json').exists():
        raise ValueError('Fleet already prepared; plans are immutable')
    state, config = read(campaign / 'state.json'), read(campaign / 'config.json')
    if state['status'] != 'running' or state['stage'] in ('final', 'done'):
        raise ValueError('Campaign must still be developing candidates')
    incumbent = state['best']
    source = campaign / 'snapshots' / incumbent['snapshot'] / 'code'
    sys.path.insert(0, str(source))
    from scripts.research_support import assert_snapshot, digest
    from models.experiment_config import inventory, put, repair, validate
    expected = read(source.parent / 'manifest.json')
    assert_snapshot(source, expected)
    assert digest(expected) == incumbent['snapshot']
    environment = read(campaign / 'protocol.json')['environment']
    selectable = {p['path'] for p in inventory() if p['tunable']}
    fleet = dict(version=1, campaign=str(campaign), created_at=time.time(),
        total_budget_usd=40, maximum_auxiliary_compute_usd=6.5,
        primary_compute_ceiling_usd=9.65, agent_allowance_ceiling_usd=16,
        safety_reserve_usd=5, storage_and_rounding_allowance_usd=1,
        worst_case_reserved_usd=38.15, tasks=[])
    for i, definition in enumerate(definitions()):
        assert set(definition['paths']) <= selectable
        for variant in definition['variants']:
            candidate = copy.deepcopy(incumbent['config'])
            for path, value in variant['overrides'].items():
                put(candidate, path, value)
            validate(repair(candidate))
        folder = out / definition['name']
        plan = dict(**definition, campaign=str(campaign), root=str(folder), source=str(source),
            snapshot=incumbent['snapshot'], incumbent=incumbent, environment=environment,
            diagnostics=config['diagnostics'], promotion=config['promotion'],
            screening_seeds=config['evaluation']['screen_seeds'],
            comparison_seeds=config['evaluation']['comparison_seeds'],
            policy_seed=config['evaluation']['policy_seed'], search_seed=9100+i,
            workers=16, trials=16, search_seconds=4200, validation_seconds=1800,
            maximum_hours_from_creation=2.1, hourly_rate_usd=.965,
            storage_limit_bytes=4*1024**3, in_flight_bytes=16*128*1024**2+128*1024**2,
            evidence_note='Independent fixed-source development study. No automatic promotion; no holdout access.')
        write(folder / 'plan.json', plan)
        fleet['tasks'].append(dict(name=plan['name'], title=plan['title'], root=str(folder),
            plan_sha256=hashlib.sha256((folder / 'plan.json').read_bytes()).hexdigest()))
    write(out / 'fleet.json', fleet)
    print(json.dumps(fleet, indent=2))


def run(plan_path, metadata_path):
    plan, metadata = read(plan_path), read(metadata_path)
    root, campaign, source = (Path(plan[k]) for k in ('root', 'campaign', 'source'))
    if (root / 'state.json').exists():
        raise ValueError('Refusing to restart an existing worker study')
    sys.path.insert(0, str(source))
    from scripts.research_support import assert_snapshot, research_case, promotion
    from scripts.optimize_policy import versions, run_case, summarize
    from scripts.simulation_backend import identity
    expected = read(source.parent / 'manifest.json')
    assert_snapshot(source, expected)
    if versions() != plan['environment']:
        raise ValueError('Worker environment differs from frozen campaign environment')
    identity('fastsim')
    created = datetime.fromisoformat(metadata['createdAt'].replace('Z', '+00:00')).timestamp()
    hard_deadline = created + plan['maximum_hours_from_creation']*3600
    deadline = hard_deadline - 120
    state = dict(status='running', stage='screening', started_at=time.time(), pod_id=metadata['id'],
        deadline=hard_deadline, name=plan['name'], title=plan['title'], source=plan['snapshot'])
    control = root / 'control'
    control.mkdir(exist_ok=True)
    stopping = False
    storage_checked = 0

    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def pulse():
        nonlocal stopping, storage_checked
        now = time.time()
        primary = read(campaign / 'state.json')
        reason = None
        if now >= deadline:
            reason = 'worker deadline'
        elif stopping or (root / 'STOP').exists():
            reason = 'operator stop'
        elif primary['status'] != 'running' or primary['stage'] in ('final', 'done'):
            reason = 'primary campaign left development'
        elif now - (campaign / 'state.json').stat().st_mtime > 120:
            reason = 'primary supervisor heartbeat expired'
        if now - storage_checked > 30:
            storage_checked = now
            state['used_bytes'] = size(root)
            if state['used_bytes'] + plan['in_flight_bytes'] >= plan['storage_limit_bytes']:
                reason = 'worker storage limit'
        if reason:
            stopping = True
            state['stop_reason'] = reason
            (control / 'STOP').touch()
        (control / 'heartbeat').touch()
        state.update(updated_at=now, estimated_compute_usd=max(0, now-created)*plan['hourly_rate_usd'])
        state['estimated_compute_usd'] /= 3600
        write(root / 'state.json', state)

    pulse()
    request = dict(mode='focus', environment=plan['environment'], starting_config=plan['incumbent']['config'],
        variants=plan['variants'], paths=plan['paths'], max_dimensions=16,
        seeds=plan['screening_seeds'], policy_seed=plan['policy_seed'],
        engine='fastsim', diagnostics=plan['diagnostics'], trials=plan['trials'],
        workers=plan['workers'], case_timeout=3600, case_retries=2, case_cache=str(root / 'evaluation-cache'),
        search_seed=plan['search_seed'], deadline=min(time.time()+plan['search_seconds'], deadline-plan['validation_seconds']))
    write(root / 'request.json', request)
    with (root / 'search.console.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(source / 'scripts/research_study.py'),
            '--request', str(root / 'request.json'), '--out', str(root / 'study'), '--control', str(control)],
            cwd=source, stdout=log, stderr=subprocess.STDOUT)
        stop_since = None
        while process.poll() is None:
            pulse()
            if stopping:
                stop_since = stop_since or time.time()
                if time.time()-stop_since > 60:
                    process.terminate()
                    process.wait(timeout=15)
                    break
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(f'Screening process exited {process.returncode}; see search.console.log')
    report = read(root / 'study' / 'report.json') if (root / 'study' / 'report.json').exists() else {}
    ranked = report.get('ranking', [])
    baseline = next((t for t in ranked if t['id'] == 0), None)
    eligible = [t for t in ranked if t['id'] != 0 and baseline and
        t['summary'].get('worst_survival', 0) >= baseline['summary']['worst_survival']-plan['promotion']['maximum_worst_loss_seconds']]
    finalist = eligible[0] if eligible else None
    if finalist and not stopping and time.time()+120 < deadline:
        state['stage'] = 'python-validation'
        configs = [('incumbent', plan['incumbent']['config']), ('finalist', finalist['config'])]
        requests = {(label, seed): research_case(plan['snapshot'], plan['environment'], config, False,
            seed, plan['policy_seed'], plan['diagnostics'], 'python', infrastructure_retries=2)
            for label, config in configs for seed in plan['comparison_seeds']}
        rows = {'incumbent': [], 'finalist': []}
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            pending = {pool.submit(run_case, r, root / 'evaluation-cache', control, 3600,
                       runtime_root=source): key for key, r in requests.items()}
            while pending:
                pulse()
                for future in list(pending):
                    if future.done():
                        label, seed = pending.pop(future)
                        rows[label].append(future.result())
                state['validation_completed'] = len(requests)-len(pending)
                state['validation_total'] = len(requests)
                time.sleep(1)
        for group in rows.values():
            group.sort(key=lambda r: r['seed'])
        complete = all(r['status'] in ('horizon', 'extinct') for group in rows.values() for r in group)
        result = dict(engine='python', seeds=plan['comparison_seeds'], rows=rows, complete=complete,
            incumbent=plan['incumbent'], finalist=finalist,
            summaries={k: summarize(v, 3000) for k, v in rows.items()} if complete else {},
            decision=promotion(rows['incumbent'], rows['finalist'], plan['promotion']) if complete else None,
            note='Advisory comparison against this study\'s frozen incumbent; main supervisor retains promotion authority.')
        write(root / 'comparison.json', result)
    assert_snapshot(source, expected)
    state.update(status='inconclusive' if stopping else 'complete', stage='done', finished_at=time.time(),
                 finalist_id=finalist['id'] if finalist else None)
    pulse()
    write(root / 'state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--pod', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.campaign, args.out)
    else:
        try:
            run(args.plan, args.pod)
        except BaseException as error:
            root = Path(read(args.plan)['root'])
            write(root / 'state.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                finished_at=time.time()))
            raise


if __name__ == '__main__':
    main()

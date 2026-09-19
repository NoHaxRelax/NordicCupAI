"""Useful capture studies for spare shared workers, independent of LLM latency.

Every study pins its source, incumbent and fresh screen/confirmation panels.
Results are advisory and never promote a candidate or consume the final holdout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from research_dispatch import read, write, remote_case


def run_task(plan_path, queue):
    plan = read(plan_path)
    root, campaign, source = (Path(plan[k]) for k in ('root', 'campaign', 'source'))
    sys.path.insert(0, str(source))
    from scripts.research_support import research_case, promotion, assert_snapshot
    from scripts.optimize_policy import summarize
    control = root/'control'
    control.mkdir(exist_ok=True)
    state = dict(status='running', stage='screening', started_at=time.time(), pid=os.getpid(), name=plan['name'],
                 title=plan['title'], pod_id='shared four-Pod pool', deadline=plan['deadline'])
    def pulse():
        live = read(campaign/'state.json')
        if (live['stage'] in ('final', 'done') or live['status'] not in ('running', 'recovering')
                or time.time()-(campaign/'state.json').stat().st_mtime > 180
                or time.time() >= plan['deadline']):
            (control/'STOP').touch()
        (control/'heartbeat').touch()
        state['updated_at'] = time.time()
        write(root/'state.json', state)
    request = dict(mode='focus', starting_config=plan['incumbent']['config'], variants=plan['variants'],
        paths=plan['paths'], max_dimensions=16, environment=plan['environment'],
        policy_seed=plan['policy_seed'], seeds=plan['screening_seeds'], search_seed=plan['search_seed'],
        workers=40, deadline=min(time.time()+7200, plan['deadline']-3600), trials=24,
        case_cache=str(root/'evaluation-cache'), diagnostics=plan['diagnostics'], engine='fastsim',
        case_timeout=7200, case_retries=2)
    write(root/'request.json', request)
    pulse()
    with (root/'screen.console.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('research_async_study.py')),
            '--source', str(source), '--queue', str(queue), '--request', str(root/'request.json'),
            '--out', str(root/'study'), '--control', str(control), '--priority', '40'],
            stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            pulse()
            time.sleep(3)
    if process.returncode:
        raise RuntimeError('Background screening failed; inspect screen.console.log')
    ranked = read(root/'study/report.json', {}).get('ranking', [])
    anchor = next((r for r in ranked if r['id'] == 0), None)
    finalists = [r for r in ranked if r['id'] and anchor and
        r['summary']['worst_survival'] >= anchor['summary']['worst_survival']-plan['promotion']['maximum_worst_loss_seconds']]
    if finalists and not (control/'STOP').exists():
        finalist = finalists[0]
        state['stage'] = 'python-validation'
        rows = dict(incumbent=[], finalist=[])
        with ThreadPoolExecutor(max_workers=16) as pool:
            pending = {}
            for label, config in [('incumbent', plan['incumbent']['config']), ('finalist', finalist['config'])]:
                for seed in plan['comparison_seeds']:
                    request = research_case(plan['snapshot'], plan['environment'], config, False, seed,
                        plan['policy_seed'], plan['diagnostics'], 'python', infrastructure_retries=2)
                    future = pool.submit(remote_case, queue, request, root/'evaluation-cache', control,
                        7200, runtime_root=source, priority=20)
                    pending[future] = label
            state['validation_total'] = len(pending)
            while pending:
                pulse()
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    rows[pending.pop(future)].append(future.result())
                state['validation_completed'] = sum(map(len, rows.values()))
        for group in rows.values():
            group.sort(key=lambda r: r['seed'])
        complete = all(r['status'] in ('horizon', 'extinct') for group in rows.values() for r in group)
        write(root/'comparison.json', dict(engine='python', seeds=plan['comparison_seeds'], rows=rows,
            complete=complete, incumbent=plan['incumbent'], finalist=finalist,
            summaries={k: summarize(v, 3000) for k, v in rows.items()} if complete else {},
            decision=promotion(rows['incumbent'], rows['finalist'], plan['promotion']) if complete else None,
            note='Advisory independent paired evidence; main supervisor alone can promote.'))
    assert_snapshot(source, read(source.parent/'manifest.json'))
    state.update(status='inconclusive' if (control/'STOP').exists() else 'complete', stage='done', finished_at=time.time())
    pulse()


def make_plan(campaign, root, index, definition):
    state, cfg = read(campaign/'state.json'), read(campaign/'config.json')
    incumbent = state['best']
    source = campaign/'snapshots'/incumbent['snapshot']/'code'
    # Catalog obtained from this immutable candidate, not the producer interpreter.
    catalog_path = root/'catalog.json'
    request_path = root/'catalog-request.json'
    write(request_path, dict(mode='catalog'))
    subprocess.run([sys.executable, '-B', str(source/'scripts/research_study.py'), '--request', str(request_path),
                    '--out', str(root), '--control', str(root/'control')], check=True, stdout=subprocess.DEVNULL)
    selectable = {s['path'] for s in read(catalog_path)['inventory'] if s['tunable']}
    definition = copy.deepcopy(definition)
    definition['paths'] = [p for p in definition['paths'] if p in selectable]
    if not definition['paths']:
        raise ValueError('Capture topic has no supported tuning dimensions')
    # Existing policy features can change between generations. Unsupported
    # variants are skipped explicitly rather than modifying accepted source.
    known = set(selectable) | {s['path'] for s in read(catalog_path)['inventory']}
    definition['variants'] = [v for v in definition['variants'] if set(v['overrides']) <= known]
    first = 1000000 + index*32
    seeds = list(range(first, first+4)), list(range(first+8, first+16))
    forbidden = set(cfg['evaluation']['holdout_seeds'] + cfg['evaluation'].get('retired_holdout_seeds', []))
    for panel in cfg['evaluation'].get('seed_panels', {}).values():
        for group in panel.values():
            forbidden.update(group)
    if (set(seeds[0]) | set(seeds[1])) & forbidden:
        raise ValueError('Background seed panel overlaps reserved research data')
    protocol = read(campaign/'protocol.json')
    plan = dict(**definition, campaign=str(campaign), root=str(root), source=str(source),
        snapshot=incumbent['snapshot'], incumbent=incumbent, environment=protocol['environment'],
        screening_seeds=seeds[0], comparison_seeds=seeds[1], search_seed=95000+index*100003,
        diagnostics=cfg['diagnostics'], promotion=cfg['promotion'], policy_seed=cfg['evaluation']['policy_seed'],
        trials=24, workers=40, major=state['major'], index=index, deadline=time.time()+14400,
        note='Fresh-panel capture evidence during reviews and straggler tails; no automatic promotion.')
    write(root/'plan.json', plan)
    return plan


def produce(campaign, root, queue):
    from research_parallel import definitions
    initial = read(campaign/'state.json')['initial_snapshot']
    sys.path.insert(0, str(campaign/'snapshots'/initial/'code'))
    from scripts.research_io import directory_bytes
    root.mkdir(parents=True, exist_ok=True)
    index = read(root/'fleet.json', dict(version=2, campaign=str(campaign), tasks=[]))
    active = {}
    used_bytes, last_storage_check = 0, 0
    for task in index['tasks']:
        status = read(Path(task['root'])/'state.json', {})
        pid = status.get('pid')
        if status.get('status') == 'running' and pid and Path(f'/proc/{pid}').exists():
            active[task['name']] = pid
    while True:
        state = read(campaign/'state.json')
        if state['stage'] in ('final', 'done') or state['status'] not in ('running', 'recovering'):
            for task in index['tasks']:
                control = Path(task['root'])/'control'
                if control.exists():
                    (control/'STOP').touch()
            break
        for name, process in list(active.items()):
            alive = Path(f'/proc/{process}').exists() if isinstance(process, int) else process.poll() is None
            if not alive:
                del active[name]
        if time.time()-(campaign/'state.json').stat().st_mtime > 180:
            time.sleep(5)
            continue
        # Start follow-on studies before older studies reach their final slow
        # maps. Outstanding jobs, rather than whole-study completion, determine
        # when useful work needs replenishment.
        this_major = sum(t.get('major') == state['major'] for t in index['tasks'])
        if time.time()-last_storage_check > 60:
            used_bytes = directory_bytes(root)
            last_storage_check = time.time()
        settings = read(queue/'config.json')
        pods = len(settings['pods'])
        target_jobs = int(pods*settings['slots_per_pod']*1.5)
        queued_jobs = len(list((queue/'jobs').glob('*.json')))
        reserved_jobs = 0
        while (len(active) < max(6, 3*pods) and queued_jobs+reserved_jobs < target_jobs
                and used_bytes < 180*1024**3):
            number = len(index['tasks'])
            definition = definitions()[number % 3]
            name = f'g{state["major"]}-{number:03d}-{definition["name"]}'
            folder = root/name
            if folder.exists():
                raise ValueError('Refusing to replace a background study')
            plan = make_plan(campaign, folder, number, definition)
            task = dict(name=name, title=plan['title'], root=str(folder), major=state['major'],
                        plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest())
            index['tasks'].append(task)
            write(root/'fleet.json', index)
            with (folder/'console.log').open('ab') as log:
                active[name] = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    'task', '--plan', str(folder/'plan.json'), '--queue', str(queue)], stdout=log, stderr=subprocess.STDOUT)
            this_major += 1
            reserved_jobs += 40
        write(root/'producer.json', dict(updated_at=time.time(), pid=__import__('os').getpid(),
            major=state['major'], active=list(active), studies=len(index['tasks']), used_bytes=used_bytes,
            target_jobs=target_jobs, queued_jobs=queued_jobs, reserved_jobs=reserved_jobs))
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['producer', 'task'])
    p.add_argument('--campaign', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    a = p.parse_args()
    if a.mode == 'producer':
        produce(a.campaign, a.root, a.queue)
    else:
        try:
            run_task(a.plan, a.queue)
        except BaseException as exc:
            write(a.plan.parent/'state.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                                 finished_at=time.time()))
            raise


if __name__ == '__main__':
    main()

"""Resumable feature comparisons and block evolutionary policy search.

No third-party optimizer is required. Search uses full natural-predator games,
paired training maps and separate validation. Importing this module starts no games.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_case import write_json
from scripts.simulation_backend import identity as engine_identity

# Selected once in main(); 'trapping' keeps the original search unchanged.
SPACES = dict(trapping='models.experiment_config', notrap='models.notrap_config')
SPACE_MODES = dict(trapping=['trapping'], notrap=['orchard_evasion', 'expert_harvest'])
SPACE_BASELINES = dict(trapping='models.core.EntrapmentPolicy',
                       notrap='models.orchard_evasion_policy.OrchardEvasionPolicy'
                              '+models.optimization_policy.OptimizationPolicy')
SPACE = 'trapping'


def space():
    import importlib
    return importlib.import_module(SPACES[SPACE])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@contextmanager
def study_lock(path):
    """An OS lock is released automatically after a crash; no stale PID deletion."""
    handle = path.open('a+b')
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError('Another optimizer is using this output directory') from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == 'nt':
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def source_manifest():
    paths = [ROOT/'run.py', ROOT/'requirements.txt']
    for name in ('models', 'src', 'scripts'):
        paths.extend((ROOT/name).rglob('*.py'))
    paths.extend((ROOT/'scripts').rglob('*.sh'))
    paths.extend((ROOT/'models').rglob('*.json'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def versions():
    # Include transitive dependencies too; equal top-level pins alone are not
    # enough to reproduce a Pydantic/geometry/numerical runtime.
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        name = line.strip().split('==')[0]
        if not name or name.startswith('#'):
            continue
        try:
            packages[name.lower().replace('_', '-')] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name.lower().replace('_', '-')] = 'not installed'
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                os=sys.platform, architecture=platform.machine(), packages=packages)


def initial_trials(baseline, features):
    plan = [dict(label='current_baseline', baseline=True, config=copy.deepcopy(baseline)),
            dict(label='configurable_control', baseline=False, config=copy.deepcopy(baseline))]
    if not features:
        # A space without optional feature blocks goes straight to scalar probes
        # rather than fabricating an empty 'features' section its schema rejects.
        return plan
    for feature in features:
        config = copy.deepcopy(baseline)
        config['features'][feature] = True
        plan.append(dict(label=f'only_{feature}', baseline=False, config=config))
    all_on = copy.deepcopy(baseline)
    all_on['features'] = dict.fromkeys(features, True)
    plan.append(dict(label='all_features', baseline=False, config=all_on))
    for feature in features:
        config = copy.deepcopy(all_on)
        config['features'][feature] = False
        plan.append(dict(label=f'without_{feature}', baseline=False, config=config))
    return plan


def sample(spec, current, rng):
    if spec['kind'] == 'bool':
        return not current
    if spec['nullable'] and current is not None and rng.random() < .15:
        return None
    low, high = spec['low'], spec['high']
    if low > high:
        raise ValueError(f"Empty search range: {spec['path']}")
    for _ in range(30):
        if current is None or rng.random() < .3:
            value = rng.uniform(low, high)
        elif low > 0:
            value = math.exp(math.log(max(low, min(high, current)))+rng.gauss(0, .3))
        else:
            value = current+rng.gauss(0, max(.001, (high-low)*.2))
        value = max(low, min(high, value))
        value = int(round(value)) if spec['kind'] == 'int' else float(value)
        if value != current:
            return value
    return high if current != high else low


def activate(config, path):
    """Enable the owning optional block when probing one of its parameters."""
    if path.startswith('safety.'):
        name = path.split('.')[1]
        if name == 'guide_geometry_interval':
            feature = 'refresh_guide_geometry'
        elif name.startswith('guide_') or name == 'minimum_workers':
            feature = 'role_budget'
        elif name.startswith('emergency_') or name == 'young_age':
            feature = 'emergency_reproduction'
        elif name.startswith('conservation_') or name == 'idle_turn_fraction':
            feature = 'late_conservation'
        elif name == 'memory_seconds':
            feature = 'escape_memory'
        elif name == 'danger_radius':
            feature = 'consistent_escape'
        elif name == 'trap_radius':
            feature = 'trap_exclusion'
        elif name.startswith(('shared_', 'uncertainty_', 'lure_')):
            feature = 'shared_danger'
        else:
            feature = 'safe_steering'
        config['features'][feature] = True
    parts = path.split('.')
    if parts[0] == 'planner' and len(parts) == 2 and parts[1] != 'enabled':
        config['planner']['enabled'] = True
    branch = config
    for name in parts[:-1]:
        branch = branch[name]
        if isinstance(branch, dict) and 'enabled' in branch and name != 'planner':
            branch['enabled'] = True
    # These existing controller settings are masked by the corresponding feature.
    if 'features' not in config:
        return
    if path == 'expert.perception.predator_danger_radius':
        config['features']['consistent_escape'] = False
    if path == 'expert.memory.predator_escape_seconds':
        config['features']['escape_memory'] = False


def propose(index, trials, initial, specs, search_seed):
    module = space()
    get, put, repair, validate, leaves = module.get, module.put, module.repair, module.validate, module.leaves
    if index < len(initial):
        trial = copy.deepcopy(initial[index])
        trial['changed'] = [p for p, value in leaves(trial['config'])
                            if value != get(initial[0]['config'], p)]
        return trial
    rng = random.Random(search_seed + index*1000003)
    ranked = sorted((t for t in trials if t.get('summary', {}).get('rank') is not None),
                    key=lambda t: t['summary']['rank'], reverse=True)
    parent = ranked[0] if index-len(initial) < len(specs) else rng.choice(ranked[:5])
    offset = index-len(initial)
    if offset < len(specs):
        chosen = [specs[offset]]
        label = f"parameter_{specs[offset]['path']}"
    else:
        blocks = sorted({s['block'] for s in specs})
        block = blocks[(offset-len(specs)) % len(blocks)]
        options = [s for s in specs if s['block'] == block]
        chosen = rng.sample(options, min(len(options), rng.randint(1, 4)))
        label = f'evolve_{block}'
    for attempt in range(50):
        config = copy.deepcopy(parent['config'])
        # Recombine one complete, related block rather than arbitrary scalar soup.
        if offset >= len(specs) and len(ranked) > 1 and rng.random() < .25:
            donor = rng.choice(ranked[:5])
            block = rng.choice(sorted({s['block'] for s in specs}))
            for spec in specs:
                if spec['block'] == block:
                    put(config, spec['path'], get(donor['config'], spec['path']))
        for spec in chosen:
            activate(config, spec['path'])
            put(config, spec['path'], sample(spec, get(config, spec['path']), rng))
        if offset >= len(specs) and rng.random() < .3:
            feature = rng.choice(list(config['features']))
            config['features'][feature] = not config['features'][feature]
        config = repair(config)
        try:
            validate(config)
        except ValueError:
            continue
        return dict(label=label, baseline=False, config=config, parent=parent['id'],
                    changed=[p for p, _ in leaves(config)
                             if get(config, p) != get(parent['config'], p)])
    raise RuntimeError(f'Could not construct a valid candidate for {label}')


def summarize(results, seconds, enabled_count=0, objective='survival'):
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        return dict(rank=None, errors=len(errors), error=errors[0].get('error', 'case failed'))
    if not results or any(r['status'] not in ('horizon', 'extinct') for r in results):
        raise ValueError('Cannot rank interrupted or incomplete trials')
    survival = [min(seconds, r['sim_time']) for r in results]
    average, worst = statistics.mean(survival), min(survival)
    scores = [r['score'] for r in results]
    score, worst_score = statistics.mean(scores), min(scores)
    modes = {}
    for mode in sorted({r['mode'] for r in results}):
        subset = [r for r in results if r['mode'] == mode]
        modes[mode] = dict(mean_survival=statistics.mean(r['sim_time'] for r in subset),
                           mean_score=statistics.mean(r['score'] for r in subset),
                           survived=sum(r['sim_time'] >= seconds-1e-6 for r in subset), cases=len(subset))
    # Reaching the horizon is roughly a 1-in-60 lottery given the food economy,
    # so with --objective score every pod reports 0 horizon games and survival
    # cannot discriminate between candidates; score still separates them. Rank
    # stays a comparable 3-tuple either way so callers never need to branch on it.
    if objective == 'score':
        rank = [score+.25*worst_score, average/seconds, -enabled_count]
    else:
        rank = [average/seconds+.25*worst/seconds, score, -enabled_count]
    return dict(rank=rank,
                mean_survival=average, worst_survival=worst, mean_score=score, worst_score=worst_score,
                survived=sum(t >= seconds-1e-6 for t in survival), cases=len(results),
                wall_seconds=sum(r['wall_seconds'] for r in results), modes=modes, errors=0)


def stop_process(process):
    """On Linux also reap the policy child when an evaluator wedges or times out."""
    if os.name == 'nt':
        if process.poll() is None:
            process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if os.name != 'nt':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def case_request(trial, seed, mode, protocol):
    # protocol['engine']/['engine_build'] flow into digest(protocol) below, so a case
    # computed under one engine can never satisfy a request under another: cache
    # identity, not just the executing evaluator, is engine-specific.
    key = dict(protocol=digest(protocol), seed=seed, mode=mode, seconds=protocol['seconds'],
              policy_seed=protocol['policy_seed'], baseline=trial['baseline'],
              config=None if trial['baseline'] else trial['config'])
    return dict(case_id=digest(key), seed=seed, mode=mode, seconds=protocol['seconds'],
                policy_seed=protocol['policy_seed'], baseline=trial['baseline'], config=trial['config'],
                engine=protocol.get('engine', 'python'), engine_build=protocol.get('engine_build'))


def retryable_case_error(result):
    if result.get('status') != 'error':
        return False
    error = str(result.get('error', ''))
    return (error == 'Wall-time limit reached' or 'ConnectionResetError' in error
            or 'Temporary failure' in error or 'FileNotFoundError' in error
            or error.startswith(('Worker exited -9;', 'Worker exited -15;')))


def completed_result(folder, case_id, *, skip_retryable_errors=False):
    # Recover finished games even if the coordinator crashed before copying the result.
    paths = [folder/'result.json', *sorted(folder.glob('attempt-*/result.json'))]
    for path in paths:
        if not path.exists():
            continue
        result = read_json(path)
        if result.get('case_id') != case_id:
            raise RuntimeError(f'Cached case identity mismatch: {path}')
        if skip_retryable_errors and retryable_case_error(result):
            continue
        if result['status'] in ('horizon', 'extinct', 'error'):
            return result
    return None


def run_case(request, out, control, timeout, *, runtime_root=None):
    retries = request.get('infrastructure_retries', 0)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError('Invalid infrastructure retry allowance')
    folder = out/'cases'/request['case_id']
    cached = completed_result(folder, request['case_id'], skip_retryable_errors=retries > 0)
    if cached is not None:
        return cached
    failures = []
    if retries:
        for path in sorted(folder.glob('attempt-*/result.json')):
            previous = read_json(path)
            if previous.get('case_id') != request['case_id']:
                raise RuntimeError('Retry artifact has a different case identity')
            if retryable_case_error(previous):
                failures.append(previous)
    if len(failures) > retries:
        return failures[-1]
    for number in range(len(failures), retries+1):
        # Give a slow valid game more wall time, never select retries by score.
        result = _run_case_once(request, out, control, timeout*(2**min(number, 1)),
                                runtime_root=runtime_root, skip_retryable_errors=retries > 0)
        if not retryable_case_error(result) or number >= retries or (control/'STOP').exists():
            return result
    raise AssertionError('Unreachable retry state')


def _run_case_once(request, out, control, timeout, *, runtime_root=None, skip_retryable_errors=False):
    # Research comparisons may select a verified immutable source snapshot.
    runtime_root = Path(runtime_root) if runtime_root is not None else ROOT
    folder = out/'cases'/request['case_id']
    cached = completed_result(folder, request['case_id'], skip_retryable_errors=skip_retryable_errors)
    if cached is not None:
        return cached
    if (control/'STOP').exists():
        return dict(request, status='interrupted')
    attempt = folder/f'attempt-{uuid.uuid4().hex}'
    attempt.mkdir(parents=True)
    write_json(attempt/'request.json', request)
    write_json(folder/'active.json', dict(attempt=str(attempt), mode=request['mode'], seed=request['seed']))
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy')
    for key in list(env):
        if key.endswith(('_API_KEY', '_ACCESS_TOKEN')):
            env.pop(key, None)
    command = [sys.executable, '-u', str(runtime_root/'scripts/experiment_case.py'),
               '--request', str(attempt/'request.json'), '--output', str(attempt/'result.json'),
               '--stop-file', str(control/'STOP')]
    start = time.monotonic()
    with (attempt/'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=runtime_root, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        timed_out, stopping_since = False, None
        try:
            while process.poll() is None:
                if (control/'STOP').exists():
                    stopping_since = stopping_since or time.monotonic()
                    if time.monotonic()-stopping_since > 15.:
                        stop_process(process)
                        break
                if timeout > 0 and time.monotonic()-start > timeout:
                    timed_out = True
                    stop_process(process)
                    break
                time.sleep(.25)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.poll() is None:
                stop_process(process)
    result_path = attempt/'result.json'
    if result_path.exists() and not timed_out:
        result = read_json(result_path)
    elif (control/'STOP').exists():
        result = dict(case_id=request['case_id'], seed=request['seed'], mode=request['mode'],
                      status='interrupted', wall_seconds=time.monotonic()-start)
    else:
        result = dict(case_id=request['case_id'], seed=request['seed'], mode=request['mode'], status='error',
                      error='Wall-time limit reached' if timed_out else f'Worker exited {process.returncode}; see {attempt / "console.log"}',
                      wall_seconds=time.monotonic()-start)
    if result['status'] in ('horizon', 'extinct', 'error'):
        if not result_path.exists():
            write_json(result_path, result)
        write_json(folder/'result.json', result)
    return result


def evaluate_batch(trials, seeds, protocol, args, control, deadline, on_complete=None, *, request_factory=None):
    """One global game queue across candidates; no nested worker pools.

    Save each finished candidate immediately. Deduplicate identical cases across
    candidates before submission, so they cannot race to overwrite cache files.
    """
    requests, owners, expected, results, completed = {}, {}, {}, {}, {}
    for trial in trials:
        tid = trial['id']
        expected[tid], results[tid] = [], {}
        for seed in seeds:
            for mode in protocol['modes']:
                request = (request_factory(trial, seed, mode) if request_factory is not None
                           else case_request(trial, seed, mode, protocol))
                cid = request['case_id']
                requests[cid] = request
                owners.setdefault(cid, []).append(tid)
                expected[tid].append(cid)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    cache = getattr(args, 'case_cache', args.out)
    pending = {pool.submit(run_case, request, cache, control, args.case_timeout): request
               for request in requests.values()}
    next_message = 0.
    start = time.monotonic()
    try:
        while pending:
            (control/'heartbeat').touch()
            if time.monotonic() >= deadline:
                (control/'STOP').touch()
            done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                request = pending.pop(future)
                result = future.result()
                cid = request['case_id']
                for tid in owners[cid]:
                    results[tid][cid] = result
                print(f"  trials {owners[cid]} seed {request['seed']}: {result['status']}, "
                      f"{result.get('sim_time', 0):.1f}/{args.seconds:g}s, score {result.get('score', 0):.2f}", flush=True)
            for trial in trials:
                tid = trial['id']
                if tid in completed or len(results[tid]) != len(expected[tid]):
                    continue
                rows = [results[tid][cid] for cid in expected[tid]]
                if any(r['status'] == 'interrupted' for r in rows):
                    continue
                completed[tid] = rows
                if on_complete is not None:
                    on_complete(trial, rows)
            if time.monotonic() >= next_message and pending:
                messages = []
                for request in pending.values():
                    active = cache/'cases'/request['case_id']/'active.json'
                    if not active.exists():
                        continue
                    progress = Path(read_json(active)['attempt'])/'progress.json'
                    if progress.exists():
                        row = read_json(progress)
                        messages.append(f"{request['mode']} seed {request['seed']}: "
                                        f"{row['sim_time']:.0f}/{args.seconds:g}s, alive {row['alive']}")
                print(f'  batch: {len(requests)-len(pending)}/{len(requests)} games finished; '
                      f'{len(completed)}/{len(trials)} candidates; '
                      f'{max(0., deadline-time.monotonic())/3600:.2f}h left in phase. '
                      + ('; '.join(messages[:6]) if messages else 'starting workers...'), flush=True)
                write_json(args.out/'progress.json', dict(
                    phase=getattr(args, 'active_phase', args.phase), workers=args.workers,
                    completed_games=len(requests)-len(pending), queued_or_active_games=len(pending),
                    batch_candidates=len(trials), completed_candidates=len(completed),
                    batch_wall_seconds=time.monotonic()-start,
                    phase_seconds_remaining=max(0., deadline-time.monotonic()), active=messages))
                next_message = time.monotonic()+30.
    except KeyboardInterrupt:
        args.interrupted = True
        print('\nStopping active games; completed games remain saved.', flush=True)
        (control/'STOP').touch()
    except BaseException:
        (control/'STOP').touch()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return completed


def evaluate_trial(trial, seeds, protocol, args, control, deadline):
    return evaluate_batch([trial], seeds, protocol, args, control, deadline).get(trial['id'])


def write_report(out, state, protocol):
    ranked = sorted((t for t in state['trials'] if t.get('summary', {}).get('rank') is not None),
                    key=lambda t: t['summary']['rank'], reverse=True)
    lines = ['# Policy search results', '',
             'Training results only. Run the separate validation phase before accepting an improvement.', '',
             f"Rank: {protocol['objective']}.", '',
             '| Trial | Variant | Mean survival | Worst survival | Mean score |',
             '| --- | --- | ---: | ---: | ---: |']
    for t in ranked:
        s = t['summary']
        lines.append(f"| {t['id']} | {t['label']} | {s['mean_survival']:.1f} | {s['worst_survival']:.1f} | {s['mean_score']:.2f} |")
    failures = [t for t in state['trials'] if t.get('summary', {}).get('errors')]
    lines += ['', f'Failed trials: {len(failures)}. Errors are retained and never treated as good low-population runs.', '']
    by_label = {t['label']: t['summary'] for t in ranked}
    effects = {}
    lines += ['## Feature comparisons', '',
              'Training-map differences; interactions and simulation variation can change these effects.', '',
              '| Addition | Alone: survival / score delta | Within all additions: survival / score delta |',
              '| --- | ---: | ---: |']
    def delta(left, right):
        if left not in by_label or right not in by_label:
            return None
        return {key: by_label[left][key]-by_label[right][key] for key in ('mean_survival', 'mean_score')}
    def display(value):
        return 'pending or failed' if value is None else f"{value['mean_survival']:+.1f}s / {value['mean_score']:+.2f}"
    for feature in protocol['parameter_inventory']:
        if not feature['path'].startswith('features.'):
            continue
        name = feature['path'].split('.')[1]
        alone = delta(f'only_{name}', 'configurable_control')
        combined = delta('all_features', f'without_{name}')
        effects[name] = dict(alone=alone, within_all_additions=combined)
        lines.append(f'| {name} | {display(alone)} | {display(combined)} |')
    write_json(out/'feature-effects.json', effects)
    lines.append('')
    if ranked:
        best = ranked[0]
        write_json(out/'best.json', dict(id=best['id'], baseline=best['baseline'], config=best['config'],
                                       summary=best['summary'], protocol_hash=digest(protocol)))
        baseline = state['trials'][0].get('summary', {})
        if baseline.get('rank') is not None:
            lines += [f"Best training change versus current baseline: {best['summary']['mean_survival']-baseline['mean_survival']:+.1f} seconds mean survival; "
                      f"{best['summary']['mean_score']-baseline['mean_score']:+.2f} mean score.", '']
    touched = sorted({path for t in state['trials'] if t.get('summary', {}).get('rank') is not None for path in t.get('changed', [])})
    tunable = {s['path'] for s in protocol['parameter_inventory'] if s['tunable']}
    write_json(out/'coverage.json', dict(evaluated_config_changes=touched,
                                      not_yet_varied=sorted(tunable-set(touched)),
                                      completed_trials=sum('summary' in t for t in state['trials'])))
    # Rendering must never take down a search that has already produced results.
    try:
        from scripts.research_progress_plot import render
        if render(state, protocol['seconds'], out/'progress.png'):
            lines += ['## Improvement over time', '',
                      'Best-so-far against completed trials and against cumulative game '
                      'wall-hours. A flat stretch means trials completed without improving.',
                      '', '![Search progress](progress.png)', '']
    except Exception as exc:
        lines += ['', f'Progress plot unavailable: {type(exc).__name__}.', '']
    temporary = out/'report.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(out/'report.md')


def search(args, protocol, state, baseline, specs, features, control, deadline):
    if (args.out/'validation-selection.json').exists():
        raise RuntimeError('Validation selection is frozen. Use a new study for further tuning.')
    if any(t.get('summary', {}).get('errors') for t in state['trials'][:2]):
        raise RuntimeError('A saved control run failed. Inspect its error; fix it and start a new output directory.')
    initial = initial_trials(baseline, features)
    numeric_specs = [s for s in specs if s['tunable'] and not s['path'].startswith('features.')]
    # Prioritize survival/role behavior before expensive mapping refinements.
    order = {'evasion': 0, 'bystander': 0, 'safety': 1, 'trapping_orchard': 2, 'orchard': 2,
             'guide': 3, 'navigator': 4, 'expert': 5, 'planner': 6}
    # Round-robin blocks so a time-limited night reaches every policy subsystem.
    blocks = {}
    for spec in sorted(numeric_specs, key=lambda s: (order.get(s['path'].split('.')[0], 99), s['path'])):
        blocks.setdefault(spec['block'], []).append(spec)
    numeric_specs = [items[i] for i in range(max(map(len, blocks.values()), default=0))
                     for items in blocks.values() if i < len(items)]
    total = args.trials or len(initial)+len(numeric_specs)+args.evolution_trials
    print(f'Plan: {len(initial)} feature/control trials, {len(numeric_specs)} parameter probes, then evolution. This session cap: {total}.', flush=True)

    def save_trial(trial, results):
        trial['case_ids'] = [r['case_id'] for r in results]
        trial['summary'] = summarize(results, args.seconds, sum(trial['config'].get('features', {}).values()),
                                     args.objective)
        write_json(args.out/'study.json', state)
        write_report(args.out, state, protocol)
        if trial['summary'].get('errors') and trial['id'] < 2:
            raise RuntimeError(f"Control run failed; fix the cause before searching: {trial['summary']['error']}")

    batch_size = args.batch_candidates or max(1, math.ceil(2*args.workers/len(args.train_seeds)))
    # Control failures gate everything else; finish all ablations before mutations.
    for lower, upper in ((0, min(2, total)), (2, min(len(initial), total)), (len(initial), total)):
        while time.monotonic() < deadline and not (control/'STOP').exists():
            batch = [t for t in state['trials'][lower:upper] if 'summary' not in t][:batch_size]
            while len(batch) < batch_size and len(state['trials']) < upper:
                index = len(state['trials'])
                trial = dict(propose(index, state['trials'], initial, numeric_specs, args.search_seed), id=index)
                state['trials'].append(trial)
                batch.append(trial)
            if not batch:
                break
            write_json(args.out/'study.json', state)
            print(f"Candidates {[t['id'] for t in batch]} of {total}; {args.workers} game slots", flush=True)
            evaluate_batch(batch, args.train_seeds, protocol, args, control, deadline, save_trial)
    write_report(args.out, state, protocol)


def refine(args, protocol, state, control, deadline):
    """Retest a frozen shortlist on more training maps before opening holdout."""
    if len(state['trials']) < 2 or any(t.get('summary', {}).get('rank') is None for t in state['trials'][:2]):
        raise ValueError('Complete both screening controls successfully before refinement')
    plan_path = args.out/'refinement-plan.json'
    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        ranked = sorted((t for t in state['trials'] if t.get('summary', {}).get('rank') is not None),
                        key=lambda t: t['summary']['rank'], reverse=True)
        if not ranked:
            raise ValueError('No completed training candidates to refine')
        ids, seen = [], set()
        for trial in ranked:
            key = digest(dict(baseline=trial['baseline'], config=trial['config']))
            if key not in seen:
                ids.append(trial['id'])
                seen.add(key)
            if len(ids) >= args.finalists:
                break
        ids = sorted(set([0, 1]+ids))
        plan = dict(trial_ids=ids, screening_winner=ranked[0]['id'], seeds=args.refine_seeds)
        write_json(plan_path, plan)
    trials = [state['trials'][i] for i in plan['trial_ids']]

    def save_trial(trial, results):
        trial['refinement'] = summarize(results, args.seconds, sum(trial['config']['features'].values()),
                                        args.objective)
        trial['refinement_case_ids'] = [r['case_id'] for r in results]
        write_json(args.out/'study.json', state)
        if trial['id'] < 2 and trial['refinement'].get('errors'):
            raise RuntimeError('A refinement control failed; inspect saved case logs.')

    evaluate_batch(trials, args.refine_seeds, protocol, args, control, deadline, save_trial)
    complete = all('refinement' in t for t in trials)
    if complete:
        eligible = [t for t in trials if t['refinement'].get('rank') is not None]
        winner = max(eligible, key=lambda t: t['refinement']['rank'])
        summary = winner['refinement']
    else:
        # Do not favor candidates just because their games completed sooner.
        winner = state['trials'][plan['screening_winner']]
        if winner.get('refinement', {}).get('errors'):
            winner = state['trials'][0]
        summary = winner['summary']
    selection = dict(id=winner['id'], baseline=winner['baseline'], config=winner['config'], summary=summary,
                     protocol_hash=digest(protocol), refinement_complete=complete,
                     selection_basis='refinement' if complete else 'screening fallback (refinement incomplete)')
    write_json(args.out/'refinement.json', dict(plan=plan, complete=complete, selection=selection,
        outcomes={str(t['id']): t.get('refinement') for t in trials}))
    write_json(args.out/'best.json', selection)
    print(f"Final training selection: {winner['id']} ({selection['selection_basis']})", flush=True)


def paired_intervals(pairs):
    """Paired bootstrap over complete held-out maps; descriptive, not a guarantee."""
    if len(pairs) < 2 or any(p.get('error') for p in pairs):
        return None
    rng = random.Random(8751)
    summary = {}
    for key in ('survival_delta', 'score_delta'):
        values = [p[key] for p in pairs]
        samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
        summary[key] = dict(mean=statistics.mean(values), median=statistics.median(values),
                            bootstrap_95_percent=[samples[50], samples[1949]],
                            wins=sum(v > 0 for v in values), losses=sum(v < 0 for v in values),
                            ties=sum(v == 0 for v in values), maps=len(values))
    return summary


def validation(args, protocol, state, control, deadline):
    if not (args.out/'best.json').exists():
        raise ValueError('Complete at least the baseline before validation')
    selection_path = args.out/'validation-selection.json'
    if selection_path.exists():
        selection = read_json(selection_path)
    else:
        selection = read_json(args.out/'best.json')
        write_json(selection_path, selection)
    # Freeze the chosen candidate before examining any validation outcome.
    trials = [dict(state['trials'][0], id='baseline'), dict(selection, id='candidate')]
    outcomes = {}
    def save_trial(trial, results):
        outcomes[trial['id']] = dict(summary=summarize(results, args.seconds, objective=args.objective), results=results)
        write_json(args.out/'validation-progress.json', dict(selection=selection, outcomes=outcomes))
    evaluate_batch(trials, args.validation_seeds, protocol, args, control, deadline, save_trial)
    if len(outcomes) != 2:
        print('Validation incomplete; finished games are saved for resume.', flush=True)
        return False
    pairs = []
    refs = {(r['mode'], r['seed']): r for r in outcomes['baseline']['results']}
    for row in outcomes['candidate']['results']:
        ref = refs[row['mode'], row['seed']]
        if row['status'] == 'error' or ref['status'] == 'error':
            pairs.append(dict(mode=row['mode'], seed=row['seed'], error=True))
        else:
            pairs.append(dict(mode=row['mode'], seed=row['seed'], survival_delta=row['sim_time']-ref['sim_time'],
                              score_delta=row['score']-ref['score']))
    intervals = paired_intervals(pairs)
    write_json(args.out/'validation.json', dict(selection=selection, outcomes=outcomes, paired_deltas=pairs,
        paired_summary=intervals, complete=True,
        note='Held-out comparison, not a guarantee. Do not tune against these seeds or silently select a new winner.'))
    lines = ['# Held-out predator evaluation', '', f"Selected training trial: {selection['id']}.", '',
             f"Selection basis: {selection.get('selection_basis', 'screening')}.", '',
             '| Policy | Mean survival | Worst survival | Mean score | Full-horizon maps |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label, outcome in outcomes.items():
        s = outcome['summary']
        if s.get('errors'):
            lines += [f'| {label} | ERROR | ERROR | ERROR | ERROR |']
        else:
            lines += [f"| {label} | {s['mean_survival']:.1f} | {s['worst_survival']:.1f} | "
                      f"{s['mean_score']:.2f} | {s['survived']}/{s['cases']} |"]
    if intervals:
        for key, s in intervals.items():
            lines += ['', f"{key}: mean {s['mean']:+.2f}, paired bootstrap 95% interval "
                      f"[{s['bootstrap_95_percent'][0]:+.2f}, {s['bootstrap_95_percent'][1]:+.2f}]; "
                      f"{s['wins']} wins, {s['losses']} losses, {s['ties']} ties."]
    lines += ['', 'Held-out seeds were not used for candidate selection. These estimates are not a guarantee.', '']
    (args.out/'validation-report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"Validation saved to {args.out / 'validation.json'}", flush=True)
    return True


def seed_list(text):
    values = [int(v) for v in text.split(',')]
    if not values or len(set(values)) != len(values) or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError('Use distinct nonnegative seeds separated by commas')
    return values


def overnight(args, protocol, state, baseline, specs, features, deadline):
    """Persist stage transitions; a new session resumes the unfinished stage."""
    stage = state.setdefault('overnight_stage', 'search')
    while stage != 'done' and time.monotonic() < deadline and not args.interrupted:
        remaining = deadline-time.monotonic()
        # Unused early-stage time flows forward. Reserve time for actual holdout.
        fraction = .60 if stage == 'search' else .50 if stage == 'refine' else 1.
        stage_deadline = time.monotonic()+remaining*fraction
        control = args.out/'control'/uuid.uuid4().hex
        control.mkdir(parents=True)
        (control/'heartbeat').touch()
        args.active_phase = stage
        print(f'Overnight stage: {stage}, budget {remaining*fraction/3600:.2f}h', flush=True)
        done = True
        try:
            if stage == 'search':
                search(args, protocol, state, baseline, specs, features, control, stage_deadline)
                if not all('summary' in t for t in state['trials'][:2]) or len(state['trials']) < 2:
                    print('Controls incomplete. Resume before selecting an experimental policy.', flush=True)
                    done = False
            elif stage == 'refine':
                refine(args, protocol, state, control, stage_deadline)
            else:
                done = validation(args, protocol, state, control, stage_deadline)
        finally:
            (control/'STOP').touch()
        if args.interrupted or not done:
            break
        stage = {'search': 'refine', 'refine': 'validate', 'validate': 'done'}[stage]
        state['overnight_stage'] = stage
        write_json(args.out/'study.json', state)


def main():
    # Detached shells can inherit SIGINT=ignore; still support a graceful stop.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'logs/policy-search')
    parser.add_argument('--profile', choices=('laptop', 'runpod'), default='laptop')
    parser.add_argument('--space', choices=tuple(SPACES), default='trapping',
                        help="Parameter space and policy families; 'notrap' selects the no-trapping campaign")
    parser.add_argument('--family', choices=('all', 'orchard_evasion', 'expert_harvest'), default='all',
                        help='Evaluate one policy family per study; mixing families wastes games on '
                             'parameters that cannot affect the other family')
    parser.add_argument('--engine', choices=('python', 'fastsim', 'native'), default='python',
                        help="Simulation backend for every game in this study; 'native' runs the engine "
                             "and the orchard/evasion policy together in one in-process C++ call and only "
                             "supports --space notrap --family orchard_evasion")
    parser.add_argument('--objective', choices=('survival', 'score'), default='survival',
                        help="Ranking rule for summarize()/best.json. 'survival' (default) keeps every "
                             "existing study's ranking unchanged. 'score' ranks by mean_score + "
                             "0.25*worst_score instead: reaching the horizon is roughly a 1-in-60 lottery "
                             "given the food economy, so survival barely discriminates between candidates "
                             "while score still does. Recorded in protocol.json, so resuming a study with "
                             "a different --objective is rejected by the existing protocol-equality check.")
    parser.add_argument('--phase', choices=('search', 'refine', 'validate', 'overnight'))
    parser.add_argument('--describe', action='store_true', help='Write defaults/inventory only; run no games')
    parser.add_argument('--hours', type=float, help='Session budget; resume with the same command')
    parser.add_argument('--trials', type=int, default=0, help='Total trial cap, not additional trials; 0 = full planned search')
    parser.add_argument('--workers', type=int, help='Parallel games; 0 = CPU quota/memory-aware auto sizing')
    parser.add_argument('--memory-per-worker-gb', type=float, default=2.)
    parser.add_argument('--batch-candidates', type=int, default=0, help='0 = enough candidates to queue two waves of games')
    parser.add_argument('--evolution-trials', type=int, help='Candidates after feature and scalar probes')
    parser.add_argument('--finalists', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=3000.)
    parser.add_argument('--train-seeds', type=seed_list)
    parser.add_argument('--refine-seeds', type=seed_list, default=list(range(12)))
    parser.add_argument('--validation-seeds', type=seed_list)
    parser.add_argument('--policy-seed', type=int, default=0)
    parser.add_argument('--search-seed', type=int, default=123)
    parser.add_argument('--case-timeout', type=float, default=3600., help='Per-game wall seconds; 0 disables this limit')
    args = parser.parse_args()
    runpod = args.profile == 'runpod'
    profile_defaults = dict(workers=0 if runpod else 1, hours=9. if runpod else 8.,
                            phase='overnight' if runpod else 'search',
                            train_seeds=list(range(4)) if runpod else [0, 1],
                            validation_seeds=list(range(1001, 1017)) if runpod else [101, 102, 103],
                            evolution_trials=1024 if runpod else 64)
    for key, value in profile_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if (args.workers < 0 or args.hours <= 0 or args.trials < 0 or not .1 <= args.seconds <= 3000
            or args.policy_seed < 0 or args.case_timeout < 0 or not math.isfinite(args.hours)
            or not math.isfinite(args.case_timeout) or abs(args.seconds*10-round(args.seconds*10)) > 1e-6
            or args.batch_candidates < 0 or args.evolution_trials < 0 or args.finalists < 1
            or not math.isfinite(args.memory_per_worker_gb) or args.memory_per_worker_gb <= 0):
        parser.error('Invalid workers/budget/horizon/seed; horizon must be a multiple of 0.1 seconds')
    if (set(args.train_seeds) | set(args.refine_seeds)) & set(args.validation_seeds):
        parser.error('Screening/refinement and validation seeds must not overlap')
    if args.phase in ('refine', 'overnight') and not set(args.train_seeds) <= set(args.refine_seeds):
        parser.error('--refine-seeds must include all screening --train-seeds')
    from scripts.search_resources import allocation
    resources = allocation(args.memory_per_worker_gb)
    if args.workers == 0:
        args.workers = resources['suggested_workers']
    resources['selected_workers'] = args.workers
    args.interrupted = False
    global SPACE
    SPACE = args.space
    modes = SPACE_MODES[SPACE]
    if args.family != 'all':
        if args.family not in modes:
            parser.error(f'--family {args.family} is not part of the {SPACE} space')
        modes = [args.family]
    if args.engine == 'native' and modes != ['orchard_evasion']:
        parser.error("--engine native only supports --space notrap --family orchard_evasion "
                     "(no native policy exists for trapping or expert_harvest)")
    module = space()
    defaults, inventory, validate = module.defaults, module.inventory, module.validate
    FEATURES = getattr(module, 'FEATURES', ())
    baseline, specs = defaults(), inventory()
    validate(baseline)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with study_lock(args.out/'study.lock'):
        # Fail fast (before any game starts) if the requested engine is not built,
        # stale, or built against a different Python/NumPy on this host.
        engine_build = engine_identity(args.engine)
        objective_text = ('mean_score + .25*worst_score; then mean_survival/horizon; then fewer features'
                          if args.objective == 'score' else
                          'mean_survival/horizon + .25*worst_survival/horizon; then mean_score; then fewer features')
        protocol = dict(version=4, baseline=SPACE_BASELINES[SPACE], space=SPACE,
            train_seeds=args.train_seeds, validation_seeds=args.validation_seeds,
            refine_seeds=args.refine_seeds, finalists=args.finalists,
            modes=modes, predators_enabled=True, seconds=args.seconds, policy_seed=args.policy_seed, search_seed=args.search_seed,
            engine=args.engine, engine_build=engine_build,
            source_hashes=source_manifest(), environment=versions(), parameter_inventory=specs,
            objective=objective_text)
        protocol_path = args.out/'protocol.json'
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise RuntimeError('Source, environment, seeds, objective or search space changed. Use a new --out directory.')
        write_json(protocol_path, protocol)
        write_json(args.out/'default-config.json', baseline)
        write_json(args.out/'parameter-inventory.json', specs)
        write_json(args.out/'resources.json', resources)
        if args.describe:
            print(f"Wrote {len(specs)} parameter entries ({sum(s['tunable'] for s in specs)} tunable); no games started.")
            return
        state_path = args.out/'study.json'
        state = read_json(state_path) if state_path.exists() else dict(trials=[])
        control = args.out/'control'/uuid.uuid4().hex
        control.mkdir(parents=True)
        (control/'heartbeat').touch()
        deadline = time.monotonic()+args.hours*3600.
        try:
            if args.phase == 'overnight':
                overnight(args, protocol, state, baseline, specs, FEATURES, deadline)
            elif args.phase == 'search':
                search(args, protocol, state, baseline, specs, FEATURES, control, deadline)
            elif args.phase == 'refine':
                if (args.out/'validation-selection.json').exists():
                    raise RuntimeError('Cannot change selection after opening held-out validation.')
                refine(args, protocol, state, control, deadline)
            else:
                validation(args, protocol, state, control, deadline)
        finally:
            (control/'STOP').touch()
        print(f"Saved in {args.out}. Repeat the same command to resume.", flush=True)


if __name__ == '__main__':
    main()

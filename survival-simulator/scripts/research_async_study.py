"""External asynchronous study driver using the frozen evaluator and GP model."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
from pathlib import Path
import random
import sys
import time

from research_dispatch import read, write, remote_case


def stream(next_trial, requests_for, evaluate, completed, *, slots, stopped, pulse):
    """Refill after any game finishes; summarize only complete candidate panels."""
    pending, groups = {}, {}
    exhausted = False
    with ThreadPoolExecutor(max_workers=slots) as pool:
        while pending or not exhausted:
            pulse()
            if stopped():
                exhausted = True
            while not exhausted and len(pending) < slots:
                trial = next_trial()
                if trial is None:
                    exhausted = True
                    break
                requests = requests_for(trial)
                groups[trial['id']] = [trial, requests, {}]
                for request in requests:
                    future = pool.submit(evaluate, request)
                    pending[future] = (trial['id'], request['case_id'])
            if not pending:
                break
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                tid, cid = pending.pop(future)
                groups[tid][2][cid] = future.result()
            for tid, (trial, requests, results) in list(groups.items()):
                if len(results) == len(requests):
                    rows = [results[r['case_id']] for r in requests]
                    if not any(row['status'] == 'interrupted' for row in rows):
                        completed(trial, rows)
                    del groups[tid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--priority', type=int, default=10)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from scripts.research_support import digest, manifest, assert_snapshot, research_case
    from models.experiment_config import defaults, inventory, leaves, get, put, repair, validate
    from scripts.optimize_policy import summarize, sample, activate
    from scripts.research_bo import choose
    request = read(args.request)
    base, specs = defaults(), inventory()
    args.out.mkdir(parents=True, exist_ok=True)
    args.control.mkdir(parents=True, exist_ok=True)
    write(args.out/'catalog.json', dict(defaults=base, inventory=specs))
    if request['mode'] == 'catalog':
        return
    known, inherited = dict(leaves(base)), dict(leaves(request['starting_config']))
    for path, value in inherited.items():
        if path in known:
            put(base, path, value)
    base = validate(repair(base))
    write(args.out/'migration.json', dict(dropped=sorted(set(inherited)-set(known)), added=sorted(set(known)-set(inherited))))
    selectable = {s['path']: s for s in specs if s['tunable'] and not s['path'].startswith('features.')}
    paths = request['paths']
    if len(paths) > request['max_dimensions'] or any(p not in selectable for p in paths):
        raise ValueError('Unknown, fixed, feature-switch, or excessive search dimensions')
    selected_specs = [selectable[p] for p in paths]
    files = read(args.source.parent/'manifest.json')
    assert_snapshot(args.source, files)
    protocol = dict(version=1, source=files, environment=request['environment'], seconds=3000,
        policy_seed=request['policy_seed'], modes=['trapping'], seeds=request['seeds'], mode=request['mode'],
        start=base, variants=request['variants'], paths=paths, search_seed=request['search_seed'],
        engine=request.get('engine', 'python'), diagnostics=request.get('diagnostics'))
    previous = read(args.out/'protocol.json', {})
    if previous and previous != protocol:
        raise ValueError('Study provenance changed')
    write(args.out/'protocol.json', protocol)
    write(args.out/'scheduling.json', dict(method='asynchronous case refill',
        pending_case_target=min(request['workers'], 40), completion_rule='entire predeclared seed panel',
        driver_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
    state = read(args.out/'study.json', {'trials': []})
    initial, labels = [base], {digest(base): 'inherited configuration'}
    for variant in request['variants']:
        config = copy.deepcopy(base)
        for path, value in variant['overrides'].items():
            if path not in known:
                raise ValueError('Unknown override: '+path)
            put(config, path, value)
        config = validate(repair(config))
        initial.append(config)
        labels[digest(config)] = variant['label']
    initial = list({digest(c): c for c in initial}.values())
    limit = max(request['trials'], len(initial))
    resumed = [t for t in state['trials'] if 'summary' not in t]
    initial_ids = {digest(c) for c in initial}
    started = time.time()
    last_pulse = 0

    def ranking():
        return sorted((t for t in state['trials'] if t.get('summary', {}).get('rank') is not None),
                      key=lambda t: t['summary']['rank'], reverse=True)

    def next_trial():
        if resumed:
            return resumed.pop(0)
        if len(state['trials']) >= limit:
            return None
        seen = {digest(t['config']) for t in state['trials']}
        config = next((c for c in initial if digest(c) not in seen), None)
        parent_id = None
        if config is None:
            ranked = ranking()
            # Prior exploration can run while the anchor's slowest map finishes.
            parents = ranked[:3] or [dict(id=0, config=base)]
            rng = random.Random(request['search_seed']+len(state['trials'])*100003)
            candidates, ancestry = [], {}
            for _ in range(256):
                parent = rng.choice(parents)
                candidate = copy.deepcopy(parent['config'])
                active = []
                for spec in selected_specs:
                    if 'requires_features' in spec:
                        if all(candidate['features'].get(f, False) for f in spec['requires_features']):
                            active.append(spec)
                    else:
                        probe = copy.deepcopy(candidate)
                        activate(probe, spec['path'])
                        if probe == candidate:
                            active.append(spec)
                if not active:
                    continue
                n = rng.randint(1, min(4 if request['mode'] == 'focus' else len(active), len(active)))
                for spec in rng.sample(active, n):
                    put(candidate, spec['path'], sample(spec, get(candidate, spec['path']), rng))
                candidate = repair(candidate)
                try:
                    validate(candidate)
                except ValueError:
                    continue
                identity = digest(candidate)
                if identity not in seen:
                    seen.add(identity)
                    candidates.append(candidate)
                    ancestry[identity] = parent['id']
            if not candidates:
                return None
            if request['mode'] == 'bo':
                varying = {p for c in candidates for p, v in leaves(c) if v != get(base, p)}
                varying.update(p for t in ranked for p, v in leaves(t['config']) if v != get(base, p))
                model_specs = [s for s in specs if s['path'] in varying or s['path'] in paths]
                config = choose(candidates, ranked, model_specs, get, 1)[0]
            else:
                config = candidates[0]
            parent_id = ancestry[digest(config)]
        trial = dict(id=len(state['trials']), baseline=False, config=config,
            label=labels.get(digest(config), request['mode']+' asynchronous suggestion'), parent=parent_id,
            changed=[p for p, v in leaves(config) if v != get(base, p)])
        state['trials'].append(trial)
        write(args.out/'study.json', state)
        return trial

    def requests_for(trial):
        return [research_case(digest(files), request['environment'], trial['config'], False, seed,
            request['policy_seed'], request.get('diagnostics'), request.get('engine', 'python'),
            infrastructure_retries=request.get('case_retries', 0)) for seed in request['seeds']]

    def evaluate(case):
        return remote_case(args.queue, case, Path(request['case_cache']), args.control,
            request['case_timeout'], runtime_root=args.source, priority=args.priority)

    def completed(trial, rows):
        trial['case_ids'] = [r['case_id'] for r in rows]
        trial['summary'] = summarize(rows, 3000, sum(trial['config']['features'].values()))
        write(args.out/'study.json', state)
        ranked = ranking()
        if ranked:
            write(args.out/'best.json', ranked[0])
        write(args.out/'report.json', dict(completed=sum('summary' in t for t in state['trials']),
            ranking=[{k: t[k] for k in ('id', 'label', 'parent', 'changed', 'case_ids', 'summary', 'config')} for t in ranked],
            failed=[t['id'] for t in state['trials'] if t.get('summary', {}).get('rank', []) is None],
            method='asynchronous GP expected improvement' if request['mode'] == 'bo' else 'asynchronous focused mutation'))

    def stopped():
        return (time.time() >= request['deadline'] or (args.control/'STOP').exists()
                or (state['trials'] and state['trials'][0].get('summary', {}).get('errors')))

    def pulse():
        nonlocal last_pulse
        (args.control/'heartbeat').touch()
        if stopped():
            (args.control/'STOP').touch()
        if time.time()-last_pulse > 10:
            write(args.out/'progress.json', dict(phase=request['mode'], workers=request['workers'],
                proposed=len(state['trials']), completed_candidates=sum('summary' in t for t in state['trials']),
                wall_seconds=time.time()-started, phase_seconds_remaining=max(0, request['deadline']-time.time())))
            last_pulse = time.time()

    stream(next_trial, requests_for, evaluate, completed, slots=min(request['workers'], 40), stopped=stopped, pulse=pulse)
    assert_snapshot(args.source, files)
    write(args.out/'finished.json', dict(planned=limit, proposed=len(state['trials']),
        completed=sum('summary' in t for t in state['trials']), budget_exhausted=time.time() >= request['deadline']))


if __name__ == '__main__':
    main()

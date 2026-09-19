"""Internal frozen-source focused/BO study. Only the supervisor launches this."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_support import read, write, digest, manifest, research_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    request = read(args.request)
    from models.experiment_config import defaults, inventory, leaves, get, put, repair, validate
    from scripts.optimize_policy import evaluate_batch, summarize, sample, activate
    from scripts.research_bo import choose

    base, specs = defaults(), inventory()
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/'catalog.json', dict(defaults=base, inventory=specs))
    if request['mode'] == 'catalog':
        return
    # Explicit migration: carry only fields that still exist; report every drop.
    known = dict(leaves(base))
    inherited = dict(leaves(request['starting_config']))
    for path, value in inherited.items():
        if path in known:
            put(base, path, value)
    base = validate(repair(base))
    write(args.out/'migration.json', dict(dropped=sorted(set(inherited)-set(known)),
                                         added=sorted(set(known)-set(inherited))))
    selectable = {s['path']: s for s in specs if s['tunable'] and not s['path'].startswith('features.')}
    paths = request['paths']
    if len(paths) > request['max_dimensions'] or any(p not in selectable for p in paths):
        raise ValueError('Unknown, fixed, feature-switch, or excessive search dimensions')
    selected_specs = [selectable[p] for p in paths]
    protocol = dict(version=1, source=manifest(ROOT), environment=request['environment'],
                    seconds=3000, policy_seed=request['policy_seed'], modes=['trapping'],
                    seeds=request['seeds'], mode=request['mode'], start=base,
                    variants=request['variants'], paths=paths, search_seed=request['search_seed'],
                    engine=request.get('engine', 'python'), diagnostics=request.get('diagnostics'))
    if (args.out/'protocol.json').exists() and read(args.out/'protocol.json') != protocol:
        raise RuntimeError('Study provenance changed')
    write(args.out/'protocol.json', protocol)
    state = read(args.out/'study.json') if (args.out/'study.json').exists() else {'trials': []}
    initial = [base]
    labels = {digest(base): 'inherited configuration'}
    for variant in request['variants']:
        config = copy.deepcopy(base)
        for path, value in variant['overrides'].items():
            if path not in known:
                raise ValueError(f'Unknown override: {path}')
            put(config, path, value)
        config = validate(repair(config))
        initial.append(config)
        labels[digest(config)] = variant['label']
    initial = list({digest(c): c for c in initial}.values())
    limit = max(request['trials'], len(initial))
    params = SimpleNamespace(out=args.out, workers=request['workers'], seconds=3000,
                             case_timeout=request['case_timeout'], phase=request['mode'], interrupted=False,
                             case_cache=Path(request['case_cache']))
    snapshot = digest(protocol['source'])
    def case_factory(trial, seed, _mode):
        return research_case(snapshot, request['environment'], trial['config'], False, seed, request['policy_seed'],
                             request.get('diagnostics'), request.get('engine', 'python'),
                             infrastructure_retries=request.get('case_retries', 0))
    deadline = request['deadline'] - time.time() + time.monotonic()

    def save_trial(trial, results):
        trial['case_ids'] = [r['case_id'] for r in results]
        trial['summary'] = summarize(results, 3000, sum(trial['config']['features'].values()))
        write(args.out/'study.json', state)
        ranked = sorted((t for t in state['trials'] if t.get('summary', {}).get('rank') is not None),
                        key=lambda t: t['summary']['rank'], reverse=True)
        if ranked:
            write(args.out/'best.json', ranked[0])
        write(args.out/'report.json', dict(completed=len([t for t in state['trials'] if 'summary' in t]),
              ranking=[dict(id=t['id'], label=t['label'], parent=t['parent'], changed=t['changed'],
                            case_ids=t['case_ids'], summary=t['summary'], config=t['config']) for t in ranked],
              failed=[t['id'] for t in state['trials'] if t.get('summary', {}).get('rank', []) is None],
              method='Gaussian-process expected improvement' if request['mode'] == 'bo' else 'focused block mutation'))

    batch_size = max(1, math.ceil(request['workers']/len(request['seeds'])))
    while time.monotonic() < deadline and not (args.control/'STOP').exists():
        pending = [t for t in state['trials'] if 'summary' not in t][:batch_size]
        available = min(batch_size-len(pending), limit-len(state['trials']))
        configs = []
        parents = {}
        seen = {digest(t['config']) for t in state['trials']}
        while available > len(configs) and len(state['trials'])+len(configs) < len(initial):
            configs.append(initial[len(state['trials'])+len(configs)])
        if not configs and available > 0 and selected_specs:
            ranked = sorted((t for t in state['trials'] if t.get('summary', {}).get('rank') is not None),
                            key=lambda t: t['summary']['rank'], reverse=True)
            if not ranked:
                break
            rng = random.Random(request['search_seed']+len(state['trials'])*100003)
            pool = []
            for _ in range(512):
                parent_trial = rng.choice(ranked[:3])
                parent = parent_trial['config']
                config = copy.deepcopy(parent)
                # Do not silently activate optional blocks when tuning scalars.
                active = []
                for spec in selected_specs:
                    if 'requires_features' in spec:
                        if all(config['features'].get(name, False) for name in spec['requires_features']):
                            active.append(spec)
                        continue
                    probe = copy.deepcopy(config)
                    activate(probe, spec['path'])
                    if probe == config:
                        active.append(spec)
                if not active:
                    continue
                count = rng.randint(1, min(4 if request['mode'] == 'focus' else len(active), len(active)))
                for spec in rng.sample(active, count):
                    put(config, spec['path'], sample(spec, get(config, spec['path']), rng))
                config = repair(config)
                try:
                    validate(config)
                except ValueError:
                    continue
                identity = digest(config)
                if identity not in seen:
                    seen.add(identity)
                    parents[identity] = parent_trial['id']
                    pool.append(config)
            if request['mode'] == 'bo':
                # Include variant differences as covariates: identical scalars
                # with different feature switches are not the same GP input.
                varying = {p for c in pool for p, value in leaves(c) if value != get(base, p)}
                varying.update(p for t in ranked for p, value in leaves(t['config']) if value != get(base, p))
                model_specs = [s for s in specs if s['path'] in varying or s['path'] in paths]
                configs = choose(pool, ranked, model_specs, get, available)
            else:
                configs = pool[:available]
        for config in configs:
            trial = dict(id=len(state['trials']), baseline=False, config=config,
                         label=labels.get(digest(config), request['mode']+' suggestion'),
                         parent=parents.get(digest(config)),
                         changed=[p for p, value in leaves(config) if value != get(base, p)])
            state['trials'].append(trial)
            pending.append(trial)
        if not pending:
            break
        write(args.out/'study.json', state)
        evaluate_batch(pending, request['seeds'], protocol, params, args.control, deadline, save_trial,
                       request_factory=case_factory)
        if params.interrupted:
            break
        # A failed starting configuration indicates a broken code revision.
        if state['trials'] and state['trials'][0].get('summary', {}).get('errors'):
            break
    write(args.out/'finished.json', dict(planned=limit, proposed=len(state['trials']),
          completed=sum('summary' in t for t in state['trials']),
          budget_exhausted=time.monotonic() >= deadline))


if __name__ == '__main__':
    main()

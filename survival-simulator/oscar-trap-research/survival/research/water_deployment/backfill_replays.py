"""Resumable replay audit and NEW reruns; preserve historical metrics and files."""
from common import *
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gc
import traceback

RECEIPTS = OUT / 'reproductions'
ORIGINALS = {
    ('native-record.json', 0): 'native-hold.replay.json',
    ('approach-relay-record.json', 0): 'approach-relay-native.replay.json',
    ('two-stations-record.json', 0): 'two-stations-native.replay.json',
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=lambda x: x.item()) + '\n')
    os.replace(temporary, path)


def inventory():
    jobs = []
    for source in sorted(OUT.glob('*.json')):
        payload = json.loads(source.read_text())
        rows = payload.get('data', [])
        if not isinstance(rows, list):
            continue
        for index, old in enumerate(rows):
            if not isinstance(old, dict) or not ('case' in old or source.name == 'observation-transects.json'):
                continue
            original = ORIGINALS.get((source.name, index))
            if source.name == 'alternative-gap-native-multiple.json' and old.get('recording'):
                original = 'alternative-gap-native-three-predators.replay.json.gz'
            if original:
                original = str((OUT / original).relative_to(ROOT))
            original = old.get('recording', {}).get('path') or original
            jobs.append(dict(source_name=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                row_index=index, old=old, metadata=payload.get('metadata', {}),
                family='probe' if source.name == 'observation-transects.json' else
                       'gap' if source.name.startswith('alternative-gap-') else 'water',
                original=original, receipt=str(RECEIPTS / f'{source.stem}-row-{index + 1:03d}.json')))
    return jobs


def controller_for(job):
    """Use exact archived controller/base hashes where those sources survive."""
    original = job['metadata'].get('experiment_sha256', {})
    if (original.get('controller.py') == EXPERIMENT_HASHES.get('controller_v2.py') and
            original.get('base_snapshot.py') == EXPERIMENT_HASHES.get('base_snapshot_v1.py')):
        import controller_v2
        import base_snapshot_v1
        controller_v2.Config = base_snapshot_v1.Config
        controller_v2.Scenario = base_snapshot_v1.Scenario
        controller_v2.Controller = base_snapshot_v1.Controller
        def factory(site, poses, **kwargs):
            return controller_v2.PairPolicy(site, poses, **{k: v for k, v in kwargs.items()
                if k in {'outside', 'relay', 'mapped_obstacles', 'food'}})
        return factory, 'Archived controller and base source hashes match the historical result; recording and diagnostic harness is current.'
    if job['family'] != 'water':
        return None, 'Saved configuration rerun with the current recording-enabled harness and unchanged mechanism; original historical frames are unavailable.'
    if all(original.get(k) == EXPERIMENT_HASHES.get(k) for k in ('controller.py', 'base_snapshot.py')):
        return None, 'Current controller and base source hashes match the historical result; recording and diagnostic harness is current.'
    note = 'Exact historical controller revision is unavailable; this is a current-controller rerun of the saved configuration, not an exact historical replay.'
    known = {
        'native-pilot.json': ' Current dead reckoning uses expanded rectangular obstacle collision instead of the original approximate corner model.',
        'relay-early-birth-pilot.json': ' Current newborn scheduling is energy-triggered, whereas the pilot used an earlier timed birth.',
        'dogleg-negative-pilot.json': ' Current close-entry dogleg goes in the positive direction; the historical pilot used the opposite direction.',
    }
    return None, note + known.get(job['source_name'], '')


def compare(old, new):
    keys = ('elapsed_s', 'alive', 'score', 'acquired_s', 'first_capture_s', 'first_depletion_s',
            'capture_s', 'first_escape_s', 'joint_fraction', 'energy_remaining', 'energy_used',
            'completed_transects', 'qualified', 'measured_common_water', 'measured_dry_envelope')
    def equal(a, b):
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            return math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-9)
        if isinstance(a, dict) and isinstance(b, dict):
            aa, bb = {str(k): v for k,v in a.items()}, {str(k): v for k,v in b.items()}
            return aa.keys() == bb.keys() and all(equal(aa[k], bb[k]) for k in aa)
        return a == b
    return {k: dict(historical=old[k], reproduced=new.get(k)) for k in keys
            if k in old and not equal(old[k], new.get(k))}


def reproduce(job):
    old = job['old']
    case = copy.deepcopy(old.get('case', {}))
    factory, note = controller_for(job)
    changes = []
    if job['source_name'] in {'renewal-v3.json', 'native-renewal-v3.json'}:
        case['relay'] = 'renewal-urgent'
        changes.append('Historical v3 relay=renewal is now explicitly named renewal-urgent.')
    provenance = {k: job[k] for k in ('source_name', 'source_sha256', 'row_index')}
    provenance.update(fidelity_note=note, configuration_changes=changes,
                      historical_experiment_sha256=job['metadata'].get('experiment_sha256', {}))
    if job['family'] == 'gap':
        from alternative_traps import run
        result = run(case, reproduction=provenance)
    elif job['family'] == 'probe':
        from probe import probe
        maps = {m['seed']: m['sites'] for m in json.loads((OUT/'sites-1-40.json').read_text())['data']}
        result = probe(old['seed'], maps[old['seed']][old['site_index']], old['site_index'], reproduction=provenance)
    else:
        from run import run
        result = run(case, reproduction=provenance, policy_factory=factory)
    receipt = dict(status='new_reproduction' if 'recording' in result else 'reproduction_skipped',
                   provenance=provenance, metric_differences=compare(old, result), result=result)
    atomic_json(Path(job['receipt']), receipt)
    gc.collect()
    return dict(source=job['source_name'], row=job['row_index'] + 1, status=receipt['status'],
                metric_differences=list(receipt['metric_differences']), recording=result.get('recording'))


def audit(jobs):
    entries = []
    for job in jobs:
        entry = {k: job[k] for k in ('source_name', 'source_sha256', 'row_index', 'family')}
        if 'skipped' in job['old']:
            entry.update(status='preflight_skipped_no_simulation', reason=job['old']['skipped'])
        elif job['original'] and (ROOT/job['original']).is_file():
            entry.update(status='original_replay', replay=job['original'])
        elif Path(job['receipt']).is_file():
            receipt = json.loads(Path(job['receipt']).read_text())
            if receipt['provenance']['source_sha256'] != job['source_sha256']:
                raise ValueError(f"Historical metrics changed: {job['source_name']}")
            entry.update(status=receipt['status'], receipt=str(Path(job['receipt']).relative_to(ROOT)),
                replay=receipt['result'].get('recording', {}).get('path'),
                fidelity_note=receipt['provenance']['fidelity_note'],
                metric_differences=receipt['metric_differences'])
            if entry['replay'] and not (ROOT/entry['replay']).is_file():
                entry['status'] = 'missing_reproduction_file'
        else:
            entry['status'] = 'missing_replay'
        entries.append(entry)
    counts = {status: sum(e['status'] == status for e in entries) for status in sorted({e['status'] for e in entries})}
    value = dict(scope='Top-level water_deployment saved simulation rows; static map scans excluded.',
                 note='New reproductions are not original historical frames. Historical metric files are preserved.',
                 counts=counts, entries=entries)
    atomic_json(OUT/'replay-audit.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    jobs = inventory()
    before = audit(jobs)
    print('Audit:', before['counts'], flush=True)
    missing = {(e['source_name'], e['row_index']) for e in before['entries'] if e['status'] in {'missing_replay', 'missing_reproduction_file'}}
    pending = [j for j in jobs if (j['source_name'], j['row_index']) in missing]
    if args.limit is not None:
        pending = pending[:args.limit]
    if args.run:
        # Keep workers alive for this bounded batch. Python 3.12 worker recycling
        # stalled after every worker reached max_tasks_per_child in this setup.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(reproduce, job): job for job in pending}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    print(n, '/', len(pending), json.dumps(future.result()), flush=True)
                except Exception:
                    traceback.print_exc()
                audit(jobs)
    print('Final audit:', audit(jobs)['counts'], flush=True)


if __name__ == '__main__':
    main()

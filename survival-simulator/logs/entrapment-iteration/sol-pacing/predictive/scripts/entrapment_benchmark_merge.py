"""Validate seed coverage and merge independently downloaded benchmark shards."""
import argparse
import json
from pathlib import Path

from entrapment_benchmark import BASELINE_COMMIT, aggregate, verify_sources, write_json


def merge(root, expected_count, allow_incomplete=False):
    expected_hashes = verify_sources()['sources']
    protocols = sorted(root.rglob('protocol.json'))
    if not protocols:
        raise ValueError('No shard protocols found')
    planned = set()
    cases = {}
    unfinished = []
    for path in protocols:
        protocol = json.loads(path.read_text())
        if protocol['source_hashes'] != expected_hashes:
            raise ValueError(f'Source mismatch: {path}')
        if protocol['baseline_commit'] != BASELINE_COMMIT or protocol['seconds'] != 3000:
            raise ValueError(f'Baseline/horizon mismatch: {path}')
        if protocol['policy_rng_seed'] != 0:
            raise ValueError(f'Policy seed mismatch: {path}')
        seeds = protocol['seeds']
        if len(set(seeds)) != len(seeds) or planned.intersection(seeds):
            raise ValueError(f'Duplicate planned seeds: {path}')
        planned.update(seeds)
        parent_cases = path.parent/'cases.json'
        rows = json.loads(parent_cases.read_text()) if parent_cases.exists() else []
        by_seed = {r['seed']: r for r in rows}
        for seed in seeds:
            result_path = path.parent/f'{seed:06d}'/'result.json'
            if result_path.exists():
                row = json.loads(result_path.read_text())
                if seed in by_seed and by_seed[seed] != row:
                    raise ValueError(f'Parent/case disagreement: {result_path}')
                by_seed[seed] = row
            if seed not in by_seed:
                if not allow_incomplete:
                    raise ValueError(f'Missing result: seed {seed}')
                progress = result_path.with_name('progress.json')
                unfinished.append(dict(seed=seed, status='incomplete',
                    last_saved_progress=json.loads(progress.read_text()) if progress.exists() else None))
                continue
            row = by_seed[seed]
            if row['seed'] != seed:
                raise ValueError(f'Incorrect case seed: {result_path}')
            if row['status'] != 'error':
                if row['baseline_commit'] != BASELINE_COMMIT or row['horizon'] != 3000:
                    raise ValueError(f'Incorrect case protocol: {result_path}')
                if row['status'] == 'survived_horizon' and row['sim_time'] < 2999.9:
                    raise ValueError(f'Premature survival: {result_path}')
            cases[seed] = row
        if set(by_seed) - set(seeds):
            raise ValueError(f'Unplanned results: {path}')
    if planned != set(range(expected_count)):
        raise ValueError(f'Seed coverage mismatch: missing {sorted(set(range(expected_count))-planned)}, extra {sorted(planned-set(range(expected_count)))}')
    rows = [cases[seed] for seed in sorted(cases)]
    return rows, dict(aggregate(rows), baseline_commit=BASELINE_COMMIT,
                     exact_seed_coverage=not unfinished, all_seeds_planned=True,
                     expected_games=expected_count, incomplete_games=len(unfinished),
                     incomplete_cases=unfinished,
                     source_hashes=expected_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Explicitly report missing cases instead of rejecting the batch')
    args = parser.parse_args()
    cases, summary = merge(args.root, args.count, args.allow_incomplete)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'cases.json', cases)
    write_json(args.out/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()

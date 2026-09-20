"""Combine completed benchmark shards; reject incomplete/mixed headline data."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.harvest_benchmark import summary


def combine(paths, expected=1000):
    rows = []
    manifests = []
    hashes = []
    for path in map(Path, paths):
        part = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        if len(part) != manifest['arguments']['seeds']:
            raise ValueError(f'incomplete shard: {path}')
        rows.extend(part)
        manifests.append(manifest)
        hashes.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len(rows) != expected or len({r['seed'] for r in rows}) != expected:
        raise ValueError(f'need {expected} distinct completed seeds')
    for key in ('engine', 'arm', 'horizon', 'extra_drain_actions'):
        values = {m['arguments'].get(key, 0) for m in manifests}
        if len(values) != 1:
            raise ValueError(f'mixed {key}: {values}')
    for path in manifests[0]['source_sha256']:
        # Build metadata includes host/compiler binaries; source files must match.
        if path.endswith('build-info-policy.json'):
            continue
        if len({m['source_sha256'].get(path) for m in manifests}) != 1:
            raise ValueError(f'mixed source: {path}')
    for r in rows:
        if r['population'] and r['sim_time'] < r['horizon']-.01:
            raise ValueError(f'incomplete game: {r["seed"]}')
    rows.sort(key=lambda r:r['seed'])
    totals = summary(rows)
    totals.update(engine=rows[0]['engine'], horizon=rows[0]['horizon'],
                  extra_drain_actions=manifests[0]['arguments'].get('extra_drain_actions', 0),
                  seeds=[r['seed'] for r in rows], shards=hashes)
    totals['harvest_success_rate'] = totals['actual_harvests']/max(1, totals['attempts'])
    totals['peak_actions'] = max(r['peak_actions'] for r in rows)
    totals['games_with_failed_farms'] = sum(bool(r['failed_farms']) for r in rows)
    return rows, totals, manifests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--expected', type=int, default=1000)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows, totals, manifests = combine(a.inputs, a.expected)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(totals, indent=2)+'\n')
    out.with_suffix('.seeds.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='log'})+'\n' for r in rows))
    out.with_suffix('.manifests.json').write_text(json.dumps(manifests, indent=2)+'\n')
    print(json.dumps({k:v for k,v in totals.items() if k not in ('seeds','shards')}, indent=2))


if __name__ == '__main__':
    main()

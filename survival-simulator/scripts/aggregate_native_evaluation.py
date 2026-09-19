#!/usr/bin/env python3
"""Combine disjoint frozen native benchmark shards without selecting results."""
import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'native_policy'))
import evaluate as evaluation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(folder, seed_start, count, commit):
    manifests = sorted((folder / 'shards').glob('*/manifest.json'))
    if not manifests:
        raise ValueError('No shard manifests')
    rows, identities, evidence = {}, [], []
    starts, ends = [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        data_path = path.with_name('games.jsonl')
        if manifest['git_commit'] != commit or manifest['git_dirty']:
            raise ValueError(f'Unfrozen source: {path}')
        if digest(data_path) != manifest['results_sha256']:
            raise ValueError(f'Result hash mismatch: {path}')
        identity = dict(commit=manifest['git_commit'], source=manifest['source_sha256'],
                        config_sha256=manifest['config']['sha256'], config_name=manifest['config']['name'],
                        runtime=manifest['runtime'], horizon=manifest['run']['horizon'])
        identities.append(identity)
        expected = set(manifest['run']['seeds'])
        shard_rows = [json.loads(line) for line in data_path.read_text().splitlines()]
        if len(shard_rows) != manifest['run']['games'] or {r['seed'] for r in shard_rows} != expected:
            raise ValueError(f'Incomplete shard: {path}')
        for row in shard_rows:
            if row['seed'] in rows:
                raise ValueError(f'Duplicate seed: {row["seed"]}')
            rows[row['seed']] = row
        starts.append(datetime.fromisoformat(manifest['created_utc']))
        ends.append(datetime.fromisoformat(manifest['completed_utc']))
        evidence.append(dict(path=str(path.relative_to(folder)), sha256=digest(path),
                             results_sha256=digest(data_path), seeds=manifest['run']['seeds'],
                             workers=manifest['run']['workers']))
    if any(item != identities[0] for item in identities):
        raise ValueError('Mixed source/config/runtime/horizon across shards')
    if set(rows) != set(range(seed_start, seed_start + count)):
        raise ValueError('Batch does not contain exactly the requested seed range')
    ordered = [rows[seed] for seed in sorted(rows)]
    results_path = folder / 'games.jsonl'
    results_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in ordered))
    unknown = {}
    for row in ordered:
        unknown.update(row['metrics_unknown'])
    measured = [row['native_evaluation'] for row in ordered]
    active = [m for m in measured if m['bait_occupancy_started']]
    summary = dict(
        games=count,
        score=evaluation.describe([r['score'] for r in ordered], 20260919),
        simulated_seconds=evaluation.describe([r['simulated_seconds'] for r in ordered], 20260920),
        reached_horizon=sum(r['reached_horizon'] for r in ordered),
        reached_horizon_rate=sum(r['reached_horizon'] for r in ordered) / count,
        deaths_by_cause={cause: sum(r['deaths_by_cause'][cause] for r in ordered)
                         for cause in ('starvation', 'predator')},
        guide_attempts=sum(m['guide_attempts'] for m in measured),
        guide_deliveries=sum(m['guide_deliveries'] for m in measured),
        native_evaluation=evaluation.summarize_native_evaluation(ordered, 20261019),
        games_with_bait=len(active),
        games_with_bait_gap=sum(m['bait_gap_seconds_total'] > 1e-6 for m in active),
        games_with_a_predator_near_occupied_bait_30s=sum(m['maximum_predators_near_bait_30s'] > 0 for m in measured),
        games_with_sprint_available_premature_capture=sum(m['premature_captures_with_sprint_available'] > 0 for m in measured),
        total_worker_seconds=sum(r['wall_seconds'] for r in ordered),
        batch_wall_seconds=(max(ends) - min(starts)).total_seconds(),
        metrics_unknown=unknown,
    )
    evaluation.atomic_json(folder / 'summary.json', summary)
    evaluation.histogram_svg([r['score'] for r in ordered], folder / 'score-histogram.svg')
    evaluation.atomic_json(folder / 'manifest.json', dict(
        schema_version=1, kind='complete_disjoint_shards', identity=identities[0],
        seed_start=seed_start, games=count, seeds=sorted(rows), shards=evidence,
        created_utc=min(starts).isoformat(), completed_utc=max(ends).isoformat(),
        results_sha256=digest(results_path), aggregation_script_sha256=digest(pathlib.Path(__file__)),
        note='All requested seeds included once. Native evaluation fields are authoritative; top-level guide fields in raw rows are legacy debug counters. No runtime mixing.',
    ))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=pathlib.Path)
    p.add_argument('--seed-start', type=int, default=2026091900)
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--commit', default='04173f8ee9d957ff947e6f7d5e73f76ef27750b0')
    a = p.parse_args()
    aggregate(a.folder, a.seed_start, a.games, a.commit)

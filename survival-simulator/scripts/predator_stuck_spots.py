"""Analyze every recorded trap location, streaming the compressed Runpod archives.

Examples (from the repository root):
  python survival-simulator/scripts/predator_stuck_spots.py --workers 8
  python survival-simulator/scripts/predator_stuck_spots.py --report-only
  python survival-simulator/scripts/predator_stuck_spots.py --entrances --workers 8

Completed games are cached atomically, so an interrupted run can be resumed.
Only NumPy and the Python standard library are required; no pod or C++ build.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import gzip
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HELPERS = Path(__file__).resolve().with_name('predator_stuck_cpp')
sys.path.insert(0, str(HELPERS))
from spot_features import FEATURE_VERSION, process_game

GAME_MEMBER = re.compile(r'(?:^|/)games/seed-(\d+)/(.*)$')


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def archive_games(archive, completed, cluster_radius):
    """No extraction: at most one game's compressed event files are buffered."""
    seed = None
    files = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            match = GAME_MEMBER.search(member.name)
            if match:
                member_seed, relative = int(match[1]), match[2]
                if seed is not None and member_seed != seed:
                    if seed not in completed:
                        yield seed, archive.name, files, cluster_radius
                    files = {}
                seed = member_seed
                wanted = relative in ('map.json', 'result.json', 'biomes.bin.gz', 'unflagged-control.json.gz') or (
                    relative.startswith('events/') and relative.endswith('.json.gz'))
                if wanted and member.isfile() and seed not in completed:
                    files[relative] = tar.extractfile(member).read()
            # Streaming callers never need TarInfo objects after this member.
            tar.members.clear()
    if seed is not None and seed not in completed:
        yield seed, archive.name, files, cluster_radius


def input_games(folder, completed, cluster_radius):
    archives = sorted(folder.glob('shard-*.tar.gz'))
    if archives:
        for archive in archives:
            marker = folder/(archive.name.removesuffix('.tar.gz')+'-verified.json')
            if marker.is_file():
                shard = json.loads(marker.read_text())
                seeds = range(shard['start_seed'], shard['start_seed']+shard['games'])
                if all(seed in completed for seed in seeds):
                    continue
            yield from archive_games(archive, completed, cluster_radius)
    else:
        for game in sorted((folder/'games').glob('seed-*')):
            seed = int(game.name.split('-')[1])
            if seed in completed:
                continue
            paths = [game/'map.json', game/'result.json', *sorted((game/'events').glob('*.json.gz'))]
            if (game/'unflagged-control.json.gz').is_file():
                paths.append(game/'unflagged-control.json.gz')
            if (game/'biomes.bin.gz').is_file():
                paths.append(game/'biomes.bin.gz')
            yield seed, 'directory', {p.relative_to(game).as_posix(): p.read_bytes() for p in paths}, cluster_radius


def save_game(cache, result):
    path = cache/f"seed-{result['seed']:08d}.json.gz"
    temporary = path.with_suffix('.gz.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as f:
        json.dump(result, f, separators=(',', ':'), allow_nan=False)
    temporary.replace(path)


def export_tables(output, expected=None, partial=False):
    totals = dict(games=0, cases=0, controls=0, spots=0, candidate_reconstruction_mismatches=0)
    seeds = set()
    with (output/'windows.csv.tmp').open('w', newline='', encoding='utf-8') as windows, \
         (output/'spots.csv.tmp').open('w', newline='', encoding='utf-8') as spots:
        window_writer = spot_writer = None
        for path in sorted((output/'game-features').glob('seed-*.json.gz')):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                result = json.load(f)
            if result['seed'] in seeds:
                raise ValueError('Duplicate seed in feature cache')
            seeds.add(result['seed']); totals['games'] += 1
            for row in result['rows']:
                if window_writer is None:
                    window_writer = csv.DictWriter(windows, fieldnames=list(row))
                    window_writer.writeheader()
                window_writer.writerow(row)
                totals['cases' if row['is_case'] else 'controls'] += 1
                totals['candidate_reconstruction_mismatches'] += row['candidate_reconstruction_mismatch']
            for row in result['spots']:
                if spot_writer is None:
                    spot_writer = csv.DictWriter(spots, fieldnames=list(row))
                    spot_writer.writeheader()
                spot_writer.writerow(row); totals['spots'] += 1
            if totals['games'] % 500 == 0:
                progress = totals | dict(status='exporting_tables')
                atomic_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if not seeds:
        raise ValueError('No games were analyzed')
    if expected and not partial:
        coverage = expected['coverage']
        if seeds != set(range(coverage['first_seed'], coverage['last_seed']+1)):
            raise ValueError('Seed coverage does not match the source dataset')
        if totals['cases'] != expected['counts']['flagged_predators']:
            raise ValueError('Finding count does not match the source dataset')
        if totals['controls'] != expected['groups']['unflagged_control_windows']['windows']:
            raise ValueError('Control count does not match the source dataset')
    for name in ('windows.csv', 'spots.csv'):
        (output/(name+'.tmp')).replace(output/name)
    totals.update(first_seed=min(seeds), last_seed=max(seeds), partial=partial,
                  feature_version=FEATURE_VERSION, status='complete')
    atomic_json(output/'coverage.json', totals)
    return totals


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base/'runs/predator-stuck-diagnostics-10000')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--cluster-radius', type=float, default=5.)
    parser.add_argument('--limit-games', type=int, help='Pilot only: analyze this many new games')
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--entrances', action='store_true', help='Add observed-entry and constructive escape checks to cached spot features')
    parser.add_argument('--base-features', type=Path, default=base/'runs/predator-stuck-spots-10000')
    parser.add_argument('--escape-max-nodes', type=int, default=256)
    parser.add_argument('--escape-angle-step', type=float, default=5.)
    args = parser.parse_args()
    if args.workers < 1 or args.cluster_radius <= 0 or (args.limit_games is not None and args.limit_games < 1):
        parser.error('workers, cluster-radius and limit-games must be positive')
    if args.escape_max_nodes < 1 or not 0 < args.escape_angle_step <= 90:
        parser.error('escape-max-nodes must be positive and escape-angle-step must be in (0, 90]')
    if args.output is None:
        args.output = base/('runs/predator-stuck-entrances-10000' if args.entrances else 'runs/predator-stuck-spots-10000')
    processor = process_game
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'game-features'
    cache.mkdir(exist_ok=True)
    settings = dict(feature_version=FEATURE_VERSION, input=str(args.input.resolve()),
        cluster_radius=args.cluster_radius, feature_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (HELPERS/'spot_features.py', HELPERS/'diagnostics.py')})
    if args.entrances:
        from entrance_features import ENTRANCE_VERSION, process_entrance_game
        base_settings = json.loads((args.base_features/'settings.json').read_text())
        if base_settings != settings:
            raise ValueError('Base features must use the same input, geometry sources and cluster radius')
        if args.output.resolve() == args.base_features.resolve():
            raise ValueError('Entrance output must be separate from the existing base features')
        settings.update(entrance_version=ENTRANCE_VERSION,
            base_features=str(args.base_features.resolve()),
            entrance_source_sha256=hashlib.sha256((HELPERS/'entrance_features.py').read_bytes()).hexdigest(),
            escape_max_nodes=args.escape_max_nodes, escape_angle_step=args.escape_angle_step)
        processor = partial(process_entrance_game, base_cache=args.base_features/'game-features',
                            max_nodes=args.escape_max_nodes, angle_step=args.escape_angle_step)
    settings_path = args.output/'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError('Feature settings/source changed: choose a new output directory')
    else:
        atomic_json(settings_path, settings)
    started = time.monotonic()
    completed = {int(p.name.split('-')[1].split('.')[0]) for p in cache.glob('seed-*.json.gz')}
    print(json.dumps(dict(status='starting', cached_games=len(completed), workers=args.workers)), flush=True)
    if not args.report_only:
        pending = {}
        submitted = 0
        last_progress = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            def harvest(block):
                nonlocal last_progress
                done, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
                for future in done:
                    seed = pending.pop(future)
                    result = future.result()
                    if result['seed'] != seed:
                        raise ValueError('Worker returned the wrong seed')
                    save_game(cache, result); completed.add(seed)
                if time.monotonic()-last_progress >= 10:
                    progress = dict(status='extracting_features', completed_games=len(completed),
                        elapsed_seconds=round(time.monotonic()-started, 1), pending_games=len(pending))
                    atomic_json(args.output/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    last_progress = time.monotonic()
            for bundle in input_games(args.input, completed, args.cluster_radius):
                seed = bundle[0]
                if seed in pending.values():
                    raise ValueError(f'Duplicate input seed {seed}')
                pending[pool.submit(processor, bundle)] = seed
                submitted += 1
                while len(pending) >= 2*args.workers:
                    harvest(True)
                harvest(False)
                if args.limit_games and submitted >= args.limit_games:
                    break
            while pending:
                harvest(True)
    expected_path = args.input/'diagnostic_analysis.json'
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else None
    if expected and 'coverage' not in expected:
        expected = None
    totals = export_tables(args.output, expected, partial=bool(args.limit_games))
    atomic_json(args.output/'progress.json', totals | dict(elapsed_seconds=round(time.monotonic()-started, 1)))
    print(json.dumps(totals), flush=True)
    if not args.features_only:
        from spot_report import make_report
        make_report(args.output)
        if args.entrances:
            from entrance_report import make_entrance_report
            make_entrance_report(args.output)


if __name__ == '__main__':
    main()

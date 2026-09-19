"""Read-only timing audit of development games and Linux worker processes.

Run outside a frozen campaign source tree. Never opens holdout case results.
CPU cores means observed CPU seconds / elapsed seconds, not allocated capacity.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def confined(path, root):
    return path.resolve().is_relative_to(root.resolve())


def cases(root, allowed):
    rows, active = [], []
    for meta_path in (root / 'evaluation-cache' / 'cases').glob('*/active.json'):
        if not confined(meta_path, root):
            continue
        meta = read(meta_path)
        if meta.get('seed') not in allowed or not meta.get('attempt'):
            continue
        attempt = Path(meta['attempt'])
        if not confined(attempt, root):
            continue
        result = read(attempt / 'result.json')
        if not result:
            progress = read(attempt / 'progress.json')
            active.append({k: progress.get(k) for k in
                           ('seed', 'engine', 'sim_time', 'alive', 'wall_seconds')})
            continue
        audit, diagnostics = result.get('audit', {}), result.get('diagnostics', {})
        row = {k: result.get(k) for k in ('seed', 'engine', 'status', 'sim_time',
               'wall_seconds', 'evaluator_cpu_seconds', 'policy_rpc_wall_seconds',
               'agent_decisions')}
        row.update(case_id=meta_path.parent.name,
                   decision_cpu_seconds=audit.get('decision_cpu_seconds'),
                   decision_wall_seconds=audit.get('decision_wall_seconds'),
                   collection_wall_seconds=diagnostics.get('collection_wall_seconds'),
                   clip_serialization_wall_seconds=diagnostics.get('clip_serialization_wall_seconds'),
                   bytes_on_disk=diagnostics.get('bytes_on_disk'),
                   result_at=(attempt / 'result.json').stat().st_mtime)
        samples = result.get('samples', [])
        row['initial_wall_seconds'] = samples[0].get('wall_seconds') if samples else None
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row['engine']].append(row)
    summaries = {}
    for engine, group in groups.items():
        good = [r for r in group if r['status'] in ('horizon', 'extinct')]
        totals = {key: sum(r.get(key) or 0 for r in good) for key in (
            'wall_seconds', 'sim_time', 'policy_rpc_wall_seconds', 'decision_cpu_seconds',
            'decision_wall_seconds', 'evaluator_cpu_seconds', 'collection_wall_seconds',
            'clip_serialization_wall_seconds', 'agent_decisions')}
        wall = max(totals['wall_seconds'], 1e-9)
        summaries[engine] = dict(completed=len(good), statuses=dict(Counter(r['status'] for r in group)),
            median_wall_seconds=statistics.median([r['wall_seconds'] for r in good]) if good else None,
            maximum_wall_seconds=max([r['wall_seconds'] for r in good], default=None),
            simulation_seconds_per_wall_second=totals['sim_time'] / wall,
            policy_rpc_wall_fraction=totals['policy_rpc_wall_seconds'] / wall,
            decision_cpu_wall_fraction=totals['decision_cpu_seconds'] / wall,
            policy_cpu_over_policy_wall=totals['decision_cpu_seconds'] / max(totals['decision_wall_seconds'], 1e-9),
            collection_wall_fraction=totals['collection_wall_seconds'] / wall,
            evaluator_cpu_wall_fraction=totals['evaluator_cpu_seconds'] / wall,
            policy_cpu_seconds_per_1000_decisions=1000*totals['decision_cpu_seconds'] / max(totals['agent_decisions'], 1),
            totals=totals)
    return dict(root=str(root), active=active, summary=summaries, cases=rows)


def proc_snapshot():
    output = {}
    ticks = os.sysconf('SC_CLK_TCK')
    for path in Path('/proc').glob('[0-9]*'):
        try:
            command = (path / 'cmdline').read_bytes().replace(b'\0', b' ')
            if b'experiment_case.py' in command:
                role = 'evaluator'
            elif b'multiprocessing.spawn' in command:
                role = 'policy'
            elif b'research_study.py' in command:
                role = 'study'
            elif b'research_loop.py' in command or b'research-parallel.py' in command:
                role = 'supervisor'
            else:
                continue
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            io = dict(line.split(': ', 1) for line in (path / 'io').read_text().splitlines())
            sched = [int(n) for n in (path / 'schedstat').read_text().split()]
            output[int(path.name)] = dict(role=role, cpu=(int(stat[11])+int(stat[12]))/ticks,
                start=stat[19], rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),
                write_bytes=int(io['write_bytes']), read_bytes=int(io['read_bytes']),
                runqueue_wait_ns=sched[1])
        except (OSError, ValueError, IndexError):
            continue
    return output


def processes(seconds):
    first, started = proc_snapshot(), time.monotonic()
    time.sleep(seconds)
    second, elapsed = proc_snapshot(), time.monotonic()-started
    by_role = defaultdict(lambda: dict(count=0, cpu_cores=0., rss_bytes=0, read_bytes=0,
                                      write_bytes=0, runqueue_wait_seconds=0.))
    for pid, now in second.items():
        if pid not in first or first[pid]['start'] != now['start']:
            continue
        old, row = first[pid], by_role[now['role']]
        row['count'] += 1
        row['cpu_cores'] += (now['cpu']-old['cpu'])/elapsed
        row['rss_bytes'] += now['rss_bytes']
        for key in ('read_bytes', 'write_bytes'):
            row[key] += now[key]-old[key]
        row['runqueue_wait_seconds'] += (now['runqueue_wait_ns']-old['runqueue_wait_ns'])/1e9
    cgroup = {}
    for name in ('cpu.max', 'cpu.weight', 'cpu.stat', 'cpu.pressure',
                 'memory.current', 'memory.max', 'memory.events', 'io.pressure', 'cpuset.cpus.effective'):
        try:
            cgroup[name] = (Path('/sys/fs/cgroup') / name).read_text().strip()
        except OSError:
            pass
    return dict(sample_seconds=elapsed, by_role=dict(by_role),
                matched_start=len(first), matched_end=len(second), cgroup=cgroup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=Path('/workspace/research-night-1'))
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--sample-seconds', type=float, default=20.)
    args = parser.parse_args()
    if not 0 <= args.sample_seconds <= 30:
        parser.error('sample-seconds must be between 0 and 30')
    report = dict(at=datetime.now(timezone.utc).isoformat(), processes=processes(args.sample_seconds))
    if args.aggregate:
        evaluation = read(args.campaign / 'config.json').get('evaluation', {})
        allowed = set().union(*(evaluation.get(k, []) for k in
                              ('screen_seeds', 'comparison_seeds', 'major_seeds')))
        allowed -= set(evaluation.get('holdout_seeds', []))
        roots = [args.campaign] + [args.campaign.parent / 'research-capture-fleet' / task
            for task in ('guide-delivery', 'bait-continuity', 'capture-allocation')]
        report['campaigns'] = [cases(root, allowed) for root in roots]
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()

"""Read-only terminal summary of a campaign; never reads final holdout results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if default is None else default


def age(path, now):
    try:
        return max(0, now-Path(path).stat().st_mtime)
    except OSError:
        return float('inf')


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def render(out):
    out, now = Path(out), time.time()
    state, config, budget = (read(out/name) for name in ('state.json', 'config.json', 'budget.json'))
    if not state:
        raise SystemExit(f'Cannot read campaign state: {out}')
    b, schedule = config.get('budget', {}), config.get('schedule', {})
    elapsed = max(0, now-(state.get('started_at') or now))
    lines = ['SURVIVAL RESEARCH - '+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        f"Status: {state.get('status')} | Generation {state.get('major')}.{state.get('sub')} | Stage: {state.get('stage')}",
        f"Elapsed: {duration(elapsed)} | Limit: {b.get('max_hours', '?')} h | Last checkpoint: {age(out/'state.json', now):.0f}s ago",
        f"Schedule: up to {schedule.get('major_generations')} major generations, {schedule.get('subgenerations')} subgenerations each; adaptive stopping",
        f"LLM reviews: {state.get('agent_calls', 0)}/{config.get('agent', {}).get('max_calls', '?')}"]
    estimated = budget.get('estimated_compute_usd', 0)+budget.get('external_spend_usd', 0)
    lines += [f"Budget: ${b.get('total_usd', 0):.2f} | Estimated compute/setup: ${estimated:.2f} | LLM allowance reserved: ${budget.get('agent_reserved_usd', 0):.2f} | Safety reserve: ${b.get('reserve_usd', 0):.2f}",
              'Cost figures are supervisor estimates, not a Runpod invoice.']
    if state.get('error'):
        lines.append('ERROR: '+str(state['error']))
    for name, step in state.get('steps', {}).items():
        if step.get('status') not in ('pending', 'running'):
            continue
        remaining = duration(step.get('deadline', now)-now)
        progress = read(out/'steps'/name/'progress.json')
        detail = f"{progress['completed']}/{progress['total']} runs finished" if 'total' in progress else 'working'
        study = read(out/'steps'/name/'study'/'report.json')
        if study:
            detail = f"{study.get('completed', 0)} candidates evaluated ({study.get('method', 'search')})"
        lines.append(f'Active step: {name} | {detail} | Step time left: {remaining}')

    best = state.get('best')
    if not best:
        lines.append('Best candidate: initial controls are still being evaluated')
    else:
        lines.append(f"Best candidate: {best.get('label', '')} [{best['id'][:12]}]")
        # Only authoritative development comparisons, never final reports.
        reports = [p for p in (out/'steps').glob('*/comparison.json') if not p.parent.name.startswith('final')]
        for path in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True):
            report = read(path)
            if report.get('control_equivalence') or report.get('engine', 'python') != 'python':
                continue
            summary = report.get('summaries', {}).get(best['id'])
            if summary:
                lines.append(f"  Python development ({len(report.get('seeds', []))} maps): mean survival {summary['mean_survival']:.1f}s | worst {summary['worst_survival']:.1f}s | mean score {summary['mean_score']:.2f}")
                break

    # Inspect only active development progress, excluding results and holdout seeds.
    development = set(config.get('evaluation', {}).get('major_seeds', []))
    active = []
    for path in (out/'evaluation-cache'/'cases').glob('*/active.json'):
        item = read(path)
        if item.get('seed') not in development or not item.get('attempt'):
            continue
        attempt = Path(item['attempt'])
        if not attempt.resolve().is_relative_to(out.resolve()) or (attempt/'result.json').exists():
            continue
        progress_path = attempt/'progress.json'
        progress = read(progress_path)
        if progress and age(progress_path, now) < 90:
            active.append(progress)
    if active:
        lines.append(f'Active development runs with recent progress: {len(active)}')
        for item in sorted(active, key=lambda x: (x.get('seed', 0), x.get('case_id', '')))[:16]:
            lines.append(f"  {item.get('case_id', '')[:8]} | {item.get('engine', '?'):7} | seed {item.get('seed')} | sim {item.get('sim_time', 0):7.1f}/{item.get('horizon', 3000):.0f}s | alive {item.get('alive', '?')} | wall {duration(item.get('wall_seconds', 0))}")
    lines.append('Holdout: '+('selection frozen; final stage may evaluate it' if (out/'final-selection.json').exists() else 'unopened'))

    guard = read(out.parent/'research-shutdown.json')
    if guard:
        cutoff = datetime.fromtimestamp(guard['deadline_epoch'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        pid = None
        try:
            for line in (out.parent/'research-shutdown.jsonl').read_text().splitlines():
                event = json.loads(line)
                if event.get('event') == 'armed':
                    pid = event.get('pid')
            running = bool(pid and b'runpod_shutdown_guard.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
        except (OSError, ValueError):
            running = False
        lines.append(f"Shutdown watchdog: {'running' if running else 'not observed running'} | Hard cutoff: {cutoff}")
    events = list(state.get('events', {}).items())[-3:]
    if events:
        lines.append('Recent research events:')
        lines.extend(f"  {name}: {event.get('kind', '')}" for name, event in events)
    lines.extend(['', f'Campaign files: {out}', 'Detailed reasoning: journal.md; individual reviews/*/proposal.json'])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(render(parser.parse_args().out))

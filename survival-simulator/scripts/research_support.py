"""Pure bookkeeping for the research supervisor; no policy or simulator imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics

VERSION = 4
CODE_DIRS = ('models', 'src', 'scripts', 'tests', 'fastsim')
CODE_SUFFIXES = {'.py', '.json', '.sh', '.cmd', '.cpp', '.hpp', '.md', '.so', '.pyd'}
EDITABLE_FILES = {'models/core.py', 'models/experimental_policy.py', 'models/experiment_config.py'}
EDITABLE_DIRS = ('models/survival/', 'models/exploration/', 'models/entrapment/')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    from scripts.research_io import atomic_json
    atomic_json(path, value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def research_case(snapshot, environment, config, baseline, seed, policy_seed, diagnostics=None, engine='python', *, infrastructure_retries=0):
    from scripts.research_diagnostics import DEFAULTS, VERSION as TELEMETRY_VERSION
    from scripts.simulation_backend import identity
    request = dict(mode='trapping', seconds=3000, seed=seed, policy_seed=policy_seed,
                   baseline=baseline, config=config, diagnostics={**DEFAULTS, **(diagnostics or {})},
                   telemetry_version=TELEMETRY_VERSION, engine=engine, engine_build=identity(engine))
    if infrastructure_retries:
        request['infrastructure_retries'] = infrastructure_retries
    request['case_id'] = digest(dict(request, snapshot=snapshot, environment=environment, protocol=VERSION))
    return request


def editable(name):
    path = Path(name)
    return (not path.is_absolute() and '..' not in path.parts and '\\' not in name
            and path.suffix in {'.py', '.json'}
            and (name in EDITABLE_FILES or name.startswith(EDITABLE_DIRS)))


def manifest(root, *, native=True):
    """Include untracked working files, but never environments, recordings or Git."""
    root = Path(root)
    paths = [root / name for name in ('run.cmd', 'requirements.txt', 'AGENTS.md')]
    paths.extend(root.glob('*.py'))
    paths.extend(p for p in (root/'docs').rglob('*') if p.suffix in {'.json', '.md'})
    paths.extend(root.glob('*.md'))
    for directory in CODE_DIRS:
        paths.extend(p for p in (root / directory).rglob('*') if p.suffix in CODE_SUFFIXES
                     and '__pycache__' not in p.parts)
    result = {}
    for path in sorted(set(paths)):
        if not native and (path.suffix in {'.so', '.pyd'} or path.name == 'build-info.json'):
            continue
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and root in p.parents):
            raise ValueError(f'Symlink in snapshot: {path}')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def copy_source(source, target, files=None):
    source, target = Path(source), Path(target)
    for name in files or manifest(source):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)


def assert_snapshot(root, expected):
    actual = manifest(root)
    if actual != expected:
        changes = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise RuntimeError(f'Frozen source changed: {changes[:20]}')


def changed_policy(before, after):
    changes = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    forbidden = [name for name in changes if not editable(name)]
    if forbidden:
        raise ValueError(f'Protected source changes: {forbidden}')
    return changes


def apply_proposed_edits(root, edits):
    """Validate all literal edits before writing a disposable candidate draft."""
    root = Path(root).resolve()
    if not isinstance(edits, list) or len(edits) > 32:
        raise ValueError('At most 32 literal policy edits are allowed')
    pending, size = {}, 0
    for item in edits:
        if (not isinstance(item, dict) or set(item) != {'path', 'old_text', 'new_text'}
                or any(not isinstance(v, str) for v in item.values())):
            raise ValueError('Invalid literal policy edit')
        name, old, new = item['path'], item['old_text'], item['new_text']
        if not editable(name):
            raise ValueError(f'Protected or invalid edit path: {name}')
        target = root/name
        if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root and root in p.parents):
            raise ValueError('Symlinks are not editable')
        if not target.resolve().is_relative_to(root):
            raise ValueError('Edit escapes draft')
        size += len(old.encode('utf-8'))+len(new.encode('utf-8'))
        if size > 512*1024:
            raise ValueError('Proposed edits exceed 512 KiB')
        if name not in pending:
            pending[name] = target.read_text(encoding='utf-8') if target.exists() else None
        current = pending[name]
        if not old:
            if current is not None:
                raise ValueError('Empty old_text is allowed only when creating a new file')
            pending[name] = new
        else:
            if current is None or current.count(old) != 1:
                raise ValueError(f'old_text must match exactly once: {name}')
            pending[name] = current.replace(old, new, 1)
    # All edits were checked. The caller freezes this draft only after success;
    # a crash or I/O failure cannot publish a partly applied candidate.
    for name, content in pending.items():
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    return sorted(pending)


def validate_config(config):
    for name in ('proposal_only', 'legacy_landlock'):
        if type(config['agent'].get(name, False)) is not bool:
            raise ValueError(f'agent.{name} must be boolean')
    if config['agent'].get('legacy_landlock') and not config['agent'].get('proposal_only'):
        raise ValueError('Legacy Landlock is supported only for read-only proposals')
    budget, schedule, evaluation = config['budget'], config['schedule'], config['evaluation']
    for name in ('total_usd', 'max_hours', 'hourly_rate_usd'):
        value = budget[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'budget.{name} must be explicitly set to a positive finite number')
    for name in ('reserve_usd', 'external_spend_usd', 'agent_call_reserve_usd'):
        if isinstance(budget[name], bool) or not isinstance(budget[name], (int, float)) or not math.isfinite(budget[name]) or budget[name] < 0:
            raise ValueError(f'Invalid budget.{name}')
    if budget['reserve_usd'] + budget['external_spend_usd'] >= budget['total_usd']:
        raise ValueError('No usable compute budget')
    for name in ('major_generations', 'subgenerations', 'min_subgenerations', 'patience'):
        if type(schedule[name]) is not int or schedule[name] < 1:
            raise ValueError(f'Invalid schedule.{name}')
    if not 1 <= schedule['min_subgenerations'] <= schedule['subgenerations'] <= 5:
        raise ValueError('Invalid subgeneration bounds')
    for name in ('preflight_seconds', 'agent_seconds', 'sub_search_seconds', 'comparison_seconds',
                 'bo_seconds', 'final_reserve_seconds', 'minimum_step_seconds'):
        if not isinstance(schedule[name], (int, float)) or not math.isfinite(schedule[name]) or schedule[name] <= 0:
            raise ValueError(f'Invalid schedule.{name}')
    if schedule['final_reserve_seconds'] >= budget['max_hours'] * 3600:
        raise ValueError('Final reserve consumes entire campaign')
    if evaluation['horizon_seconds'] != 3000:
        raise ValueError('Research promotion requires the full 3000-second horizon')
    if evaluation.get('search_engine', 'python') not in ('python', 'fastsim'):
        raise ValueError('Unknown search engine')
    if evaluation.get('comparison_engine', 'python') != 'python' or evaluation.get('holdout_engine', 'python') != 'python':
        raise ValueError('Promotion and final holdout require the reference Python engine')
    sets = []
    for name in ('screen_seeds', 'comparison_seeds', 'major_seeds', 'holdout_seeds'):
        seeds = evaluation[name]
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f'Invalid {name}')
        sets.append(set(seeds))
    independent = evaluation.get('independent_confirmation', False)
    if type(independent) is not bool:
        raise ValueError('independent_confirmation must be boolean')
    nested = (not sets[0] & sets[2] and sets[1] <= sets[2]) if independent else sets[0] <= sets[1] <= sets[2]
    if not nested or (sets[0] | sets[1] | sets[2]) & sets[3]:
        raise ValueError('Development seeds must nest; final holdout must be disjoint')
    if type(evaluation['workers']) is not int or evaluation['workers'] < 0:
        raise ValueError('Invalid workers')
    if (not isinstance(evaluation['case_timeout_seconds'], (int, float))
            or not math.isfinite(evaluation['case_timeout_seconds']) or evaluation['case_timeout_seconds'] <= 0
            or type(evaluation['policy_seed']) is not int or evaluation['policy_seed'] < 0):
        raise ValueError('Invalid evaluation timeout/seed')
    for value in (config['agent']['max_calls'], config['search']['focused_trials'], config['search']['bo_trials'],
                  config['search']['max_dimensions'], config['search']['max_variants']):
        if type(value) is not int or value < 1:
            raise ValueError('Call, trial and dimension limits must be positive integers')
    if not 1 <= config['search']['max_dimensions'] <= 24:
        raise ValueError('BO supports 1-24 selected scalar dimensions')
    if not 0 < config['promotion']['confidence'] < 1:
        raise ValueError('Invalid promotion confidence')
    if any(not math.isfinite(config['promotion'][k]) or config['promotion'][k] < 0
           for k in ('minimum_mean_gain_seconds', 'maximum_worst_loss_seconds')):
        raise ValueError('Invalid promotion thresholds')
    if not config['upstream']['branch'].startswith('survival-simulator/'):
        raise ValueError('Unexpected upstream branch')
    from scripts.research_diagnostics import validate_settings
    validate_settings(config.get('diagnostics', {}))
    storage = config['storage']
    for name in ('campaign_gib', 'final_reserve_gib', 'minimum_free_gib'):
        if type(storage[name]) not in (int, float) or not math.isfinite(storage[name]) or storage[name] <= 0:
            raise ValueError(f'Invalid storage.{name}')
    if storage['final_reserve_gib'] >= storage['campaign_gib']:
        raise ValueError('Final storage reserve consumes entire campaign')
    return config


def storage_status(root, config, workers, *, final=False):
    """Leave headroom for active evaluators to finish saving their diagnostics."""
    limits = config['storage']
    from scripts.research_io import directory_bytes
    used = directory_bytes(root)
    free = shutil.disk_usage(root).free
    gib = 1024**3
    reserve = 0 if final else limits['final_reserve_gib']*gib
    # Includes one per-run telemetry allowance per worker plus result/agent logs.
    in_flight = workers*config.get('diagnostics', {}).get('disk_mb', 128)*1024**2 + 128*1024**2
    allowed = limits['campaign_gib']*gib-reserve
    return dict(used_bytes=used, free_bytes=free, limit_bytes=allowed,
                in_flight_reserve_bytes=in_flight, final=final,
                exhausted=used+in_flight >= allowed or free-in_flight < limits['minimum_free_gib']*gib)


def remaining_seconds(config, state, now, *, final=False):
    """Charge elapsed calendar time, including downtime, coding and reporting."""
    b = config['budget']
    elapsed = max(0., now - state['started_at'])
    spent = b['external_spend_usd'] + b['reserve_usd'] + state.get('agent_reserved_usd', 0.)
    money_seconds = (b['total_usd'] - spent) / b['hourly_rate_usd'] * 3600 - elapsed
    remaining = min(b['max_hours'] * 3600 - elapsed, money_seconds)
    return max(0., remaining - (0 if final else config['schedule']['final_reserve_seconds']))


def promotion(previous, candidate, rules, horizon=3000):
    """Paired, complete comparisons only. No survival/score selection on holdout."""
    if not previous or not candidate or any(r['status'] not in ('horizon', 'extinct') for r in previous + candidate):
        return dict(promote=False, reason='incomplete, failed, or unpaired comparison')
    keys = lambda rows: [(r['seed'], r['policy_seed']) for r in rows]
    old = sorted(previous, key=lambda r: (r['seed'], r['policy_seed']))
    new = sorted(candidate, key=lambda r: (r['seed'], r['policy_seed']))
    if keys(old) != keys(new) or len(set(keys(old))) != len(old):
        return dict(promote=False, reason='incomplete, failed, or unpaired comparison')
    deltas = [min(horizon, b['sim_time']) - min(horizon, a['sim_time']) for a, b in zip(old, new)]
    rng = random.Random(971)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    alpha = (1 - rules['confidence']) / 2
    lower, upper = means[int(alpha * len(means))], means[min(len(means)-1, int((1-alpha)*len(means)))]
    mean = statistics.mean(deltas)
    worst_delta = min(r['sim_time'] for r in new) - min(r['sim_time'] for r in old)
    rank_delta = (mean + .25 * worst_delta) / horizon
    survival_ok = (lower > 0 and mean >= rules['minimum_mean_gain_seconds'] and rank_delta > 0)
    # At the ceiling, paired score improvement can still justify promotion.
    scores = [b['score']-a['score'] for a, b in zip(old, new)]
    ceiling = all(r['sim_time'] >= horizon - 1e-6 for r in old + new)
    score_means = sorted(statistics.mean(rng.choices(scores, k=len(scores))) for _ in range(2000))
    score_lower = score_means[int(alpha * len(score_means))]
    accepted = (survival_ok or ceiling and score_lower > 0) and worst_delta >= -rules['maximum_worst_loss_seconds']
    return dict(promote=accepted, reason='paired improvement' if accepted else 'insufficient evidence or regression',
                cases=len(old), mean_survival_delta=mean, worst_survival_delta=worst_delta,
                primary_rank_delta=rank_delta, survival_interval=[lower, upper],
                mean_score_delta=statistics.mean(scores), score_lower_bound=score_lower,
                interval_note='Descriptive paired bootstrap; adaptive selection and few maps limit certainty')

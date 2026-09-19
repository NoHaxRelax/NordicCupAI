"""Persistent major/minor research supervisor. Preparation never launches work.

Use `prepare`, edit campaign/config.json, then explicitly invoke `run` later.
Experiments, git fetches and Codex jobs are only started by the run command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import difflib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_support import (
    VERSION, read, write, digest, manifest, copy_source, assert_snapshot,
    changed_policy, editable, validate_config, remaining_seconds, promotion,
    research_case, storage_status, apply_proposed_edits,
)
from scripts.optimize_policy import study_lock, versions, run_case, summarize
from scripts.search_resources import allocation

ENV = dict(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy')


class Paused(Exception):
    pass


class NoBudget(Exception):
    pass


class NoStorage(Exception):
    pass


def reference(snapshot, config, *, baseline=False, label='candidate'):
    value = dict(snapshot=snapshot, config=config, baseline=baseline)
    return dict(value, id=digest(value), label=label)


def prepare(out, config_path):
    if out.exists():
        raise ValueError('Preparation requires a new campaign directory; resume with run')
    if out == ROOT or any(p in out.parents for p in (ROOT/'models', ROOT/'src', ROOT/'scripts', ROOT/'tests')):
        raise ValueError('Campaign outputs cannot be inside source directories')
    files = manifest(ROOT)
    identity = digest(files)
    out.mkdir(parents=True)
    code = out/'snapshots'/identity/'code'
    copy_source(ROOT, code, files)
    assert_snapshot(code, files)
    write(code.parent/'manifest.json', files)
    write(code.parent/'metadata.json', dict(label='initial working tree', snapshot=identity))
    write(out/'config.json', read(config_path))
    for name in ('research_agent_prompt.md', 'research_agent_result.schema.json'):
        (out/name).write_bytes((ROOT/'docs'/name).read_bytes())
    write(out/'state.json', dict(version=VERSION, status='prepared', stage='preflight',
          major=1, sub=1, stagnant_subs=0, stagnant_majors=0, initial_snapshot=identity,
          best=None, original=None, started_at=None, agent_calls=0, agent_reserved_usd=0.,
          steps={}, events={}, last_upstream=None))
    print(f'Prepared {out}. No tests, imports of policy code, git fetches, agents or games launched.')


class Supervisor:
    def __init__(self, out):
        self.out = out
        self.config = validate_config(read(out/'config.json'))
        self.state = read(out/'state.json')
        if self.state['version'] != VERSION:
            raise ValueError('Supervisor protocol changed; prepare a new campaign')
        self.stopping = False
        self.environment = versions()
        provenance = dict(config=self.config, environment=self.environment,
                          prompt=(out/'research_agent_prompt.md').read_text(encoding='utf-8'),
                          schema=read(out/'research_agent_result.schema.json'))
        protocol = out/'protocol.json'
        if protocol.exists() and read(protocol) != provenance:
            raise ValueError('Campaign configuration, environment or prompt changed; use a new campaign')
        write(protocol, provenance)
        trusted = self.snapshot(self.state['initial_snapshot'])
        trusted_files = read(trusted.parent/'manifest.json')
        # The running orchestration/evaluation code must be the frozen trusted code.
        live = manifest(ROOT)
        for name, checksum in trusted_files.items():
            if not editable(name) and live.get(name) != checksum:
                raise ValueError(f'Trusted launch source changed: {name}')
        self.workers = self.config['evaluation']['workers'] or allocation()['suggested_workers']
        from scripts.simulation_backend import identity as engine_identity
        engine_identity(self.config['evaluation'].get('search_engine', 'python'))
        write(out/'resources.json', dict(allocation(), selected_workers=self.workers))

    def snapshot(self, identity):
        root = self.out/'snapshots'/identity/'code'
        expected = read(root.parent/'manifest.json')
        if digest(expected) != identity:
            raise RuntimeError('Snapshot manifest does not match its content address')
        assert_snapshot(root, expected)
        return root

    def freeze(self, source, label):
        files = manifest(source)
        initial = read(self.out/'snapshots'/self.state['initial_snapshot']/'manifest.json')
        changed_policy(initial, files)
        identity = digest(files)
        destination = self.out/'snapshots'/identity/'code'
        if not destination.exists():
            copy_source(source, destination, files)
            assert_snapshot(destination, files)
            write(destination.parent/'manifest.json', files)
            write(destination.parent/'metadata.json', dict(snapshot=identity, label=label))
        self.snapshot(identity)
        return identity

    def save(self):
        write(self.out/'state.json', self.state)
        if self.state['best']:
            write(self.out/'best.json', self.state['best'])
        elapsed = max(0., time.time()-(self.state['started_at'] or time.time()))
        b = self.config['budget']
        write(self.out/'budget.json', dict(elapsed_seconds=elapsed,
              estimated_compute_usd=elapsed*b['hourly_rate_usd']/3600,
              agent_reserved_usd=self.state['agent_reserved_usd'],
              external_spend_usd=b['external_spend_usd'], reserve_usd=b['reserve_usd'],
              total_limit_usd=b['total_usd'],
              note='Calendar-time estimate plus conservative per-agent reservation, not a provider invoice'))
        self._last_checkpoint = time.monotonic()

    def pulse(self):
        if time.monotonic()-getattr(self, '_last_checkpoint', 0.) >= 10:
            self.save()

    def event(self, key, kind, details):
        self.state['events'].setdefault(key, dict(kind=kind, details=details, time=time.time()))
        self.save()
        lines = ['# Cumulative research journal', '', 'Development evidence only; hypotheses are labelled.', '']
        for name, entry in self.state['events'].items():
            lines.extend([f'## {name}: {entry["kind"]}', '', '```json',
                          json.dumps(entry['details'], indent=2), '```', ''])
        (self.out/'journal.md').write_text('\n'.join(lines), encoding='utf-8')

    def check_stop(self):
        if self.stopping or (self.out/'STOP').exists():
            raise Paused()
        final = self.state['stage'] == 'final'
        if (time.monotonic()-getattr(self, '_last_storage_check', 0.) >= 10
                or final != getattr(self, '_storage_final', None)):
            usage = storage_status(self.out, self.config, getattr(self, 'workers', 1), final=final)
            write(self.out/'storage.json', usage)
            self._last_storage_check, self._storage_final = time.monotonic(), final
            self._storage_exhausted = usage['exhausted']
        if self._storage_exhausted:
            raise NoStorage('Campaign storage allowance or free-space reserve reached')

    def step(self, key, seconds, *, final=False):
        self.check_stop()
        folder = self.out/'steps'/key
        if key not in self.state['steps']:
            left = remaining_seconds(self.config, self.state, time.time(), final=final)
            if left < self.config['schedule']['minimum_step_seconds']:
                raise NoBudget()
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'control').mkdir(exist_ok=True)
            self.state['steps'][key] = dict(deadline=time.time()+min(seconds, left), status='pending')
            self.save()
        return folder, self.state['steps'][key]

    def execute(self, key, seconds, builder, *, cwd=None, agent=False, stdin=None, final=False):
        if agent and (self.out/'final-selection.json').exists():
            raise RuntimeError('Final selection is frozen; no more agent jobs')
        folder, step = self.step(key, seconds, final=final)
        result_path = folder/'process-result.json'
        if result_path.exists():
            step['status'] = 'finished'
            return read(result_path)
        if step['status'] == 'pending':
            if time.time() >= step['deadline'] or remaining_seconds(self.config, self.state, time.time(), final=final) <= 0:
                value = dict(status='interrupted', returncode=None, error='Saved step deadline exhausted before launch')
                write(result_path, value)
                step['status'] = 'finished'
                self.save()
                return value
            if agent:
                b = self.config['budget']
                if self.state['agent_calls'] >= self.config['agent']['max_calls']:
                    return dict(status='skipped', returncode=None, error='Agent call cap reached')
                prospective = dict(self.state, agent_reserved_usd=self.state['agent_reserved_usd']+b['agent_call_reserve_usd'])
                if remaining_seconds(self.config, prospective, time.time()) < seconds:
                    raise NoBudget()
                self.state['agent_calls'] += 1
                self.state['agent_reserved_usd'] += b['agent_call_reserve_usd']
                self.save()
            command = builder(folder, step['deadline'])
            write(folder/'process-request.json', dict(command=command, cwd=str(cwd or ROOT),
                  deadline=step['deadline'], env=ENV, agent=agent, stdin=str(stdin) if stdin else None))
            (folder/'control'/'heartbeat').touch()
            step['status'] = 'running'
            self.save()
            launcher = self.snapshot(self.state['initial_snapshot'])/'scripts/research_process.py'
            try:
                subprocess.Popen([sys.executable, '-B', str(launcher), str(folder/'process-request.json')],
                    cwd=self.out, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=os.name != 'nt')
            except OSError as exc:
                write(result_path, dict(status='error', returncode=None, error=str(exc)))
        # A resumed coordinator observes the existing bounded wrapper, never duplicates it.
        while not result_path.exists():
            (folder/'control'/'heartbeat').touch()
            try:
                self.check_stop()
            except (Paused, NoStorage):
                (folder/'control'/'STOP').touch()
                raise
            if remaining_seconds(self.config, self.state, time.time(), final=final) <= 0 or time.time() >= step['deadline']:
                (folder/'control'/'STOP').touch()
            if time.time() > step['deadline']+120:
                write(result_path, dict(status='error', returncode=None, error='Wrapper missing or abandoned; attempt not retried'))
                break
            self.pulse()
            time.sleep(1)
        step['status'] = 'finished'
        self.save()
        return read(result_path)

    def catalog(self, snapshot, key):
        key = 'catalog-' + snapshot[:24]
        code = self.snapshot(snapshot)
        target = self.out/'steps'/key/'study'/'catalog.json'
        def command(folder, deadline):
            write(folder/'request.json', {'mode': 'catalog'})
            return [sys.executable, '-B', str(code/'scripts/research_study.py'),
                    '--request', str(folder/'request.json'), '--out', str(folder/'study'),
                    '--control', str(folder/'control')]
        result = self.execute(key, self.config['schedule']['preflight_seconds'], command, cwd=code)
        self.snapshot(snapshot)
        if result.get('returncode') != 0 or not target.exists():
            raise RuntimeError(f'Catalog failed; see {target.parent.parent}')
        return read(target)

    def checks(self, snapshot, key):
        key = 'checks-' + snapshot[:24]
        code = self.snapshot(snapshot)
        result = self.execute(key, self.config['schedule']['preflight_seconds'], lambda f, d:
            [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py'], cwd=code)
        self.snapshot(snapshot)
        return result.get('returncode') == 0 and result.get('status') == 'complete'

    def evaluate(self, key, refs, seeds, *, final=False, control_equivalence=False):
        # Python uses object-address hashes: independently spawned identical
        # policies can have different trajectories. Native ordering provides a
        # repeatable plumbing check, never promotion or holdout evidence.
        if control_equivalence and (final or key != 'initial-control-equivalence'):
            raise ValueError('Native control checks cannot select candidates or open holdout')
        engine = self.config['evaluation']['search_engine'] if control_equivalence else 'python'
        folder, step = self.step(key, self.config['schedule']['comparison_seconds'] if not final
                                 else self.config['schedule']['final_reserve_seconds'], final=final)
        result_path = folder/'comparison.json'
        if result_path.exists():
            return read(result_path)
        plan = dict(refs=refs, seeds=seeds, environment=self.environment, horizon=3000,
                    engine=engine, control_equivalence=control_equivalence,
                    policy_seed=self.config['evaluation']['policy_seed'])
        if (folder/'plan.json').exists() and read(folder/'plan.json') != plan:
            raise ValueError('Frozen comparison plan changed')
        write(folder/'plan.json', plan)
        # A crashed coordinator must not revive abandoned evaluators by touching
        # their old heartbeat. Give every attempt a fresh cancellation directory.
        prior_control = step.get('evaluation_control')
        if prior_control:
            Path(prior_control, 'STOP').touch()
            # The observation worker timeout is 120s. Allow old evaluators to
            # leave that call before starting replacement games after a crash.
            until = step.setdefault('recovery_until', time.time()+135)
            self.save()
            while time.time() < until:
                self.check_stop()
                self.pulse()
                time.sleep(1)
        control = folder/'control'/uuid.uuid4().hex
        control.mkdir(parents=True)
        step['evaluation_control'] = str(control)
        step.pop('recovery_until', None)
        self.save()
        control.joinpath('heartbeat').touch()
        if time.time() >= step['deadline'] or remaining_seconds(self.config, self.state, time.time(), final=final) <= 0:
            control.joinpath('STOP').touch()
        cases = {}
        by_ref = {}
        for ref in refs:
            code = self.snapshot(ref['snapshot'])
            ids = []
            for seed in seeds:
                request = research_case(ref['snapshot'], self.environment, ref['config'], ref['baseline'], seed, plan['policy_seed'], self.config.get('diagnostics'), engine,
                    infrastructure_retries=self.config.get('recovery', {}).get('case_retries', 0))
                cases[request['case_id']] = (request, code)
                ids.append(request['case_id'])
            by_ref[ref['id']] = ids
        results = {}
        pool = ThreadPoolExecutor(max_workers=self.workers)
        pending = {pool.submit(run_case, request, self.out/'evaluation-cache', control,
                    self.config['evaluation']['case_timeout_seconds'], runtime_root=code): cid
                   for cid, (request, code) in cases.items()}
        try:
            while pending:
                control.joinpath('heartbeat').touch()
                self.check_stop()
                if time.time() >= step['deadline'] or remaining_seconds(self.config, self.state, time.time(), final=final) <= 0:
                    control.joinpath('STOP').touch()
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    cid = pending.pop(future)
                    results[cid] = future.result()
                self.pulse()
                write(folder/'progress.json', dict(completed=len(results), total=len(cases)))
        finally:
            control.joinpath('STOP').touch()
            pool.shutdown(wait=True, cancel_futures=True)
        for ref in refs:
            self.snapshot(ref['snapshot'])
        rows = {rid: [results[cid] for cid in ids] for rid, ids in by_ref.items()}
        complete = all(r['status'] in ('horizon', 'extinct') for group in rows.values() for r in group)
        summaries = {rid: summarize(group, 3000) for rid, group in rows.items()
                     if all(r['status'] in ('horizon', 'extinct') for r in group)}
        report = dict(refs=refs, seeds=seeds, rows=rows, summaries=summaries, complete=complete,
                      engine=engine, control_equivalence=control_equivalence)
        write(result_path, report)
        step['status'] = 'finished'
        self.save()
        return report

    def winner(self, report):
        if report.get('control_equivalence') or report.get('engine', 'python') != 'python':
            raise ValueError('Only authoritative Python comparisons may select a winner')
        incumbent = report['refs'][0]
        decisions = []
        winner = incumbent
        best_rank = report['summaries'].get(incumbent['id'], {}).get('rank')
        for challenger in report['refs'][1:]:
            decision = promotion(report['rows'][incumbent['id']], report['rows'][challenger['id']], self.config['promotion'])
            decisions.append(dict(candidate=challenger['id'], **decision))
            rank = report['summaries'].get(challenger['id'], {}).get('rank')
            if decision['promote'] and rank is not None and (best_rank is None or rank > best_rank):
                winner, best_rank = challenger, rank
        return winner, decisions

    def upstream(self, key):
        folder = self.out/'upstream'/key
        saved = folder/'review.json'
        if saved.exists():
            return read(saved)
        folder.mkdir(parents=True, exist_ok=True)
        settings = self.config['upstream']
        mirror = self.out/'upstream.git'
        timeout = settings['fetch_timeout_seconds']
        def git_step(suffix, arguments, cwd=ROOT):
            result = self.execute(key+'.git-'+suffix, timeout, lambda f, d: ['git', *arguments], cwd=cwd)
            if result.get('returncode') != 0:
                raise RuntimeError(f'Git {suffix} failed; see steps/{key}.git-{suffix}/console.log')
            return (self.out/'steps'/(key+'.git-'+suffix)/'console.log').read_text(encoding='utf-8', errors='replace').strip()
        try:
            url = git_step('remote', ['remote', 'get-url', settings['remote']])
            # Never persist an embedded HTTP password/token as a subprocess argument.
            if re.match(r'https?://[^/]*@', url):
                raise ValueError('Use Git credential storage, not credentials embedded in remote URL')
            if not mirror.exists():
                git_step('init', ['init', '--bare', str(mirror)])
            git_step('fetch', ['-C', str(mirror), 'fetch', '--no-tags', url,
                     f'+refs/heads/{settings["branch"]}:refs/heads/review'])
            sha = git_step('head', ['-C', str(mirror), 'rev-parse', 'refs/heads/review'])
            if not re.fullmatch('[0-9a-f]{40,64}', sha):
                raise ValueError('Unexpected upstream commit response')
            prior = self.state['last_upstream'] or settings['last_reviewed_commit']
            listing = git_step('files', ['-C', str(mirror), 'ls-tree', '-rl', sha, '--',
                                        'survival-simulator/models', 'survival-simulator/docs'])
            selected, total = [], 0
            for line in listing.splitlines():
                metadata, name = line.split('\t', 1)
                mode, kind, _, size = metadata.split()
                if kind != 'blob' or mode not in ('100644', '100755') or not size.isdigit():
                    continue
                if Path(name).suffix not in ('.py', '.json', '.md') or int(size) > 2_000_000:
                    continue
                if '..' in Path(name).parts or not name.startswith('survival-simulator/'):
                    raise ValueError('Unsafe upstream path')
                total += int(size)
                selected.append(name)
            if total > 30_000_000 or not selected:
                raise ValueError('Upstream review source is empty or exceeds 30 MB')
            archive = folder/'source.tar'
            git_step('archive', ['-C', str(mirror), 'archive', '--format=tar', '--output', str(archive), sha, '--', *selected])
            with tarfile.open(archive) as bundle:
                for entry in bundle.getmembers():
                    if entry.isdir():
                        continue
                    if not entry.isfile() or entry.name not in selected:
                        raise ValueError('Unexpected archive member')
                    relative = Path(entry.name).relative_to('survival-simulator')
                    destination = folder/'source'/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(entry) as source:
                        destination.write_bytes(source.read())
            try:
                changes = git_step('diff', ['-C', str(mirror), 'diff', '--stat', prior, sha, '--', 'survival-simulator'])
                patch = git_step('patch', ['-C', str(mirror), 'diff', prior, sha, '--', 'survival-simulator/models'])
                (folder/'changes.patch').write_text(patch, encoding='utf-8')
            except RuntimeError:
                changes = 'Prior commit unavailable after fetch; review the complete pinned source.'
            review = dict(status='fetched', commit=sha, previous_commit=prior, branch=settings['branch'],
                          changes=changes, source=str(folder/'source'), patch=str(folder/'changes.patch'))
        except (RuntimeError, ValueError, OSError, tarfile.TarError) as exc:
            review = dict(status='unavailable', error=str(exc), branch=settings['branch'])
        write(saved, review)
        self.event(key+'.upstream', 'upstream review snapshot', review)
        return review

    def compare_upstream(self, key, review, incumbent):
        output = self.out/'upstream'/key/'decision.json'
        if output.exists():
            return read(output)['winner']
        winner = incumbent
        detail = {'reason': 'No compatible upstream modular entry point'}
        if review['status'] == 'fetched':
            raw = Path(review['source'])
            # Only a compatible modular source is directly runnable. Older flat
            # branches are presented to the coding review for selective adaptation.
            if (raw/'models/core.py').exists() and all((raw/'models'/p).is_dir() for p in ('survival', 'exploration', 'entrapment')):
                draft = self.out/'upstream'/key/'comparison-draft'
                ready = draft.parent/'comparison-draft-ready.json'
                if not ready.exists():
                    copy_source(self.snapshot(incumbent['snapshot']), draft)
                    for path in (raw/'models').rglob('*'):
                        if path.is_file() and editable(path.relative_to(raw).as_posix()) and path.name not in ('experiment_config.py', 'experimental_policy.py'):
                            destination = draft/path.relative_to(raw)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(path.read_bytes())
                    write(ready, {'commit': review['commit'], 'parent': incumbent['snapshot']})
                snap = self.freeze(draft, f'Lucas modular policy {review["commit"]}')
                if self.checks(snap, key+'.upstream-checks'):
                    candidate = reference(snap, incumbent['config'], baseline=True, label='Lucas modules on trusted harness')
                    report = self.evaluate(key+'.upstream-compare', [incumbent, candidate], self.config['evaluation']['comparison_seeds'])
                    winner, decisions = self.winner(report)
                    detail = dict(decisions=decisions, comparison=str(self.out/'steps'/(key+'.upstream-compare')))
                else:
                    detail = {'reason': 'Upstream modular overlay failed trusted checks; selective review only'}
        write(output, dict(winner=winner, details=detail))
        self.event(key+'.upstream-decision', 'upstream comparison', detail)
        return winner

    def agent(self, key, incumbent, upstream, major):
        folder = self.out/'reviews'/key
        draft = folder/'draft'
        proposal = folder/'proposal.json'
        proposal_only = self.config['agent'].get('proposal_only', False)
        if not (folder/'draft-ready.json').exists():
            copy_source(self.snapshot(incumbent['snapshot']), draft)
            write(folder/'draft-ready.json', {'snapshot': incumbent['snapshot']})
        catalog = self.catalog(incumbent['snapshot'], key+'.catalog')
        if not (folder/'context.json').exists():
            context = dict(round=key, major_boundary=major, incumbent=incumbent,
                proposal_only=proposal_only,
                upstream=upstream, catalog=catalog, max_dimensions=self.config['search']['max_dimensions'],
                journal=self.state['events'],
                development_reports=[str(p) for p in (self.out/'steps').glob('*/comparison.json') if not p.parent.name.startswith('final')],
                development_studies=[str(p) for p in (self.out/'steps').glob('*/study/report.json')],
                seed_panel={name: self.config['evaluation'][name] for name in
                    ('screen_seeds', 'comparison_seeds', 'major_seeds')},
                budget=read(self.out/'budget.json'),
                instruction='Only development evidence may inform edits. Final holdout remains unopened. Historical final assessments are excluded too.')
            write(folder/'context.json', context)
            prompt = (self.out/'research_agent_prompt.md').read_text(encoding='utf-8')
            prompt += '\n\nRead this context file first: ' + str(folder/'context.json')
            prompt += '\nYour editable draft is: ' + str(draft)
            if proposal_only:
                prompt += ('\nThis host uses READ-ONLY PROPOSAL mode. Read files and evidence but do not write '
                    'or execute tests. Return file_edits with exact literal old_text/new_text replacements. '
                    'Use empty old_text only to create a new allowed policy file. Each nonempty old_text '
                    'must match exactly once, with edits applied in order. The supervisor validates and '
                    'applies all edits in a separate draft; never try to bypass the read-only sandbox.')
            (folder/'prompt.md').write_text(prompt, encoding='utf-8')
        def command(_folder, _deadline):
            result = [self.config['agent']['executable']]
            if self.config['agent'].get('legacy_landlock'):
                result += ['--enable', 'use_legacy_landlock']
            result += ['--disable', 'plugins', '--disable', 'apps', '--disable', 'multi_agent',
                      'exec', '--sandbox', 'read-only' if proposal_only else 'workspace-write',
                      '--skip-git-repo-check', '--json', '--output-schema', str(self.out/'research_agent_result.schema.json'),
                      '--output-last-message', str(proposal)]
            if self.config['agent']['model']:
                result += ['--model', self.config['agent']['model']]
            return result + ['-']
        result = self.execute(key+'.agent', self.config['schedule']['agent_seconds'], command,
                              cwd=draft, agent=True, stdin=folder/'prompt.md')
        # Verify every archived policy, not only the currently selected one.
        for saved in (self.out/'snapshots').iterdir():
            self.snapshot(saved.name)
        if result.get('returncode') != 0 or result.get('status') != 'complete' or not proposal.exists():
            return None
        value = read(proposal)
        required = set(read(self.out/'research_agent_result.schema.json')['required'])
        if set(value) != required or type(value['stop_early']) is not bool:
            raise ValueError('Invalid agent response fields')
        if len(value['variants']) > self.config['search']['max_variants'] or len(value['hypotheses']) > 2:
            raise ValueError('Agent exceeded candidate/hypothesis cap')
        for name in ('focus_paths', 'broad_paths'):
            if (len(value[name]) > self.config['search']['max_dimensions']
                    or len(set(value[name])) != len(value[name]) or any(not isinstance(p, str) for p in value[name])):
                raise ValueError('Invalid selected paths')
        variants = []
        for item in value['variants']:
            overrides = {entry['path']: entry['value'] for entry in item['overrides']}
            if len(overrides) != len(item['overrides']):
                raise ValueError('Duplicate override path')
            variants.append(dict(label=item['label'], overrides=overrides))
        value['variants'] = variants
        if proposal_only:
            assert_snapshot(draft, read(self.snapshot(incumbent['snapshot']).parent/'manifest.json'))
            # A fresh application directory makes retry after interrupted writes
            # safe without modifying the reviewed draft or an existing snapshot.
            draft = folder/('applied-'+uuid.uuid4().hex)
            copy_source(self.snapshot(incumbent['snapshot']), draft)
            apply_proposed_edits(draft, value['file_edits'])
        elif value['file_edits'] != []:
            raise ValueError('Direct editing mode requires an empty file_edits array')
        snapshot = self.freeze(draft, key)
        before = read(self.snapshot(incumbent['snapshot']).parent/'manifest.json')
        after = read(self.snapshot(snapshot).parent/'manifest.json')
        changed = changed_policy(before, after)
        patch = []
        for name in changed:
            old_path, new_path = self.snapshot(incumbent['snapshot'])/name, draft/name
            old = old_path.read_text(encoding='utf-8').splitlines(True) if old_path.exists() else []
            new = new_path.read_text(encoding='utf-8').splitlines(True) if new_path.exists() else []
            patch.extend(difflib.unified_diff(old, new, fromfile='before/'+name, tofile='after/'+name))
        (folder/'policy.patch').write_text(''.join(patch), encoding='utf-8')
        self.event(key+'.review', 'major review' if major else 'subgeneration review',
                   dict(proposal=value, changed_files=changed, snapshot=snapshot, patch=str(folder/'policy.patch')))
        return dict(snapshot=snapshot, proposal=value)

    def search(self, key, snapshot, config, plan, major):
        if (self.out/'final-selection.json').exists():
            raise RuntimeError('Final selection is frozen; no more tuning')
        code = self.snapshot(snapshot)
        paths = plan['broad_paths' if major else 'focus_paths'] or self.config['search']['default_paths']
        def command(folder, deadline):
            request = dict(mode='bo' if major else 'focus', starting_config=config,
                  variants=plan['variants'], paths=paths[:self.config['search']['max_dimensions']],
                  max_dimensions=self.config['search']['max_dimensions'], environment=self.environment,
                  policy_seed=self.config['evaluation']['policy_seed'], seeds=self.config['evaluation']['screen_seeds'],
                  search_seed=self.config['search']['seed'], workers=self.workers, deadline=deadline,
                  trials=self.config['search']['bo_trials' if major else 'focused_trials'],
                  case_cache=str(self.out/'evaluation-cache'), diagnostics=self.config.get('diagnostics'),
                  engine=self.config['evaluation'].get('search_engine', 'python'),
                  case_timeout=self.config['evaluation']['case_timeout_seconds'],
                  case_retries=self.config.get('recovery', {}).get('case_retries', 0))
            write(folder/'request.json', request)
            return [sys.executable, '-B', str(code/'scripts/research_study.py'), '--request', str(folder/'request.json'),
                    '--out', str(folder/'study'), '--control', str(folder/'control')]
        self.execute(key+'.search', self.config['schedule']['bo_seconds' if major else 'sub_search_seconds'], command, cwd=code)
        self.snapshot(snapshot)
        path = self.out/'steps'/(key+'.search')/'study'/'best.json'
        study_path = path.with_name('study.json')
        if not path.exists() or not study_path.exists():
            return None
        trials = read(study_path)['trials']
        if not trials or trials[0].get('summary', {}).get('rank') is None:
            return None
        best = read(path)
        return reference(snapshot, best['config'], label=key+(' BO finalist' if major else ' finalist'))

    def round(self, key, incumbent, upstream, major=False):
        path = self.out/'rounds'/(key+'.json')
        if path.exists():
            return read(path)
        winner, decisions, stop = incumbent, [], False
        try:
            prepared = self.agent(key, incumbent, upstream, major)
            if prepared is None:
                self.event(key+'.agent-failed', 'candidate rejected', {'reason': 'Missing/failed agent result'})
                if major:
                    prepared = dict(snapshot=incumbent['snapshot'], proposal=dict(
                        variants=[], focus_paths=[], broad_paths=self.config['search']['default_paths'], stop_early=False))
            if prepared is not None:
                snapshot, plan = prepared['snapshot'], prepared['proposal']
                stop = plan['stop_early']
                if self.checks(snapshot, key+'.checks'):
                    candidate = self.search(key, snapshot, incumbent['config'], plan, major)
                    if candidate:
                        seeds = self.config['evaluation']['major_seeds' if major else 'comparison_seeds']
                        report = self.evaluate(key+'.compare', [incumbent, candidate], seeds)
                        winner, decisions = self.winner(report)
                    else:
                        self.event(key+'.search-incomplete', 'no eligible finalist',
                                   {'study': str(self.out/'steps'/(key+'.search')/'study')})
                else:
                    self.event(key+'.check-failed', 'candidate rejected', {'reason': 'Trusted checks failed'})
        except ValueError as exc:
            # Candidate/proposal incompatibilities are recorded, never promoted.
            self.event(key+'.rejected', 'candidate rejected', {'reason': str(exc)})
        result = dict(before=incumbent, winner=winner, improved=winner['id'] != incumbent['id'],
                      stop_early=stop, decisions=decisions)
        write(path, result)
        self.event(key+'.decision', 'promotion' if result['improved'] else 'retain incumbent', result)
        return result

    def final(self):
        selection = self.out/'final-selection.json'
        if not selection.exists():
            write(selection, dict(selected=self.state['best'], original=self.state['original'],
                  holdout_seeds=self.config['evaluation']['holdout_seeds'], selected_at=time.time()))
        frozen = read(selection)
        # Never make selection depend on holdout results; no agent jobs after this file exists.
        report = self.evaluate('final-holdout', [frozen['original'], frozen['selected']], frozen['holdout_seeds'], final=True)
        write(self.out/'final-report.json', dict(selection=frozen, evaluation=report,
              complete=report['complete'], note='Final assessment only; selected policy is not changed by this result'))
        lines = ['# Final research assessment', '',
                 f'Selected policy: `{frozen["selected"]["id"]}`',
                 f'Complete held-out comparison: **{report["complete"]}**', '',
                 '| Policy | Mean survival | Worst survival | Full horizon | Mean score |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for ref in (frozen['original'], frozen['selected']):
            summary = report['summaries'].get(ref['id'])
            if summary:
                lines.append(f'| {ref["label"]} | {summary["mean_survival"]:.1f} | {summary["worst_survival"]:.1f} | '
                             f'{summary["survived"]}/{summary["cases"]} | {summary["mean_score"]:.2f} |')
            else:
                lines.append(f'| {ref["label"]} | incomplete | — | — | — |')
        lines += ['', 'No policy was promoted using these held-out results.',
                  'The supervisor does not stop provider billing. Stop the provisioned compute separately.']
        (self.out/'final-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        self.state.update(stage='done', status='complete' if report['complete'] else 'inconclusive')
        self.save()

    def exhausted(self, reason='Budget exhausted'):
        """A hard budget stop still produces a usable terminal report."""
        if self.state['best'] and not (self.out/'final-selection.json').exists():
            write(self.out/'final-selection.json', dict(selected=self.state['best'], original=self.state['original'],
                  holdout_seeds=self.config['evaluation']['holdout_seeds'], selected_at=time.time()))
        write(self.out/'final-report.json', dict(complete=False, selected=self.state['best'],
              reason=reason+' before a complete final assessment; holdout evidence is inconclusive'))
        (self.out/'final-report.md').write_text(
            '# Incomplete research assessment\n\n'+reason+' before a complete final assessment.\n'
            'The last supported policy is preserved in best.json when available.\n'
            'No held-out improvement is claimed. Stop provisioned compute separately.\n', encoding='utf-8')
        self.state.update(status='inconclusive', stage='done')
        self.save()

    def run(self):
        if self.state['stage'] == 'done':
            print(f'Campaign finished: {self.state["status"]}. See final-report.md.')
            return
        if self.state['started_at'] is None:
            self.state['started_at'] = time.time()
        self.state['status'] = 'running'
        self.save()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, 'stopping', True))
        try:
            while self.state['stage'] != 'done':
                self.check_stop()
                stage = self.state['stage']
                print(f'Research: generation {self.state["major"]}.{self.state["sub"]}, {stage}', flush=True)
                if (self.out/'final-selection.json').exists():
                    self.state['stage'] = stage = 'final'
                if stage == 'preflight':
                    initial = self.state['initial_snapshot']
                    if not self.checks(initial, 'initial-checks'):
                        raise RuntimeError('Initial trusted checks failed. No experiments started.')
                    catalog = self.catalog(initial, 'initial-catalog')
                    baseline = reference(initial, catalog['defaults'], baseline=True, label='original integrated baseline')
                    control = reference(initial, catalog['defaults'], label='configurable control')
                    report = self.evaluate('initial-controls', [baseline, control], self.config['evaluation']['screen_seeds'])
                    if not report['complete']:
                        raise RuntimeError('Initial controls incomplete/failed; campaign cannot expand')
                    equivalence = self.evaluate('initial-control-equivalence', [baseline, control],
                        self.config['evaluation']['screen_seeds'], control_equivalence=True)
                    if not equivalence['complete']:
                        raise RuntimeError('Control equivalence incomplete/failed; campaign cannot expand')
                    old_rows, new_rows = equivalence['rows'][baseline['id']], equivalence['rows'][control['id']]
                    if any(abs(a['sim_time']-b['sim_time']) > .1 or abs(a['score']-b['score']) > 1.
                           for a, b in zip(old_rows, new_rows)):
                        raise RuntimeError('Baseline/control differ; inspect initial-control-equivalence before further research')
                    self.state.update(original=baseline, best=control, major_start_id=control['id'], stage='opening')
                elif stage in ('opening', 'closing'):
                    key = f'g{self.state["major"]}.{stage}'
                    review = self.upstream(key)
                    best = self.compare_upstream(key, review, self.state['best'])
                    self.state['best'] = best
                    self.state['upstream'] = review
                    if review['status'] == 'fetched':
                        self.state['last_upstream'] = review['commit']
                    self.state['stage'] = 'sub' if stage == 'opening' else 'major'
                elif stage == 'sub':
                    key = f'g{self.state["major"]}.{self.state["sub"]}'
                    result = self.round(key, self.state['best'], self.state.get('upstream', {}))
                    self.state['best'] = result['winner']
                    self.state['stagnant_subs'] = 0 if result['improved'] else self.state['stagnant_subs']+1
                    schedule = self.config['schedule']
                    early = (self.state['sub'] >= schedule['min_subgenerations'] and
                             (self.state['stagnant_subs'] >= schedule['patience'] or result['stop_early']))
                    if self.state['sub'] >= schedule['subgenerations'] or early:
                        self.state['stage'] = 'closing'
                    else:
                        self.state['sub'] += 1
                elif stage == 'major':
                    key = f'g{self.state["major"]}.major'
                    result = self.round(key, self.state['best'], self.state.get('upstream', {}), major=True)
                    self.state['best'] = result['winner']
                    major_improved = self.state['best']['id'] != self.state.get('major_start_id')
                    self.state['stagnant_majors'] = 0 if major_improved else self.state['stagnant_majors']+1
                    require_maximum = self.config['schedule'].get('require_max_generations', False)
                    if (self.state['major'] >= self.config['schedule']['major_generations']
                            or not require_maximum and (self.state['stagnant_majors'] >= self.config['schedule']['patience'] or result['stop_early'])):
                        self.state['stage'] = 'final'
                    else:
                        self.state.update(major=self.state['major']+1, sub=1, stagnant_subs=0, stage='sub',
                                          major_start_id=self.state['best']['id'])
                elif stage == 'final':
                    self.final()
                else:
                    raise ValueError(f'Unknown saved stage: {stage}')
                self.save()
        except (NoBudget, NoStorage) as limit:
            reason = 'Storage limit reached' if isinstance(limit, NoStorage) else 'Budget exhausted'
            self.state['limit_reason'] = reason
            if self.state['best'] and self.state['stage'] != 'final':
                self.state['stage'] = 'final'
                self.save()
                try:
                    self.final()
                except (NoBudget, NoStorage) as final_limit:
                    self.exhausted('Storage limit reached' if isinstance(final_limit, NoStorage) else 'Budget exhausted')
                except Paused:
                    self.state['status'] = 'paused'
                    self.save()
            else:
                self.exhausted(reason)
        except Paused:
            self.state['status'] = 'paused'
            self.save()
        except BaseException as exc:
            self.state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            self.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'stop'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'scripts/research_config.json')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == 'prepare':
        prepare(out, args.config)
    elif args.command == 'status':
        state = read(out/'state.json')
        print(json.dumps({k: state.get(k) for k in ('status', 'stage', 'major', 'sub', 'best', 'agent_calls', 'error')}, indent=2))
    elif args.command == 'stop':
        if not (out/'state.json').exists():
            raise ValueError('Not a prepared campaign')
        (out/'STOP').touch()
        print('Cooperative stop requested. Provider billing is unchanged.')
    else:
        with study_lock(out/'supervisor.lock'):
            if (out/'STOP').exists():
                # An explicit run is an explicit request to resume a stopped campaign.
                (out/'STOP').unlink()
            Supervisor(out).run()


if __name__ == '__main__':
    main()

"""Read-only campaign dashboard. Serve locally; collect over pinned SSH."""
import argparse
import base64
import hashlib
import json
import mimetypes
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


HERE = Path(__file__).resolve().parent
SUMMARY = ('mean_survival', 'worst_survival', 'mean_score', 'cases', 'survived', 'errors')


def read(path, default=None):
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            return {} if default is None else default
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {} if default is None else default


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def confined(path, root):
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def development_seeds(config):
    evaluation = config.get('evaluation', {})
    allowed = set().union(*(evaluation.get(k, []) for k in
                           ('screen_seeds', 'comparison_seeds', 'major_seeds')))
    for panel in evaluation.get('seed_panels', {}).values():
        for name in ('screen_seeds', 'comparison_seeds', 'major_seeds'):
            allowed.update(panel.get(name, []))
    return allowed - set(evaluation.get('holdout_seeds', [])) - set(evaluation.get('retired_holdout_seeds', []))


def development_cases(out, allowed):
    for path in (out / 'evaluation-cache' / 'cases').glob('*/active.json'):
        if not confined(path, out):
            continue
        meta = read(path)
        # Check seed BEFORE opening progress, results, diagnostic files or images.
        if meta.get('seed') not in allowed or not meta.get('attempt'):
            continue
        attempt = Path(meta['attempt'])
        if confined(attempt, out):
            yield path.parent.name, attempt, meta


def collect(out):
    now = time.time()
    state, config, budget, storage = (read(out / name) for name in
        ('state.json', 'config.json', 'budget.json', 'storage.json'))
    if not state:
        raise ValueError('Campaign state is unavailable')
    schedule = dict(config.get('schedule', {}))
    applied_maximum = state.get('runtime_controls', {}).get('schedule', {}).get('major_generations')
    if type(applied_maximum) is int and 1 <= applied_maximum <= 32:
        schedule['major_generations'] = applied_maximum
    allowed = development_seeds(config)
    data = dict(collected_at=now, campaign=out.name,
        state=pick(state, ('status', 'stage', 'major', 'sub', 'started_at', 'agent_calls', 'error')),
        checkpoint_at=mtime(out / 'state.json'), budget=budget,
        limits=pick(config.get('budget', {}), ('total_usd', 'reserve_usd', 'max_hours')),
        schedule=schedule, max_agent_calls=config.get('agent', {}).get('max_calls'),
        storage=pick(storage, ('used_bytes', 'limit_bytes', 'exhausted')),
        best=pick(state.get('best') or {}, ('id', 'label')), best_summary=None,
        holdout='Selection frozen; final results hidden' if (out / 'final-selection.json').exists()
                else 'Unopened', steps=[], comparisons=[], rounds=[], reviews=[], events=[],
        cases=[], active=[], upstream=config.get('upstream', {}).get('branch'))
    data['orchestrator'] = pick(read(out / 'operator-guidance' / 'astra-xhigh.receipt.json'),
        ('model', 'model_reasoning_effort', 'forced_login_method', 'changed_at'))
    throughput = read(out / 'operator-controls' / 'throughput.json')
    dispatch_root = Path(throughput.get('queue', str(out/'absent-dispatch')))
    dispatch_guard = {}
    dispatch_config = {}
    if throughput and confined(dispatch_root, out.parent):
        dispatch_config = read(dispatch_root/'config.json')
        guard_journal = Path(dispatch_config.get('guard_journal', str(dispatch_root/'guard.jsonl')))
        if confined(guard_journal, dispatch_root):
            dispatch_guard = read(guard_journal.with_suffix('.heartbeat.json'))
        data['dispatch'] = dict(workers=[], pending_or_active=0)
        data['dispatch']['compute_hourly_usd'] = .96*len(dispatch_config.get('pods', []))
        metadata = dispatch_config.get('pod_metadata', [])
        if budget.get('expandable_spending') and metadata:
            from datetime import datetime
            ages = [(max(0, now-datetime.fromisoformat(p['createdAt'].replace('Z', '+00:00')).timestamp()),
                     p.get('cost', .96)) for p in metadata]
            budget['coordinator_reserved_estimate_usd'] = budget.get('estimated_compute_usd')
            budget['estimated_compute_usd'] = sum(age*rate/3600 for age, rate in ages) + max(age for age, _ in ages)*.11/3600
            budget['note'] = 'Live fleet estimate from each Pod creation time plus storage allowance; not a provider invoice.'
        for pod in dispatch_config.get('pods', []):
            row = pick(read(dispatch_root/'workers'/f'{pod}.json'), ('id', 'updated_at', 'slots',
                'active', 'legacy_active', 'busy_slots', 'cpu_percent', 'cpu_cores_used', 'stopping', 'pending_or_active'))
            data['dispatch']['workers'].append(row)
            data['dispatch']['pending_or_active'] = max(data['dispatch']['pending_or_active'], row.get('pending_or_active', 0))
        data['limits'].update(total_usd=budget.get('total_limit_usd', 300), max_hours=72)
    data['historical_spend_includes_fleet'] = bool(config.get('recovery', {}).get('parent_campaign'))
    fleet_pointer = read(out / 'operator-controls' / 'active-fleet.json')
    fleet_root = Path(fleet_pointer.get('root', str(out.parent / 'research-capture-fleet')))
    if not confined(fleet_root, out.parent) or not fleet_root.name.startswith('research-capture-fleet'):
        fleet_root = out.parent / 'research-capture-fleet'
    fleet_plan = read(fleet_root / 'fleet.json')
    fleet_guard = read(fleet_root / 'fleet-guard.json')
    fleet_heartbeat = read(fleet_root / 'fleet-guard.heartbeat.json')
    guard_fresh = 0 <= time.time() - fleet_heartbeat.get('updated_at', 0) < 35
    excluded = set(config.get('evaluation', {}).get('holdout_seeds', [])) | set(
        config.get('evaluation', {}).get('retired_holdout_seeds', []))
    data['fleet'] = dict(tasks=[], estimated_compute_usd=0,
        reserved_usd=fleet_plan.get('maximum_auxiliary_compute_usd', 0),
        maximum_total_allocation_usd=fleet_plan.get('worst_case_reserved_usd'))
    if fleet_plan.get('campaign') == str(out):
        for task in fleet_plan.get('tasks', []):
            folder = Path(task['root'])
            if not confined(folder, fleet_root):
                continue
            plan = read(folder / 'plan.json')
            study_seeds = set(plan.get('screening_seeds', []) + plan.get('comparison_seeds', []))
            registered = (fleet_plan.get('version') == 2 and plan.get('campaign') == str(out)
                and task.get('plan_sha256') == hashlib.sha256((folder / 'plan.json').read_bytes()).hexdigest())
            if study_seeds & excluded or (not registered and not study_seeds.issubset(allowed)):
                continue
            study_allowed = study_seeds if registered else allowed
            status = read(folder / 'state.json')
            study = read(folder / 'study' / 'report.json')
            row = dict(name=task['name'], title=task['title'],
                **pick(status, ('status', 'stage', 'updated_at', 'pod_id', 'deadline',
                   'estimated_compute_usd', 'error', 'stop_reason', 'validation_completed', 'validation_total')),
                completed_candidates=study.get('completed', 0), planned_candidates=plan.get('trials'),
                active_cases=[], guard_armed=(guard_fresh and status.get('pod_id') in
                    fleet_heartbeat.get('checked', [])) if registered else (folder / 'launch-receipt.json').exists())
            for case_id, attempt, meta in development_cases(folder, study_allowed):
                if (attempt / 'result.json').exists():
                    continue
                progress = read(attempt / 'progress.json')
                if progress:
                    row['active_cases'].append(dict(id=case_id, updated_at=mtime(attempt / 'progress.json'),
                        **pick(progress, ('seed', 'engine', 'sim_time', 'horizon', 'alive'))))
            data['fleet']['tasks'].append(row)
            data['fleet']['estimated_compute_usd'] += status.get('estimated_compute_usd', 0)
            if study.get('ranking'):
                data['comparisons'].append(dict(name='Parallel / '+task['name']+' (screening)',
                    engine='fastsim', seeds=plan['screening_seeds'], control=False, complete=False,
                    rows=[dict(id=str(r['id']), label=r['label'], **pick(r['summary'], SUMMARY))
                          for r in study['ranking'][:8]]))
            comparison = read(folder / 'comparison.json')
            if comparison.get('complete') and set(comparison.get('seeds', [])).issubset(study_allowed):
                data['comparisons'].append(dict(name='Parallel / '+task['name']+' (Python)',
                    engine='python', seeds=comparison['seeds'], control=False, complete=True,
                    rows=[dict(id=key, label=key, **pick(value, SUMMARY))
                          for key, value in comparison.get('summaries', {}).items()]))
    if dispatch_config:
        background_root = Path(dispatch_config['background_root'])
        if confined(background_root, out.parent):
            background = read(background_root/'fleet.json')
            for task in background.get('tasks', [])[-12:]:
                folder = Path(task['root'])
                if not confined(folder, background_root):
                    continue
                plan = read(folder/'plan.json')
                seeds = set(plan.get('screening_seeds', []) + plan.get('comparison_seeds', []))
                if (plan.get('campaign') != str(out) or seeds & excluded
                        or task.get('plan_sha256') != hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest()):
                    continue
                status = read(folder/'state.json')
                report = read(folder/'study/report.json')
                data['fleet']['tasks'].append(dict(name=task['name'], title=task['name'],
                    **pick(status, ('status', 'stage', 'updated_at', 'deadline', 'error', 'validation_completed', 'validation_total')),
                    pod_id=f"Shared {len(dispatch_config['pods'])}-Pod pool", completed_candidates=report.get('completed', 0),
                    planned_candidates=plan['trials'], active_cases=[], shared=True, guard_armed=False))
                comparison = read(folder/'comparison.json')
                if comparison.get('complete') and set(comparison.get('seeds', [])).issubset(seeds):
                    data['comparisons'].append(dict(name='Background / '+task['name']+' (Python)',
                        engine='python', seeds=comparison['seeds'], control=False, complete=True,
                        rows=[dict(id=key, label=key, **pick(value, SUMMARY))
                              for key, value in comparison.get('summaries', {}).items()]))
    for key, step in state.get('steps', {}).items():
        item = dict(name=key, **pick(step, ('status', 'deadline')))
        if not key.startswith('final'):
            item['progress'] = pick(read(out / 'steps' / key / 'progress.json'), ('completed', 'total'))
            study = read(out / 'steps' / key / 'study' / 'report.json')
            if study:
                item['study'] = dict(**pick(study, ('completed', 'method')),
                    ranking=[dict(**pick(row, ('id', 'label', 'changed')),
                                  summary=pick(row.get('summary', {}), SUMMARY))
                             for row in study.get('ranking', [])[:8]])
                data['comparisons'].append(dict(name=key + ' (screening)',
                    engine=config.get('evaluation', {}).get('search_engine', 'fastsim'),
                    seeds=config.get('evaluation', {}).get('screen_seeds', []),
                    complete=False, control=False,
                    rows=[dict(id=str(row.get('id', '')), label=row.get('label', 'trial'),
                               **pick(row.get('summary', {}), SUMMARY))
                          for row in study.get('ranking', [])[:8]]))
        data['steps'].append(item)
    paths = sorted((out / 'steps').glob('*/comparison.json'), key=mtime, reverse=True)
    for path in paths:
        if path.parent.name.startswith('final') or not confined(path, out):
            continue
        plan = read(path.parent / 'plan.json')
        if plan and not set(plan.get('seeds', [])).issubset(allowed):
            continue
        report = read(path)
        seeds = report.get('seeds', [])
        if not seeds or not set(seeds).issubset(allowed):
            continue
        refs = {r['id']: r.get('label', r['id'][:12]) for r in report.get('refs', [])}
        comparison = dict(name=path.parent.name, seeds=seeds,
            engine=report.get('engine', 'python'), control=report.get('control_equivalence', False),
            complete=report.get('complete', False), rows=[dict(id=key, label=refs.get(key, key[:12]),
                **pick(summary, SUMMARY)) for key, summary in report.get('summaries', {}).items()])
        data['comparisons'].append(comparison)
        if (data['best_summary'] is None and comparison['complete']
                and comparison['engine'] == 'python' and not comparison['control']):
            for row in comparison['rows']:
                if row['id'] == data['best'].get('id'):
                    data['best_summary'] = dict(**row, seeds=len(seeds), comparison=comparison['name'],
                                                evaluated_at=mtime(path))
    for path in sorted((out / 'rounds').glob('g*.json')):
        round_ = read(path)
        data['rounds'].append(dict(name=path.stem, **pick(round_, ('improved', 'stop_early')),
            before=pick(round_.get('before', {}), ('id', 'label')),
            winner=pick(round_.get('winner', {}), ('id', 'label')),
            decisions=[pick(d, ('candidate', 'promote', 'reason', 'mean_survival_delta',
                'worst_survival_delta', 'survival_interval')) for d in round_.get('decisions', [])]))
    for path in sorted((out / 'reviews').glob('g*/proposal.json'), key=mtime, reverse=True):
        proposal = read(path)
        if proposal:
            data['reviews'].append(dict(name=path.parent.name, at=mtime(path),
                **pick(proposal, ('summary', 'evidence', 'hypotheses', 'focus_paths',
                                 'direction', 'complexity', 'upstream', 'stop_early'))))
    for name, event in state.get('events', {}).items():
        # Final-stage details can contain holdout statistics; never export them.
        data['events'].append(dict(name=name, **pick(event, ('kind', 'time'))))
    cases = list(development_cases(out, allowed))
    data['development_cases'] = len(cases)
    completed = []
    for case_id, attempt, meta in cases:
        result_path = attempt / 'result.json'
        if result_path.exists():
            completed.append((case_id, attempt, meta))
            continue
        progress = read(attempt / 'progress.json')
        if progress:
            data['active'].append(dict(id=case_id, updated_at=mtime(attempt / 'progress.json'),
                **pick(progress, ('seed', 'mode', 'engine', 'sim_time', 'horizon', 'score',
                                  'alive', 'wall_seconds'))))
    data['completed_cases'] = len(completed)
    # Bounded payload and disk reads even after thousands of experiments.
    for case_id, attempt, meta in sorted(completed, key=lambda x: mtime(x[1] / 'result.json'), reverse=True)[:64]:
        result = read(attempt / 'result.json')
        diagnostics = result.get('diagnostics') or read(attempt / 'diagnostics' / 'summary.json')
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        data['cases'].append(dict(id=case_id, seed=meta['seed'], at=mtime(attempt / 'result.json'),
            **pick(result, ('mode', 'engine', 'status', 'sim_time', 'score', 'final_agents',
                           'wall_seconds', 'policy_rpc_seconds_per_1000_agent_decisions', 'error')),
            diagnostics=pick(diagnostics, ('totals', 'deaths', 'mean_gross_energy_per_fruit',
                'mean_absorbed_energy_per_fruit', 'absorbed_energy_per_agent_second', 'screenshots'))))
    guard_root = out if (out / 'shutdown.json').exists() else out.parent
    guard_name = 'shutdown' if guard_root == out else 'research-shutdown'
    guard = read(guard_root / (guard_name+'.json'))
    data['watchdog'] = dict(deadline=guard.get('deadline_epoch'), observed=False)
    try:
        pid = None
        for line in (guard_root / (guard_name+'.jsonl')).read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'armed':
                pid = event.get('pid')
        data['watchdog']['observed'] = bool(pid and b'runpod_shutdown_guard.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
    except (OSError, ValueError):
        pass
    if fleet_plan.get('campaign') == str(out) and fleet_guard.get('primary'):
        primary = fleet_guard['primary']
        data['watchdog'] = dict(deadline=primary.get('deadline_epoch'),
            observed=guard_fresh and primary.get('id') in fleet_heartbeat.get('checked', []))
        from datetime import datetime
        created = datetime.fromisoformat(primary['createdAt'].replace('Z', '+00:00')).timestamp()
        data['limits']['max_hours'] = (primary['deadline_epoch'] - created) / 3600
    if dispatch_config:
        fresh = 0 <= time.time()-dispatch_guard.get('updated_at', 0) < 45
        data['watchdog'] = dict(deadline=dispatch_config['deadline_epoch'],
            observed=fresh and set(dispatch_config['pods']).issubset(dispatch_guard.get('checked', [])))
        data['limits']['max_hours'] = 72
        for task in data['fleet']['tasks']:
            task['guard_armed'] = fresh and (task.get('shared') or task.get('pod_id') in dispatch_guard.get('checked', []))
    return data


def collect_image(out, case_id, filename):
    if not re.fullmatch(r'[a-f0-9]{8,64}', case_id) or not re.fullmatch(r'frame-[0-9]+\.png', filename):
        raise ValueError('Invalid image identifier')
    allowed = development_seeds(read(out / 'config.json'))
    for found, attempt, _ in development_cases(out, allowed):
        if found != case_id:
            continue
        folder = attempt / 'diagnostics'
        summary = read(folder / 'summary.json')
        if filename not in summary.get('screenshots', []):
            raise ValueError('Image is not listed in development diagnostics')
        path = folder / filename
        if not confined(path, out) or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('Image is outside the campaign or too large')
        raw = path.read_bytes()
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Invalid PNG')
        return dict(png=base64.b64encode(raw).decode('ascii'))
    raise ValueError('Development case not found')


class Feed:
    def __init__(self, launch, interval):
        self.launch, self.interval = launch, interval
        self.cache = launch / 'dashboard-latest.json'
        self.value = read(self.cache) or dict(data=None, last_success=None)
        self.value.update(connected=False, error='Connecting to the campaign…')
        self.lock = threading.Lock()
        self.images = launch / 'dashboard-images'
        self.images.mkdir(exist_ok=True)

    def remote(self, extra=None):
        state = read(self.launch / 'remote-state.json')
        campaign, host = state.get('campaign', ''), state.get('ssh_host', '')
        if not re.fullmatch(r'/workspace/[a-zA-Z0-9_/-]+', campaign) or not re.fullmatch(r'[a-zA-Z0-9.-]+', host):
            raise ValueError('Invalid campaign SSH configuration')
        command = ['ssh', '-n', '-i', str(self.launch / 'id_ed25519_user'), '-p', str(int(state['ssh_port'])),
            '-o', 'UserKnownHostsFile=' + str(self.launch / 'known_hosts'),
            '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=1', 'root@' + host,
            'python3 /workspace/research-dashboard.py collect --out ' + campaign + (extra or '')]
        result = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace', timeout=35)
        if result.returncode:
            raise RuntimeError('SSH collection unavailable. The last successful report is retained.')
        return json.loads(result.stdout)

    def update(self):
        try:
            data = self.remote()
            value = dict(data=data, last_success=time.time(), connected=True, error=None, interval=self.interval)
            temp = self.cache.with_suffix('.tmp')
            temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temp.replace(self.cache)
            with self.lock:
                self.value = value
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            with self.lock:
                self.value = dict(self.value, connected=False, error=str(error), interval=self.interval)

    def poll(self):
        while True:
            self.update()
            time.sleep(self.interval)

    def snapshot(self):
        with self.lock:
            return dict(self.value)

    def image(self, case_id, filename):
        if not re.fullmatch(r'[a-f0-9]{8,64}', case_id) or not re.fullmatch(r'frame-[0-9]+\.png', filename):
            raise ValueError('Invalid image identifier')
        data = self.snapshot().get('data') or {}
        # Never proxy an arbitrary path or a holdout image supplied by the browser.
        if not any(c['id'] == case_id and filename in c['diagnostics'].get('screenshots', [])
                   for c in data.get('cases', [])):
            raise ValueError('Image is not in this development report')
        path = self.images / (case_id + '-' + filename)
        if path.exists():
            return path.read_bytes()
        raw = base64.b64decode(self.remote(' --case ' + case_id + ' --image ' + filename)['png'], validate=True)
        if len(raw) > 2 * 1024 * 1024 or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Invalid PNG')
        path.write_bytes(raw)
        return raw


def handler(feed, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # A browser cannot use DNS rebinding to turn this into a public proxy.
            if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                self.send_error(403)
                return
            url = urlsplit(self.path)
            download = False
            try:
                if url.path in ('/api/state', '/api/export'):
                    body = json.dumps(feed.snapshot(), ensure_ascii=False, allow_nan=False).encode()
                    mime = 'application/json; charset=utf-8'
                    download = url.path == '/api/export'
                elif url.path == '/api/image':
                    query = parse_qs(url.query)
                    body = feed.image(query.get('case', [''])[0], query.get('file', [''])[0])
                    mime = 'image/png'
                elif url.path in ('/', '/research_dashboard.css', '/research_dashboard.js'):
                    name = 'research_dashboard.html' if url.path == '/' else url.path[1:]
                    body = (HERE / name).read_bytes()
                    mime = (mimetypes.guess_type(name)[0] or 'text/plain') + '; charset=utf-8'
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                self.send_error(503, 'Report or image temporarily unavailable')
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="research-report.json"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionError):
                pass  # A tab closing during a response is normal.

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    collector = sub.add_parser('collect')
    collector.add_argument('--out', type=Path, required=True)
    collector.add_argument('--case')
    collector.add_argument('--image')
    server = sub.add_parser('serve')
    server.add_argument('--launch', type=Path, default=HERE.parent / 'runs' / 'runpod-launch-20260918')
    server.add_argument('--port', type=int, default=8765)
    server.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    if args.mode == 'collect':
        result = collect_image(args.out, args.case or '', args.image) if args.image else collect(args.out)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    if not 5 <= args.interval <= 300 or not 1024 <= args.port <= 65535:
        parser.error('Interval must be 5–300 seconds; port must be 1024–65535')
    feed = Feed(args.launch.resolve(), args.interval)
    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), handler(feed, args.port))
    threading.Thread(target=feed.poll, daemon=True).start()
    print(f'Research dashboard: http://127.0.0.1:{args.port} (read-only; Ctrl+C closes the dashboard)', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()

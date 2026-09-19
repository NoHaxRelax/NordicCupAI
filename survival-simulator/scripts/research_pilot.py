"""Development-only plumbing/throughput pilot; never evaluates campaign holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_support import digest, manifest, research_case, write, copy_source, assert_snapshot
from scripts.optimize_policy import run_case, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--case-timeout', type=float, default=3600.)
    parser.add_argument('--engine', choices=('python', 'fastsim'), default='python')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not 0 < args.seconds <= 3000 or not 1 <= args.workers <= 4 or not 1 <= args.case_timeout <= 7200:
        parser.error('Pilot requires 0 < seconds <= 3000, 1-4 workers and a timeout of 1-7200 seconds')
    if args.out.exists():
        parser.error('Use a new pilot directory')
    args.out.mkdir(parents=True)
    files = manifest(ROOT)
    code = args.out.resolve()/'source'
    copy_source(ROOT, code, files)
    assert_snapshot(code, files)
    write(args.out/'source-manifest.json', files)
    from models.experiment_config import defaults
    config = defaults()
    variant = copy.deepcopy(config)
    variant['features']['late_conservation'] = True
    control = args.out/'control'
    control.mkdir()
    done = threading.Event()
    def heartbeat():
        while not done.is_set():
            (control/'heartbeat').touch()
            done.wait(1)
    (control/'heartbeat').touch()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    requests = []
    source, runtime = digest(files), versions()
    for label, baseline, telemetry, settings in (
            ('baseline-unrecorded', True, False, config),
            ('baseline-recorded', True, True, config),
            ('control-recorded', False, True, config),
            ('variant-recorded', False, True, variant)):
        req = research_case(source, runtime, settings, baseline, 0, 0, {'enabled': telemetry}, args.engine)
        req.update(seconds=args.seconds, pilot=True, label=label)
        req['case_id'] = digest(req)
        requests.append(req)
    write(args.out/'requests.json', requests)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [(r, pool.submit(run_case, r, args.out, control, args.case_timeout, runtime_root=code)) for r in requests]
            results = []
            for request, future in futures:
                result = future.result()
                results.append(dict(result, label=request['label']))
                print(request['label'], result['status'], 'simulation', result.get('sim_time'),
                      'wall', result.get('wall_seconds'), 'error', result.get('error', result.get('diagnostic_error', '')), flush=True)
                write(args.out/'report.json', dict(pilot=True, holdout_used=False,
                      comparable_to_full_horizon=args.seconds == 3000,
                      complete=len(results) == len(requests), results=results))
        if any(r['status'] not in ('horizon', 'extinct') for r in results):
            raise SystemExit('Pilot failed; see report and individual console logs')
        assert_snapshot(code, files)
    finally:
        done.set()
        (control/'STOP').touch()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()

"""Budget-aware local launcher for a user-provisioned Runpod CPU Pod.

This never creates, stops or deletes a Pod. It limits optimizer runtime, not
Runpod billing. Idle Pods and retained storage continue to incur charges.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_case import write_json
from scripts.optimize_policy import study_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/predator-search'))
    parser.add_argument('--budget', type=float, default=25.)
    parser.add_argument('--reserve', type=float, default=5., help='Leave this amount for setup, storage and other costs')
    parser.add_argument('--hourly-rate', type=float, default=1.12, help='Actual whole-Pod compute rate in USD/hour')
    parser.add_argument('--max-hours', type=float, default=9., help='Cumulative optimizer runtime across resumes')
    parser.add_argument('--already-spent', type=float, default=0., help='External spending beyond the reserve, not tracked by this launcher')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    numbers = (args.budget, args.reserve, args.hourly_rate, args.max_hours, args.already_spent)
    if (not all(math.isfinite(n) for n in numbers) or args.hourly_rate <= 0 or args.max_hours <= 0
            or args.reserve < 0 or args.already_spent < 0 or args.budget <= args.reserve
            or args.workers < 0):
        parser.error('Invalid budget, hourly rate, hours or worker count')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with study_lock(args.out/'budget.lock'):
        ledger_path = args.out/'budget.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
            active_seconds=0., estimated_compute_usd=0., sessions=[])
        previous_seconds, previous_cost = ledger['active_seconds'], ledger['estimated_compute_usd']
        remaining_cost = args.budget-args.reserve-args.already_spent-previous_cost
        seconds = min(args.max_hours*3600-previous_seconds, remaining_cost/args.hourly_rate*3600)
        if seconds < 60 and not args.describe:
            raise SystemExit('Budget/runtime allowance exhausted. Results are saved; the Pod is still billed until stopped.')
        command = [sys.executable, '-u', str(ROOT/'scripts/optimize_policy.py'),
                   '--profile', 'runpod', '--out', str(args.out), '--workers', str(args.workers),
                   '--hours', str(max(60., seconds)/3600)]
        if args.describe:
            command.append('--describe')
        print(f"Allowance: {max(0., seconds)/3600:.2f}h; up to "
              f"${max(0., seconds)*args.hourly_rate/3600:.2f} further compute at ${args.hourly_rate:.2f}/h.", flush=True)
        print('This stops the search, not Pod billing. Stop the Pod in Runpod when finished.', flush=True)
        if args.describe:
            raise SystemExit(subprocess.call(command, cwd=ROOT))
        session = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       hourly_rate=args.hourly_rate, allowance_seconds=seconds, status='running')
        ledger['sessions'].append(session)
        ledger.update(budget_usd=args.budget, reserve_usd=args.reserve, external_spend_usd=args.already_spent)
        started = time.monotonic()

        def save():
            elapsed = time.monotonic()-started
            ledger['active_seconds'] = previous_seconds+elapsed
            ledger['estimated_compute_usd'] = previous_cost+elapsed*args.hourly_rate/3600
            session['elapsed_seconds'] = elapsed
            write_json(ledger_path, ledger)

        save()
        env = dict(os.environ, PYTHONHASHSEED='0', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
                   PYGAME_HIDE_SUPPORT_PROMPT='1')
        stopping, sent_stop, next_save = False, None, 0.

        def request_stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        # Install before spawning: background shells may have inherited SIGINT
        # as ignored. The coordinator must receive an interrupt on cancellation.
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=os.name != 'nt')
        try:
            while child.poll() is None:
                now = time.monotonic()
                if now-started >= seconds:
                    stopping = True
                if stopping and sent_stop is None:
                    for control in (args.out/'control').glob('*'):
                        if control.is_dir():
                            (control/'STOP').touch()
                    child.send_signal(signal.SIGINT)
                    sent_stop = now
                if sent_stop is not None and now-sent_stop > 45.:
                    child.kill()
                if now >= next_save:
                    save()
                    next_save = now+15.
                time.sleep(.5)
            session['returncode'] = child.wait()
            session['status'] = 'stopped' if stopping else 'finished'
        finally:
            if child.poll() is None:
                for control in (args.out/'control').glob('*'):
                    if control.is_dir():
                        (control/'STOP').touch()
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            save()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        print(f"Results: {args.out}. STOP THE POD to end compute billing; storage is separate.", flush=True)
        raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()

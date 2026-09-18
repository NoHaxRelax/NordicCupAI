"""Start the endpoint with a chosen configuration, replay the local scene, stop it.

    python run_local_eval.py --detector oracle                      # plumbing: should score close to the placement benchmark
    python run_local_eval.py --weights /path/to/last.pt --device cuda:0 --realtime

Everything after ``--`` goes to local_evaluator.py (for example ``-- --verbose``).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable, help='Interpreter for the server (needs the detector stack)')
    parser.add_argument('--evaluator-python', default=None, help='Interpreter for local_evaluator.py (needs faster-coco-eval); defaults to --python')
    parser.add_argument('--port', type=int, default=9153)
    parser.add_argument('--detector', help='DRONE_DETECTOR value; defaults to ultralytics when --weights is given')
    parser.add_argument('--weights')
    parser.add_argument('--bundle', help='Fixed-asset bundle manifest (DRONE_BUNDLE); sets --detector fixed_assets')
    parser.add_argument('--project', help='Preparation project root with the drone package (DRONE_PROJECT)')
    parser.add_argument('--family-min', default=None, help='JSON of per-family confidence floors for fixed_assets')
    parser.add_argument('--family-levels', default=None, help='JSON of per-family allowed zoom levels for fixed_assets')
    parser.add_argument('--scene', default='helsinki', help='Scene under src/ to replay')
    parser.add_argument('--overview-between-sides', default='1')
    parser.add_argument('--vertical-fraction', default='0')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', default='960')
    parser.add_argument('--conf', default='0.25')
    parser.add_argument('--extent-policy', default='blend')
    parser.add_argument('--detect-every', default='1')
    parser.add_argument('--log-dir', default=str(HERE/'logs'))
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--eval-timeout-s', default=None, help='Lift the evaluator per-request timeout (offline accuracy studies only)')
    parser.add_argument('--simulate-latency-ms', default=None)
    parser.add_argument('evaluator_args', nargs='*')
    args = parser.parse_args()

    env = dict(os.environ, DRONE_PORT=str(args.port), DRONE_DEVICE=args.device, DRONE_IMGSZ=args.imgsz, DRONE_CONF=args.conf,
               DRONE_EXTENT_POLICY=args.extent_policy, DRONE_DETECT_EVERY=args.detect_every, DRONE_LOG_DIR=args.log_dir)
    env['DRONE_OVERVIEW_BETWEEN_SIDES'] = args.overview_between_sides
    env['DRONE_VERTICAL_FRACTION'] = args.vertical_fraction
    if args.weights:
        env['DRONE_WEIGHTS'] = args.weights
    if args.bundle:
        env['DRONE_BUNDLE'] = args.bundle; env['DRONE_DETECTOR'] = 'fixed_assets'
        if args.project:
            env['DRONE_PROJECT'] = args.project
        if args.family_min:
            env['DRONE_FAMILY_MIN'] = args.family_min
        if args.family_levels:
            env['DRONE_FAMILY_LEVELS'] = args.family_levels
    if args.detector:
        env['DRONE_DETECTOR'] = args.detector
    server = subprocess.Popen([args.python, 'api.py'], cwd=HERE, env=env)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise SystemExit('Server exited during startup')
            try:
                requests.get(f'http://127.0.0.1:{args.port}/', timeout=1)
                break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise SystemExit('Server did not come up')
        command = [args.evaluator_python or args.python, 'local_evaluator.py', '--url', f'http://127.0.0.1:{args.port}/predict',
                   '--scene', args.scene]
        if args.realtime:
            command.append('--realtime')
        if args.simulate_latency_ms:
            command += ['--simulate-latency-ms', args.simulate_latency_ms]
        command += args.evaluator_args
        if args.eval_timeout_s:
            env['DRONE_EVAL_TIMEOUT_S'] = str(args.eval_timeout_s)
        return subprocess.call(command, cwd=HERE, env=env)
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == '__main__':
    raise SystemExit(main())

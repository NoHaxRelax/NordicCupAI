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
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', default='960')
    parser.add_argument('--conf', default='0.25')
    parser.add_argument('--extent-policy', default='blend')
    parser.add_argument('--detect-every', default='1')
    parser.add_argument('--log-dir', default=str(HERE/'logs'))
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--simulate-latency-ms', default=None)
    parser.add_argument('evaluator_args', nargs='*')
    args = parser.parse_args()

    env = dict(os.environ, DRONE_PORT=str(args.port), DRONE_DEVICE=args.device, DRONE_IMGSZ=args.imgsz, DRONE_CONF=args.conf,
               DRONE_EXTENT_POLICY=args.extent_policy, DRONE_DETECT_EVERY=args.detect_every, DRONE_LOG_DIR=args.log_dir)
    if args.weights:
        env['DRONE_WEIGHTS'] = args.weights
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
        command = [args.evaluator_python or args.python, 'local_evaluator.py', '--url', f'http://127.0.0.1:{args.port}/predict']
        if args.realtime:
            command.append('--realtime')
        if args.simulate_latency_ms:
            command += ['--simulate-latency-ms', args.simulate_latency_ms]
        command += args.evaluator_args
        return subprocess.call(command, cwd=HERE)
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == '__main__':
    raise SystemExit(main())
